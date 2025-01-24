# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified from DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
from typing import Optional

import torch
from torch import Tensor, nn

from models.helpers import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
                            get_clones)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        xyz_inds = None

        return xyz, output, xyz_inds


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm_fn_name="ln",
                return_intermediate=False,
                weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, tgt, memory,
                image_features_clip = None,
                text_features_clip = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                transpose_swap: Optional [bool] = False,
                return_attn_weights: Optional [bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1) # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt

        intermediate = []
        attns = []

        for layer in self.layers:
            output, attn = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_attn_weights=return_attn_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns

        return output, attns


class MaskedTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, masking_radius, interim_downsampling,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__(encoder_layer, num_layers, norm=norm, weight_init_name=weight_init_name)
        assert len(masking_radius) == num_layers
        self.masking_radius = masking_radius
        self.interim_downsampling = interim_downsampling

    def compute_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):

        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)

        output = src
        xyz_dist = None
        xyz_inds = None

        for idx, layer in enumerate(self.layers):
            mask = None
            if self.masking_radius[idx] > 0:
                mask, xyz_dist = self.compute_mask(xyz, self.masking_radius[idx], xyz_dist)
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)

            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

            if idx == 0 and self.interim_downsampling:
                # output is npoints x batch x channel. make batch x channel x npoints
                output = output.permute(1, 2, 0)
                xyz, output, xyz_inds = self.interim_downsampling(xyz, output)
                # swap back
                output = output.permute(2, 0, 1)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        return xyz, output, xyz_inds
    
    def extra_repr(self):
        radius_str = ", ".join(["%.2f"%(x) for x in self.masking_radius])
        return f"masking_radius={radius_str}"


class TransformerCrossEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers,
                 norm=None, weight_init_name="xavier_uniform", nqueries=256):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)
        self.nqueries = nqueries

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, pc_feat=None, image_clip_feat=None, text_clip_feat=None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                ):
        # image_clip_feat = image_clip_feat.unsqueeze(1).repeat(1, self.nqueries, 1)
        if text_clip_feat is not None:
            text_clip_feat = text_clip_feat.unsqueeze(0).repeat(image_clip_feat.shape[0], 1, 1)
            src = torch.cat([image_clip_feat, text_clip_feat], dim=1)
        else:
            src = image_clip_feat
        src = src.permute(1, 0, 2)
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)
            output, pc_feat = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, pc_feat=pc_feat)

        if self.norm is not None:
            pc_feat = self.norm(pc_feat)

        if transpose_swap:
            pc_feat = pc_feat.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        xyz_inds = None

        return xyz, pc_feat, xyz_inds

class TransformerEncoderEveryLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=128,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=False)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm1 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=False)

        if self.use_ffn:
            # Implementation of Feedforward model
            self.cross_linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.cross_dropout = nn.Dropout(dropout, inplace=False)
            self.cross_linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.cross_norm1 = NORM_DICT[norm_name](d_model)
            self.cross_norm2 = NORM_DICT[norm_name](d_model)
            self.cross_dropout2 = nn.Dropout(dropout, inplace=False)

        # self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)

        self.activation = ACTIVATION_DICT[activation]()

        # self.cross_norm1 = NORM_DICT[norm_name](d_model)
        self.cross_dropout1 = nn.Dropout(dropout, inplace=False)

        self.cross_activation = ACTIVATION_DICT[activation]()

        self.normalize_before = normalize_before
        self.nhead = nhead

    # def cross_attention(self, query: torch.Tensor, key=None, value=None):
    #     res = self.cross_attn(query, key, value, need_weights=False, attn_mask=self.attn_mask)
    #     return res

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # def forward_post(self,
    #                  src,
    #                  src_mask: Optional[Tensor] = None,
    #                  src_key_padding_mask: Optional[Tensor] = None,
    #                  pos: Optional[Tensor] = None, pc_feat = None):
    #     q = k = self.with_pos_embed(src, pos)
    #     value = src
    #     src2 = self.self_attn(q, k, value=value, attn_mask=src_mask,
    #                           key_padding_mask=src_key_padding_mask)[0]
    #     src = src + self.dropout1(src2)
    #     if self.use_norm_fn_on_input:
    #         src = self.norm1(src)
    #     if self.use_ffn:
    #         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    #         src = src + self.dropout2(src2)
    #         src = self.norm2(src)
    #
    #     q = pc_feat
    #     k = v = src
    #     cross_src = pc_feat
    #     cross_src2 = self.cross_attn(q, k, value=v, attn_mask=src_mask,
    #                           key_padding_mask=src_key_padding_mask)[0]
    #     cross_src = cross_src + self.cross_dropout1(cross_src2)
    #     if self.use_norm_fn_on_input:
    #         cross_src = self.cross_norm1(cross_src)
    #     if self.use_ffn:
    #         cross_src2 = self.cross_linear2(self.cross_dropout(self.cross_activation(self.cross_linear1(cross_src))))
    #         cross_src = cross_src + self.cross_dropout2(cross_src2)
    #         cross_src = self.cross_norm2(cross_src)
    #
    #     return src, cross_src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [Tensor] = False, pc_feat = None):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=value, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)

        q = pc_feat
        k = v = src[:197, :, :]
        cross_src = self.cross_norm1(pc_feat)
        cross_src2 = self.cross_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        cross_src = cross_src + self.cross_dropout1(cross_src2)
        # if self.use_norm_fn_on_input:
        #     cross_src = self.cross_norm1(cross_src)
        if self.use_ffn:
            cross_src2 = self.cross_norm2(cross_src)
            cross_src2 = self.cross_linear2(self.cross_dropout(self.cross_activation(self.cross_linear1(cross_src2))))
            cross_src = cross_src + self.cross_dropout2(cross_src2)

        if return_attn_weights:
            return src, attn_weights
        return src, cross_src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False,
                pc_feat=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights, pc_feat=pc_feat)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, pc_feat=pc_feat)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=128,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=False)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=False)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.use_norm_fn_on_input:
            src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [Tensor] = False):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=value, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     text_features_clip = None,
                     image_features_clip = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional [bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_im_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_im = nn.Linear(dim, dim)
        self.proj_drop_im = nn.Dropout(proj_drop)

    def forward(self, q=None, k=None, v=None, q_img=None, v_img=None, attn_mask=None, key_padding_mask=None):
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        q = self.q_linear(q) # (B, N, C)
        k = self.k_linear(k) # (B, N, C)
        v = self.v_linear(v) # (B, N, C)
        B_q, N_q, C_q = q.shape
        B_k, N_k, C_k = k.shape
        q = q.reshape(B_q, N_q, self.num_heads, C_q//self.num_heads) #(B_q, self.num_heads, N_q, C_q//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        k = k.permute(0, 2, 1, 3)
        v = v.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C_q)

        x = self.proj(x)
        x = self.proj_drop(x)

        attn_im = attn # .clone().detach()
        v_img = v_img.permute(1, 0, 2)
        v_img = self.v_im_linear(v_img)
        v_img = v_img.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        v_img = v_img.permute(0, 2, 1, 3)
        x_im = (attn_im @ v_img).transpose(1, 2).reshape(B_q, N_q, C_q)
        x_im = self.proj_im(x_im)
        x_im = self.proj_drop_im(x_im)
        return x, x_im

class TransformerSharedAttentionDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln"):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Attention(d_model, nhead, qkv_bias=True, attn_drop=dropout)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)
        # self.norm2_im = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.norm3_im = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout2_im = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.dropout3_im = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1_im = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.dropout_im = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear2_im = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     text_features_clip = None,
                     image_features_clip = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional [bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt=None, tgt_img=None, memory=None, memory_im=None,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # tgt_img2 = self.norm2_im(tgt_img)
        tgt2, tgt_img2 = self.multihead_attn(q=self.with_pos_embed(tgt2, query_pos),
                                   k=self.with_pos_embed(memory, pos),
                                   v=memory, q_img=None, v_img=memory_im, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2 = tgt2.permute(1, 0, 2)
        tgt_img2 = tgt_img2.permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        tgt_img = tgt_img + self.dropout2_im(tgt_img2)
        tgt_img2 = self.norm3_im(tgt_img)
        tgt_img2 = self.linear2_im(self.dropout_im(self.activation(self.linear1_im(tgt_img2))))
        tgt_img = tgt_img + self.dropout3_im(tgt_img2)
        return tgt, tgt_img

    def forward(self, tgt=None, tgt_img=None, memory=None, memory_im=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt=tgt, tgt_img=tgt_img, memory=memory, memory_im=memory_im, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos,  return_attn_weights=return_attn_weights)
        return self.forward_post(tgt, tgt_img, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)

class TransformerSharedAttentionDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm_fn_name="ln",
                return_intermediate=False,
                weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = None
        if norm_fn_name is not None:
            self.norm = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
            self.norm_im = NORM_DICT[norm_fn_name](self.layers[0].linear2.out_features)
        self.return_intermediate = return_intermediate
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, tgt=None, tgt_img=None, memory=None, memory_im=None,
                image_features_clip = None,
                text_features_clip = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                transpose_swap: Optional [bool] = False,
                return_attn_weights: Optional [bool] = False,
                ):
        if transpose_swap:
            bs, c, h, w = memory.shape
            memory = memory.flatten(2).permute(2, 0, 1) # memory: bs, c, t -> t, b, c
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = tgt
        output_img = tgt_img
        intermediate = []
        attns = []
        intermediate_im = []

        for layer in self.layers:
            output, output_img = layer(tgt=output, tgt_img=output_img, memory=memory, memory_im=memory_im,tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos,
                           return_attn_weights=return_attn_weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_im.append(self.norm_im(output_img))
            if return_attn_weights:
                attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)
            output_img = self.norm_im(output_img)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                intermediate_im.pop()
                intermediate_im.append(output_img)

        if return_attn_weights:
            attns = torch.stack(attns)

        if self.return_intermediate:
            return torch.stack(intermediate), attns, torch.stack(intermediate_im)

        return output, attns, output_img


class GuidenceAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., if_detach_the_guidence_attention=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_im_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_im_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_im_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_im = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_im = nn.Linear(dim, dim)
        self.proj_drop_im = nn.Dropout(proj_drop)
        self.if_detach_the_guidence_attention = if_detach_the_guidence_attention
        print('==========if_detach_the_guidence_attention:')
        print(self.if_detach_the_guidence_attention)

    def forward(self, q=None, k=None, v=None, q_img=None, k_img=None, v_img=None, attn_mask=None, key_padding_mask=None):
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        q = self.q_linear(q) # (B, N, C)
        k = self.k_linear(k) # (B, N, C)
        v = self.v_linear(v) # (B, N, C)
        B_q, N_q, C_q = q.shape
        B_k, N_k, C_k = k.shape
        q = q.reshape(B_q, N_q, self.num_heads, C_q//self.num_heads) #(B_q, self.num_heads, N_q, C_q//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        k = k.permute(0, 2, 1, 3)
        v = v.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C_q)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.if_detach_the_guidence_attention:
            attn_guidence = attn.clone().detach() # .clone().detach()
        else:
            attn_guidence = attn

        q_img = q_img.permute(1, 0, 2) # (1, 128, 256)
        k_img = k_img.permute(1, 0, 2)
        v_img = v_img.permute(1, 0, 2)
        q_img = self.q_im_linear(q_img)  # (B, N, C)
        k_img = self.k_im_linear(k_img)  # (1, 128, 256)
        v_img = self.v_im_linear(v_img)
        q_img = q_img.reshape(B_q, N_q, self.num_heads, C_q//self.num_heads) #(B_q, self.num_heads, N_q, C_q//self.num_heads)
        q_img = q_img.permute(0, 2, 1, 3) # (1, 4, 128, 64)
        k_img = k_img.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        k_img = k_img.permute(0, 2, 1, 3) # (1, 4, 128, 64)
        v_img = v_img.reshape(B_k, N_k, self.num_heads, C_k//self.num_heads)
        v_img = v_img.permute(0, 2, 1, 3) # (1, 4, 128, 64)

        attn_img = (q_img @ k_img.transpose(-2, -1)) * self.scale
        attn_img = attn_img * attn_guidence
        attn_img = attn_img.softmax(dim=-1)
        attn_img = self.attn_drop_im(attn_img)
        # x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, C_q)


        x_im = (attn_img @ v_img).transpose(1, 2).reshape(B_q, N_q, C_q)
        x_im = self.proj_im(x_im)
        x_im = self.proj_drop_im(x_im)
        return x, x_im


class TransformerGuidenceSharedAttentionDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=256,
                 dropout=0.1, dropout_attn=None,
                 activation="relu", normalize_before=True,
                 norm_fn_name="ln", if_detach_the_guidence_attention=False):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_im = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = GuidenceAttention(d_model, nhead, qkv_bias=True, attn_drop=dropout, if_detach_the_guidence_attention=if_detach_the_guidence_attention)

        self.norm1 = NORM_DICT[norm_fn_name](d_model)
        self.norm1_im = NORM_DICT[norm_fn_name](d_model)
        self.norm2 = NORM_DICT[norm_fn_name](d_model)
        self.norm2_im = NORM_DICT[norm_fn_name](d_model)

        self.norm3 = NORM_DICT[norm_fn_name](d_model)
        self.norm3_im = NORM_DICT[norm_fn_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=False)
        self.dropout1_im = nn.Dropout(dropout, inplace=False)
        self.dropout2 = nn.Dropout(dropout, inplace=False)
        self.dropout2_im = nn.Dropout(dropout, inplace=False)
        self.dropout3 = nn.Dropout(dropout, inplace=False)
        self.dropout3_im = nn.Dropout(dropout, inplace=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear1_im = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.dropout_im = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear2_im = nn.Linear(dim_feedforward, d_model)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     text_features_clip = None,
                     image_features_clip = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     return_attn_weights: Optional [bool] = False):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if return_attn_weights:
            return tgt, attn
        return tgt, None

    def forward_pre(self, tgt=None, tgt_img=None, memory=None, memory_im=None,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [bool] = False):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt_img2 = self.norm1_im(tgt_img)
        tgt_img2 = self.self_attn_im(tgt_img2, tgt_img2, value=tgt_img2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt_img = tgt_img + self.dropout1_im(tgt_img2)
        tgt2 = self.norm2(tgt)
        tgt_img2 = self.norm2_im(tgt_img)
        tgt2, tgt_img2 = self.multihead_attn(q=self.with_pos_embed(tgt2, query_pos),
                                   k=self.with_pos_embed(memory, pos),
                                   v=memory, q_img=tgt_img2, k_img=memory_im, v_img=memory_im, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2 = tgt2.permute(1, 0, 2)
        tgt_img2 = tgt_img2.permute(1, 0, 2)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        tgt_img = tgt_img + self.dropout2_im(tgt_img2)
        tgt_img2 = self.norm3_im(tgt_img)
        tgt_img2 = self.linear2_im(self.dropout_im(self.activation(self.linear1_im(tgt_img2))))
        tgt_img = tgt_img + self.dropout3_im(tgt_img2)
        return tgt, tgt_img

    def forward(self, tgt=None, tgt_img=None, memory=None, memory_im=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                return_attn_weights: Optional [bool] = False):
        if self.normalize_before:
            return self.forward_pre(tgt=tgt, tgt_img=tgt_img, memory=memory, memory_im=memory_im, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos,  return_attn_weights=return_attn_weights)
        return self.forward_post(tgt, tgt_img, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, return_attn_weights)