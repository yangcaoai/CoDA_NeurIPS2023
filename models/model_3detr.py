# Copyright (c) Facebook, Inc. and its affiliates.
# import random
import time
from collections import OrderedDict
import math
import os.path
from functools import partial
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from third_party_pointnet2.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party_pointnet2.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points
from torchvision.transforms import Resize
from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer, TransformerEncoderEveryLayer, TransformerCrossEncoder, TransformerSharedAttentionDecoderLayer, TransformerSharedAttentionDecoder, TransformerGuidenceSharedAttentionDecoderLayer)
from CLIP.clip import clip
from CLIP.clip.model import VisionTransformerCombine, VisionTransformerCombineLayer, CrossTransformer, Transformer, Transformer_return_attention, CrossTransformer_return_attention, ModifiedResNet
import cv2
from PIL import Image
import torchvision
import datasets.sunrgbd_utils as sunrgbd_utils
import datasets.scannet_utils as scannet_utils
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from models.vision_transformer import _create_vision_transformer, _create_multi_modal_vision_transformer, _create_two_modal_vision_transformer
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from models.resnet import resnet50
# from datasets.sunrgbd_utils import project_3dpoint_to_2dpoint_tensor, project_3dpoint_to_2dpoint_corners_tensor
from utils.box_util import box3d_iou
# from pytorch3d.ops import box3d_overlap
import cv2
from typing import Tuple, Union
# import torch.nn.functional as F
from torch.distributions import Categorical

# ALL_CLASS_PATH = "datasets/all_classes_trainval_v2_del_nonval_classes.npy"
ALL_CLASS_PATH_V1 = "datasets/all_classes_trainval_v1.npy"
ALL_CLASS_PATH_V2 = "datasets/all_classes_trainval_v2_revised_del_val_less_than_5_classes.npy"
ALL_CLASS_PATH_SCANNET = 'datasets/scannet_200_classname_no_wall_floor.npy'
ALL_CMP_CLASS_PATH_SCANNET= "datasets/ov_3detr_scannet.npy"
ALL_CMP_CLASS_PATH = "datasets/ov_3detr.npy"
# ALL_CLASS_PATH = "datasets/all_classes_trainval_v2_revised_del_val_less_than_5_classes_try_some_bg.npy"
ALL_SUPERCLASS_PATH = 'datasets/lvis_1204.npy'

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(
            center_unnormalized, src_range=point_cloud_dims
        )
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1: # do not learn angle when there is only 1 classes
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        # assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1 or cls_logits.shape[-1] == 2 or cls_logits.shape[-1] == 38 or cls_logits.shape[-1] == 3
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def compute_objectness_and_cls_prob_sigmoid(self, cls_logits):
        # assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1 or cls_logits.shape[-1] == 2 or cls_logits.shape[-1] == 38 or cls_logits.shape[-1] == 233
        # cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        cls_prob = torch.sigmoid(cls_logits)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )
    def box_parametrization_to_corners_np(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners_np(
            box_center_unnorm, box_size_unnorm, box_angle
        )
    def box_parametrization_to_corners_xyz(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners_xyz(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETRPredictedBoxDistillationHead(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
            self,
            pre_encoder,
            encoder,
            decoder,
            dataset_config,
            image_text_encoder=None,
            encoder_dim=256,
            decoder_dim=256,
            position_embedding="fourier",
            mlp_dropout=0.3,
            num_queries=256,
            if_with_clip=False,
            if_use_gt_box=False,
            if_expand_box=False,
            if_with_clip_embed=False,
            if_with_clip_train=True,
            num_cls_predict=1,
            if_with_fake_classes=False,
            pooling_methods='average',
            if_clip_more_prompts=False,
            if_keep_box=False,
            if_select_box_by_objectness=False,
            keep_objectness=0.5,
            online_nms_update_novel_label=False,
            online_nms_update_accumulate_novel_label=False,
            online_nms_update_accumulate_epoch=10,
            distillation_box_num=32,
            args=None,
    ):
        super().__init__()
        self.if_with_fake_classes = if_with_fake_classes
        print('============================if with fake classes:')
        print(self.if_with_fake_classes)
        self.num_cls_predict = num_cls_predict
        # print('==================Predict cls number in 3DETR head:')
        # print(self.num_cls_predict)
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.args = args
        self.if_second_text = False
        self.if_with_clip = if_with_clip
        self.if_clip_more_prompts = if_clip_more_prompts
        self.if_with_clip_train = if_with_clip_train
        self.box_idx_list = np.arange(128, dtype=np.int8)
        if self.if_with_clip_train:
            print('=============Adopt CLIP model to train the model=============')
            if args.dataset_name.find('scannet') != -1:
                class_id_to_idx = {2: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 13: 9, 14: 10, 15: 11,
                   16: 12,
                   17: 13, 18: 14, 19: 15, 21: 16, 22: 17, 23: 18, 24: 19, 26: 20, 27: 21, 28: 22,
                   29: 23,
                   31: 24, 32: 25, 33: 26, 34: 27, 35: 28, 36: 29, 38: 30, 39: 31, 40: 32, 41: 33,
                   42: 34,
                   44: 35, 45: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 54: 44,
                   55: 45,
                   56: 46, 57: 47, 58: 48, 59: 49, 62: 50, 63: 51, 64: 52, 65: 53, 66: 54, 67: 55,
                   68: 56,
                   69: 57, 70: 58, 71: 59, 72: 60, 73: 61, 74: 62, 75: 63, 76: 64, 77: 65, 78: 66,
                   79: 67,
                   80: 68, 82: 69, 84: 70, 86: 71, 87: 72, 88: 73, 89: 74, 90: 75, 93: 76, 95: 77,
                   96: 78,
                   97: 79, 98: 80, 99: 81, 100: 82, 101: 83, 102: 84, 103: 85, 104: 86, 105: 87,
                   106: 88,
                   107: 89, 110: 90, 112: 91, 115: 92, 116: 93, 118: 94, 120: 95, 121: 96, 122: 97,
                   125: 98, 128: 99, 130: 100, 131: 101, 132: 102, 134: 103, 136: 104, 138: 105,
                   139: 106,
                   140: 107, 141: 108, 145: 109, 148: 110, 154: 111, 155: 112, 156: 113, 157: 114,
                   159: 115, 161: 116, 163: 117, 165: 118, 166: 119, 168: 120, 169: 121, 170: 122,
                   177: 123, 180: 124, 185: 125, 188: 126, 191: 127, 193: 128, 195: 129, 202: 130,
                   208: 131, 213: 132, 214: 133, 221: 134, 229: 135, 230: 136, 232: 137, 233: 138,
                   242: 139, 250: 140, 261: 141, 264: 142, 276: 143, 283: 144, 286: 145, 300: 146,
                   304: 147, 312: 148, 323: 149, 325: 150, 331: 151, 342: 152, 356: 153, 370: 154,
                   392: 155, 395: 156, 399: 157, 408: 158, 417: 159, 488: 160, 540: 161, 562: 162,
                   570: 163, 572: 164, 581: 165, 609: 166, 748: 167, 776: 168, 1156: 169, 1163: 170,
                   1164: 171, 1165: 172, 1166: 173, 1167: 174, 1168: 175, 1169: 176, 1170: 177,
                   1171: 178,
                   1172: 179, 1173: 180, 1174: 181, 1175: 182, 1176: 183, 1178: 184, 1179: 185,
                   1180: 186,
                   1181: 187, 1182: 188, 1183: 189, 1184: 190, 1185: 191, 1186: 192, 1187: 193,
                   1188: 194,
                   1189: 195, 1190: 196, 1191: 197}
                train_idx_list = [class_id_to_idx[i] for i in args.train_range_list]
                tmp_test_range_list = args.train_range_list.copy()
                tmp_cnt = 0
                for tmp_class_id in args.test_range_list:
                    if tmp_class_id in args.train_range_list:
                        continue
                    tmp_test_range_list.append(tmp_class_id)
                    tmp_cnt += 1
                    if tmp_cnt >= args.reset_scannet_num:
                        break
                test_class_list = sorted(tmp_test_range_list)
                self.seen_idx = []
                for test_item_idx, test_item in enumerate(test_class_list):
                    if test_item in args.train_range_list:
                        self.seen_idx.append(test_item_idx)
                test_idx_list = [class_id_to_idx[i] for i in test_class_list]
                all_classes_keys = np.load(ALL_CLASS_PATH_SCANNET)
                all_cmp_classes = np.load(
                    ALL_CMP_CLASS_PATH_SCANNET)  # [:args.test_range_max]all_classes_keys # waiting to change this part
                select_class_path = ALL_CLASS_PATH_SCANNET
                if self.if_clip_more_prompts:
                    # all_classes_keys = list(all_classes.keys())[:43]
                    all_classes_keys = all_classes_keys[test_idx_list]
                else:
                    all_classes_keys = all_classes_keys[train_idx_list]  # [:dataset_config.num_semcls] just seen classes during training.
            else:
                if self.args.if_use_v1:
                    all_classes = np.load(ALL_CLASS_PATH_V1, allow_pickle=True).item()
                    select_class_path = ALL_CLASS_PATH_V1
                else:
                    all_classes = np.load(ALL_CLASS_PATH_V2, allow_pickle=True).item()
                    select_class_path = ALL_CLASS_PATH_V2
                all_cmp_classes = np.load(ALL_CMP_CLASS_PATH, allow_pickle=True)
                if self.if_clip_more_prompts:
                    # all_classes_keys = list(all_classes.keys())[:43]
                    all_classes_keys = list(all_classes.keys())[:args.test_range_max]
                else:
                    all_classes_keys = list(all_classes.keys())[:args.train_range_max]  # [:dataset_config.num_semcls] just seen classes during training.

            print('-------------------load class prompt from ----------------------')
            print(select_class_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.train_range_max = args.train_range_max
            self.test_range_max = args.test_range_max
            print(len(all_classes_keys))
            # all_classes_keys.append('wall')
            self.all_classes_keys = ['a photo of a ' + i.replace('_', ' ').lower() + ' in the scene' for i in
                                     all_classes_keys]
            
            self.if_clip_superset = args.if_clip_superset
            print('====================if clip superset====================')
            print(self.if_clip_superset)
            if self.if_clip_superset:
                superset_all_classes = np.load(ALL_SUPERCLASS_PATH, allow_pickle=True)
                superset_all_classes_keys = list(superset_all_classes)[1:]  # the  first one is "name"

                tmp_superset = ['a photo of a ' + i.replace('_', ' ').lower() + ' in the scene' for i in
                                superset_all_classes_keys]

                self.superset_all_classes_keys = []

                cnt_in = 0
                if args.dataset_name.find('scannet') != -1:
                    for idx_class in self.seen_idx:
                        i = self.all_classes_keys[idx_class] # not wrong for scannet
                    # for i in self.all_classes_keys[:10]: # only the seen classes
                        if i not in self.superset_all_classes_keys:
                        # print(i)
                            self.superset_all_classes_keys.append(i)
                            cnt_in += 1
                else:
                    for i in self.all_classes_keys[:10]: # only the seen classes
                        if i not in self.superset_all_classes_keys:
                        # print(i)
                            self.superset_all_classes_keys.append(i)
                            cnt_in += 1

                cnt_same = 0
                for i in tmp_superset:
                #     # text_item = 'a photo of a ' + i.replace('_', ' ').lower() + ' in the scene'
                    if i not in self.superset_all_classes_keys:
                    # print(i)
                        self.superset_all_classes_keys.append(i)
                    else:
                        cnt_same += 1

                print('=====================class superset======================')
                #save_dic = {item: idx for idx, item in enumerate(self.superset_all_classes_keys)}
                assert len(self.superset_all_classes_keys) == 1201 or len(self.superset_all_classes_keys) == 1216 or len(self.superset_all_classes_keys) == 232 or len(self.superset_all_classes_keys) == 1203
                print(self.superset_all_classes_keys)
                print(len(self.superset_all_classes_keys))
            
            self.clip_model, self.preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt",
                                                                    # "./CLIP/pretrain_models/ViT-L-14.pt", #"./CLIP/pretrain_models/RN50x64.pt",
                                                                    device=self.device,
                                                                    download_root='./CLIP/pretrain_models/',
                                                                    if_transform_tensor=True)
            for name, param in self.clip_model.named_parameters():
                param.requires_grad = False

            self.res_encoder = self.clip_model.visual

            self.resnet_transform = self.preprocess_for_tensor  #
            print('===============Text prompts for contrastive loss==============')
            print(len(self.all_classes_keys))
            print(self.all_classes_keys)
            self.text = clip.tokenize(self.all_classes_keys).to(self.device)
            self.text_features_fg = self.clip_model.encode_text(self.text).to(torch.float32)
            self.text_features_fg_norm = self.text_features_fg / self.text_features_fg.norm(dim=1, keepdim=True)
            self.text_features_fg_norm = self.text_features_fg_norm.to(torch.float32)

            all_cmp_classes = np.load(ALL_CMP_CLASS_PATH, allow_pickle=True)
            all_cmp_classes_keys = list(all_cmp_classes)
            print('========compare with cvpr paper=========')
            print(len(all_cmp_classes_keys))
            print(all_cmp_classes_keys)
            self.all_cmp_classes_keys = ['a photo of a ' + i.replace('_', ' ').lower() + ' in the scene' for i in
                                         all_cmp_classes_keys]
            self.cmp_text = clip.tokenize(self.all_cmp_classes_keys).to(self.device)
            # self.text_features_bg = torch.from_numpy(np.load('bg_embedding.npy')).to(self.device).unsqueeze(0)
            self.cmp_text_features_fg = self.clip_model.encode_text(self.cmp_text)
            self.cmp_text_features_fg_norm = self.cmp_text_features_fg / self.cmp_text_features_fg.norm(dim=1, keepdim=True)

            if self.if_clip_superset:
                self.superset_text = clip.tokenize(self.superset_all_classes_keys).to(self.device)
                self.superset_text_features_fg = self.clip_model.encode_text(self.superset_text).to(torch.float32)
                self.superset_text_features_fg_norm = self.superset_text_features_fg / self.superset_text_features_fg.norm(dim=1, keepdim=True)
                self.superset_text_features_fg_norm = self.superset_text_features_fg_norm.to(torch.float32)

            self.resize = Resize(self.clip_model.visual.input_resolution,
                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # .exp()
            scale_h = self.clip_model.visual.input_resolution * 1.0 / dataset_config.image_size[0]
            scale_w = self.clip_model.visual.input_resolution * 1.0 / dataset_config.image_size[0]
            self.logit_scale = self.clip_model.logit_scale
            self.trans_mtx = torch.eye(2, device=self.device)
            self.trans_mtx[0, 0] *= scale_w
            self.trans_mtx[1, 1] *= scale_h

            # ############################ train use
            self.test_clip_model, self.test_preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-L-14.pt",
                                                                              device=self.device,
                                                                              download_root='./CLIP/pretrain_models/',
                                                                              if_transform_tensor=True)

            self.test_logit_scale = self.test_clip_model.logit_scale.exp()

            for name, param in self.test_clip_model.named_parameters():
                param.requires_grad = False

            # assert len(self.all_classes_keys) == 233 or len(self.all_classes_keys) == 37 or len(self.all_classes_keys) == 43 or len(self.all_classes_keys) == 232
            if len(self.all_classes_keys) == 233:
                self.test_text = clip.tokenize(self.all_classes_keys[:-1]).to(self.device)
            else:
                self.test_text = clip.tokenize(self.all_classes_keys).to(self.device)
            self.test_text_features_fg = self.test_clip_model.encode_text(self.test_text) #.to(torch.float32)
            self.test_text_features_fg_norm = self.test_text_features_fg / self.test_text_features_fg.norm(dim=1, keepdim=True)
            if self.if_clip_superset:
                self.test_text_features_fg_norm = self.superset_text_features_fg_norm
            self.test_resize = Resize(self.test_clip_model.visual.input_resolution,
                                 interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            scale_h = self.test_clip_model.visual.input_resolution * 1.0 / dataset_config.image_size[0]
            scale_w = self.test_clip_model.visual.input_resolution * 1.0 / dataset_config.image_size[0]
            self.test_trans_mtx = torch.eye(2, device=self.device)
            self.test_trans_mtx[0, 0] *= scale_w
            self.test_trans_mtx[1, 1] *= scale_h
            ############################ train use

            print('================Adopt CLIP encoder=================')
            print('All the class for CLIP:')
            print(self.all_classes_keys)

        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=256,
            hidden_dims=[512, 512],
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)
        print('==============Begin to keep box from which epoch============')
        print(args.begin_keep_epoch)
        self.keep_objectness = keep_objectness
        
        
        print('=========================if online_nms_update_save_novel_label_clip_driven_with_cate_confidence')
        self.online_nms_update_save_novel_label_clip_driven_with_cate_confidence = args.online_nms_update_save_novel_label_clip_driven_with_cate_confidence
        self.save_objectness = args.save_objectness
        print(self.online_nms_update_save_novel_label_clip_driven_with_cate_confidence)
        print(self.save_objectness)
        self.online_nms_update_save_epoch = args.online_nms_update_save_epoch
        print(self.online_nms_update_save_epoch)
        self.clip_driven_keep_thres = args.clip_driven_keep_thres
        print(self.clip_driven_keep_thres)

        print('=============accumulate epochs==============')
        self.online_nms_update_accumulate_epoch = online_nms_update_accumulate_epoch
        print(self.online_nms_update_accumulate_epoch)
        self.distillation_box_num = distillation_box_num
        print('===================The number of distillation boxes===================')
        print(self.distillation_box_num)
        print('===================Evaluating with the layer ID=======================')
        self.eval_layer_id = args.eval_layer_id
        print(self.eval_layer_id)
        print('====================dataset name=========================')
        if args.dataset_name.find('scannet') != -1:
            self.dataset_name = 'scannet'
            self.dataset_util = scannet_utils
        else:
            self.dataset_name = 'sunrgbd'
            self.dataset_util = sunrgbd_utils
        print(self.dataset_name)
        print('====================if get weak labels=====================')
        self.if_clip_weak_labels = args.if_clip_weak_labels
        print(self.if_clip_weak_labels)
        print('=====================if accumulate former pseudo labels========================')
        self.if_accumulate_former_pseudo_labels = args.if_accumulate_former_pseudo_labels
        print(self.if_accumulate_former_pseudo_labels)
       
    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        if self.if_with_fake_classes:
            self.num_cls_predict += 1
        print('------------Predict %d classes(including background/not-an-object class) in 3DETR head------------' % (
                self.num_cls_predict + 1))
        semcls_head = mlp_func(output_dim=self.num_cls_predict + 1)

        # print('------------Predict %d seen classes(including background/not-an-object class) in 3DETR head------------' % (37 + 1)) # should be 37
        # seen_semcls_head = mlp_func(output_dim=37 + 1) # 37
        text_correlation_head = mlp_func(output_dim=512)
        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        mlp_heads = [
            ("sem_cls_head", semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
            ("text_correlation_head", text_correlation_head)
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds)
        return enc_xyz, enc_features, enc_inds

    def clip_to_class_training(self, inputs, outputs, thres_obj=0.05, if_use_gt_box=False, if_expand_box=False,
                               if_padding_input=True,
                               test=False, if_cmp_class=False):  # the max number of classes are 18(train) and 14(val)
        if if_use_gt_box:
            # ceiling experiment

            box_centers_batch = inputs['gt_box_centers'].clone().detach()
            box_angle_batch = inputs['gt_box_angles'].clone().detach()
            box_size_batch = inputs['gt_box_sizes'].clone().detach()
            batch_size = box_size_batch.shape[0]
            # print(batch_size)
            outputs['center_unnormalized'] = inputs['gt_box_centers'].clone().detach()
            outputs['angle_continuous'] = inputs['gt_box_angles'].clone().detach()
            outputs['size_unnormalized'] = inputs['gt_box_sizes'].clone().detach()
            outputs["box_corners"] = inputs['gt_box_corners'].clone().detach()
            # outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, 690)).to('cuda')
            if if_cmp_class:
                outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, len(self.all_cmp_classes_keys))).to('cuda')
            else:
                outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, self.test_range_max)).to('cuda')
            outputs['objectness_prob'] = inputs["gt_box_present"].clone().detach()  # torch.ones((8, 64)).to('cuda')
            x_offset_batch = inputs['x_offset']
            y_offset_batch = inputs['y_offset']
            ori_width_batch = inputs['ori_width']
            ori_height_batch = inputs['ori_height']

        else:
            box_centers_batch = outputs[
                'center_unnormalized']  # .cpu().numpy()#inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
            # print(box_centers.shape)
            box_angle_batch = outputs[
                'angle_continuous']  # .cpu().numpy() #inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
            # print(box_angle.shape)
            box_size_batch = outputs[
                'size_unnormalized']  # .cpu().numpy() #inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
            x_offset_batch = inputs['x_offset']
            y_offset_batch = inputs['y_offset']
            ori_width_batch = inputs['ori_width']
            ori_height_batch = inputs['ori_height']
            # outputs['gt_region_embedding'] = outputs['region_embedding'].clone().detach()
            # region_embedding_batch = outputs['region_embedding']
            # return outputs
            # if test:
        # return outputs
        # print(box_angle_batch.shape)
        # print('=============inputs key=======================')
        # print(inputs.keys())
        img_batch_name = inputs['im_name']
        calib_name_list = inputs['calib_name']
        # trans_mtx_batch = inputs['trans_mtx']
        # print(trans_mtx_batch.shape)
        input_image_batch = inputs["input_image"]
        batch_size_data = box_size_batch.shape[0]
        box_num_data = box_size_batch.shape[1]
        # if test:
        # outputs['sem_cls_prob'] = torch.zeros((batch_size_data, batch_size_data, 232), device=self.device)
        # print('=========================')
        # print(torch.max(outputs['sem_cls_prob']))
        # print(torch.min(outputs['sem_cls_prob']))

        if if_cmp_class:
            outputs['sem_cls_prob'] = torch.zeros(
                (box_centers_batch.shape[0], box_centers_batch.shape[1], len(self.all_cmp_classes_keys)),
                device=self.device)  # .to('cuda')
            outputs['sem_cls_logits'] = torch.zeros(
                (box_centers_batch.shape[0], box_centers_batch.shape[1], len(self.all_cmp_classes_keys)),
                device=self.device)  # .to('cuda')
        else:
            outputs['sem_cls_prob'] = torch.zeros(
                (box_centers_batch.shape[0], box_centers_batch.shape[1], self.test_range_max),
                device=self.device)  # .to('cuda')
            outputs['sem_cls_logits'] = torch.zeros(
                (box_centers_batch.shape[0], box_centers_batch.shape[1], self.test_range_max),
                device=self.device)  # .to('cuda')
        #     outputs['sem_cls_logits'] = torch.zeros((batch_size_data, batch_size_data, 239))
        #     outputs['objectness_prob'] = torch.zeros((batch_size_data, batch_size_data, 1))
        #     text_features = torch.cat((self.text_features_fg_test, self.bg_embed + 1e-16), dim=0)
        #     # normalized features
        #     text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
        # else:
        # test_text_features = self.test_text_features_fg  # torch.cat((, self.bg_embed + 1e-16), dim=0)
        # normalized features
        # test_text_features_norm = test_text_features / test_text_features.norm(dim=1, keepdim=True)
        # text_features = self.text_features_fg  # torch.cat((, self.bg_embed + 1e-16), dim=0)
        # normalized features
        if if_cmp_class:
            text_features = self.cmp_text_features_fg
        else:
            text_features = self.test_text_features_fg
        # normalized features
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
        
        for idx_batch in range(batch_size_data):
            # box_center_list = box_centers_batch[idx_batch]
            # print(box_angle.shape)

            x_offset = x_offset_batch[idx_batch]
            y_offset = y_offset_batch[idx_batch]

            calib_name = calib_name_list[idx_batch]
            img = input_image_batch[idx_batch]
            # img = input_image.permute(2, 0, 1)

            # input_im = Image.fromarray(input_image[:,:,::-1])
            # input_im_ = np.array(input_im)
            # img_ = Image.open(img_batch_name[idx_batch])
            # img_ = np.array(img)
            # if_seem = np.all(np.equal(img_, input_im_))
            # img_ = cv2.imread(img_batch_name[idx_batch])
            if self.dataset_name == 'scannet':
                squence_name = inputs['squence_name'][idx_batch]
                calib = self.dataset_util.SCANNET_Calibration(calib_name, squence_name)
            if self.dataset_name == 'sunrgbd':
                calib = self.dataset_util.SUNRGBD_Calibration(calib_name)
            box_center_list = box_centers_batch[idx_batch]
            box_angle_list = box_angle_batch[idx_batch]
            box_size_list = box_size_batch[idx_batch]
            # trans_mtx = trans_mtx_batch[idx_batch]
            ori_width = ori_width_batch[idx_batch]
            ori_height = ori_height_batch[idx_batch]

            proposal_feature_list = []
            idx_box_list = []
            for idx_box in range(box_num_data):
                # print(outputs['seen_sem_cls_prob'][idx_batch][idx_box])
                # outputs["sem_cls_prob"][idx_batch][idx_box]
                # ob_prob = outputs["objectness_prob"][idx_batch][idx_box]
                # others_prob = outputs["others_prob"][idx_batch][idx_box]
                # if others_prob
                # a = outputs["sem_cls_prob"][idx_batch][idx_box]
                # print(torch.sum(a)+torch.sum(1-ob_prob))
                # max_prob, max_class = torch.max(a, 0)
                # if not (others_prob > max_prob and ob_prob > 0.05):
                #     continue
                # if max_prob > 0.5 and max_class > 0:
                #
                #     print(ob_prob)
                #     print(max_prob)
                #     print(max_class)
                # print('--------------------------------')
                # if max_prob > 0.5 and max_class < 37:
                # print(max_class)
                # print(max_prob)
                # print('--------------------')
                # continue
                # outputs['sem_cls_prob'][idx_batch][idx_box] = torch.zeros((232), device=self.device)
                #
                # if max_prob > 0.5:
                #     outputs['sem_cls_prob'][idx_batch][idx_box][:37] = outputs['seen_sem_cls_prob'][idx_batch][idx_box]
                #     # print(max_prob)
                #     continue

                box_center = box_center_list[idx_box]
                box_angle = box_angle_list[idx_box]
                box_size = box_size_list[idx_box]
                # region_embed = region_embedding_batch[idx_batch, idx_box]
                # trans_mtx = trans_mtx_list[idx_box]

                if torch.max(box_size) < 1e-16:  # when displaying gt, some of the box_size will be zeros
                    # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                    # idx_box_list.append(0)
                    continue

                # print(idx_box)
                if if_padding_input:
                    corners_3d_image, corners_3d, d = self.dataset_util.compute_box_3d_offset_tensor(box_center, box_angle,
                                                                                              box_size,
                                                                                              calib, x_offset, y_offset)
                    xmin = int(min(max(torch.min(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                    ymin = int(min(max(torch.min(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                    xmax = int(min(max(torch.max(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                    ymax = int(min(max(torch.max(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                    # corners_3d_image_np, corners_3d_np = sunrgbd_utils.compute_box_3d_offset(box_center.detach().cpu().numpy(), box_angle.detach().cpu().numpy(), box_size.detach().cpu().numpy(),
                    #                                                                    calib, x_offset.detach().cpu().numpy(), y_offset.detach().cpu().numpy())
                    # print(int(np.max(corners_3d_image.detach().cpu().numpy()))==int(np.max(corners_3d_image_np)))
                    # print(int(np.min(corners_3d_image.detach().cpu().numpy())) == int(np.min(corners_3d_image_np)))
                    # print(np.sum(np.abs(corners_3d.detach().cpu().numpy() - corners_3d_np)))
                    # xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    # ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                    # xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    # ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                else:
                    corners_3d_image, corners_3d = self.dataset_util.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                       calib, trans_mtx)
                    xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                    xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))

                if torch.min(d) < 0:                    
                    # print('d less than 0')
                    continue

                if if_expand_box:
                    w = ymax - ymin
                    h = xmax - xmin
                    if w > h:
                        xmin = xmin - (w - h) // 2
                        xmax = xmax + (w - h) // 2
                    else:
                        ymin = ymin - (h - w) // 2
                        ymax = ymax + (h - w) // 2
                    xmin = int(min(max(xmin, 0), img.shape[1]))
                    ymin = int(min(max(ymin, 0), img.shape[0]))
                    xmax = int(min(max(xmax, 0), img.shape[1]))
                    ymax = int(min(max(ymax, 0), img.shape[0]))



                if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                    img_crop = img[ymin:ymax, xmin:xmax]
                    # img_crop_show = img_crop.cpu().numpy()
                    # # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
                    # cv2.imwrite('debug_show_clip_2/' + os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'_2_clip_predicted.png', img_crop_show)
                    # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
                    # continue
                else:
                    print('Box size is less than 0')
                    # idx_box_list.append(0)
                    # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                    continue

                idx_box_list.append(idx_box)

                # #####exp1: blank background#####
                # part 1
                # img_crop_ = img_crop.clone()
                w = ymax - ymin
                h = xmax - xmin
                if w > h:
                    max_edge = w
                else:
                    max_edge = h
                #
                img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8,
                                            device=self.device) * 255
                # part 2

                y_begin = (max_edge - w) // 2
                x_begin = (max_edge - h) // 2
                img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
                # part 3
                img_crop = img_beckground.permute(2, 0, 1)
                img_crop_resized = self.test_resize(img_crop)

                img_crop_resized_expand = img_crop_resized.unsqueeze(0)
                proposal_feature_list.append(img_crop_resized_expand)

            # print(self.clip_model.visual.input_resolution)
            try:
                img_proposal = torch.cat(proposal_feature_list, 0)
            except:
                print('No object on GT.')
                # outputs['objectness_prob'][idx_batch]
                continue

            # img_proposal = torch.cat(proposal_feature_list, 0)
            # np.save('img_proposal.npy', img_proposal.cpu().numpy())
            image_proposal_batch = self.test_preprocess_for_tensor(img_proposal).unsqueeze(0)[0]
            # np.save('image_proposal_batch.npy', image_proposal_batch.cpu().numpy())

            proposal_features_batch = self.test_clip_model.encode_image(image_proposal_batch)
            # np.save('proposal_features_batch.npy', proposal_features_batch.cpu().numpy())
            # proposal_features_batch = self.test_clip_model.encode_image(image_proposal_batch)
            if isinstance(proposal_features_batch, tuple):
                proposal_features_batch = proposal_features_batch[0]
            proposal_features_batch = proposal_features_batch.to(torch.float32)
            proposal_features_batch_norm = proposal_features_batch / proposal_features_batch.norm(dim=1, keepdim=True)
            # np.save('proposal_features_batch_norm.npy', proposal_features_batch_norm.cpu().numpy())
            logits_per_image = self.test_logit_scale * proposal_features_batch_norm @ text_features_norm.t().to(
                torch.float32)
            # np.save('logits_per_image.npy', logits_per_image.cpu().numpy())
            # idx_tensor = torch.tensor(idx_box_list).to(self.device)
            # outputs['gt_region_embedding'][idx_batch][idx_box_list] =  proposal_features_batch #torch.gather(proposal_features_batch, dim=0, index=idx_tensor)
            # if not self.if_adopt_region_embed:
            probs = logits_per_image.softmax(dim=-1)

            # np.save('probs.npy', probs.cpu().numpy())

            outputs['sem_cls_prob'][idx_batch][idx_box_list] = probs[:,
                                                               :].detach()  # torch.gather(probs[:, :-1].detach(), dim=-1, index=idx_tensor)
            # a = probs[:,
            #                                                    :].detach()
            # max_v, max_idx = torch.max(probs, dim=-1)
            # print(max_idx)
            # print(max_v)
            # print(outputs['objectness_prob'][idx_batch][idx_box_list].shape)
            # b = torch.where(max_v > 0.3,
            #                                                                   outputs['objectness_prob'][idx_batch][
            #                                                                       idx_box_list], torch.zeros_like(
            #         outputs['objectness_prob'][idx_batch][idx_box_list], device='cuda'))
            # outputs['objectness_prob'][idx_batch][idx_box_list] = torch.where(max_v > 0.3, outputs['objectness_prob'][idx_batch][idx_box_list], torch.zeros_like(outputs['objectness_prob'][idx_batch][idx_box_list], device='cuda'))
            # print(outputs['sem_cls_prob'][idx_batch][idx_box_list])
            # outputs['sem_cls_logits'][idx_batch][idx_box_list] = logits_per_image #torch.gather(logits_per_image, dim=-1, index=idx_tensor)
            # outputs['objectness_prob'][idx_batch][idx_box_list] = 1.0-probs[:, -1] #torch.gather(1.0-probs[:, -1], dim=-1, index=idx_tensor)
            # np.save('sem_cls_prob_first_batch.npy', outputs['sem_cls_prob'][idx_batch].cpu().numpy())
            # np.save('objectness_prob_first_batch.npy', outputs['objectness_prob'][idx_batch].cpu().numpy())
            # return outputs

        # if self.if_adopt_region_embed:
        #     region_embed_norm = region_embedding_batch / region_embedding_batch.norm(dim=-1, keepdim=True)
        #     logits_per_image = self.logit_scale * region_embed_norm @ text_features_norm.t()
        #     probs = logits_per_image.softmax(dim=-1)  # .cpu().numpy()
        #     outputs['sem_cls_prob'] = probs[:, :, :-1].detach()
        #     outputs['sem_cls_logits'] = logits_per_image
        # outputs['objectness_prob'] = 1.0 - probs[:, :, -1]
        # del outputs['objectness_prob']
        # del outputs['sem_cls_prob']
        # return 1, 2, 3
        return outputs  # , outputs['sem_cls_prob'], outputs['objectness_prob']

    def cal_iou(self, pred_box_select, gt_box_select):
        '''
        calculate the iou
        '''
        x1 = pred_box_select[0]
        y1 = pred_box_select[1]
        z1 = pred_box_select[2]
        x2 = pred_box_select[3]
        y2 = pred_box_select[4]
        z2 = pred_box_select[5]
        a1 = gt_box_select[0]
        b1 = gt_box_select[1]
        c1 = gt_box_select[2]
        a2 = gt_box_select[3]
        b2 = gt_box_select[4]
        c2 = gt_box_select[5]

        xx1 = torch.maximum(x1, a1)
        yy1 = torch.maximum(y1, b1)
        zz1 = torch.maximum(z1, c1)
        xx2 = torch.minimum(x2, a2)
        yy2 = torch.minimum(y2, b2)
        zz2 = torch.minimum(z2, c2)
        zeros_tensor = torch.zeros_like(xx1, device=self.device)
        l = torch.maximum(zeros_tensor, xx2 - xx1)
        w = torch.maximum(zeros_tensor, yy2 - yy1)
        h = torch.maximum(zeros_tensor, zz2 - zz1)
        area1 = (x2 - x1) * (y2 - y1) * (z2 - z1)
        area2 = (a2 - a1) * (b2 - b1) * (c2 - c1)
        inter = l * w * h
        iou = inter / (area1 + area2 - inter)
        return iou


    def get_predicted_box_clip_embedding(self, inputs, outputs, thres_obj=0.05, if_use_gt_box=False,
                                         if_expand_box=False,
                                         if_padding_input=True,
                                         test=False,
                                         curr_epoch=-1):  # the max number of classes are 18(train) and 14(val)
        gt_center_ori = outputs[
            "box_corners_xyz"]  # .clone().detach() #inputs["gt_box_corners_xyz"]  # shape: (8, 64, 8, 3) box_predictions["outputs"]['center_unnormalized']
        objectness_prob = outputs["objectness_prob"]
        # img_batch_name = inputs['im_name']
        # gt_angles = inputs["gt_box_angles"]
        # gt_sizes = inputs["gt_box_sizes"]
        # tmp = outputs['center_unnormalized']  # shape: (8, 128, 3)
        K_tensor = inputs['K']
        Rtilt_tensor = inputs['Rtilt']
        flip_array = inputs['flip_array'].unsqueeze(-1)

        scale_array = inputs['scale_array'].unsqueeze(1)
        rot_array = inputs['rot_array'].unsqueeze(1)
        # back the augmentation for point cloud
        gt_center = gt_center_ori * scale_array  # [:, :2, :]
        gt_center = torch.matmul(gt_center, rot_array)
        if 'zx_flip_array' in inputs.keys():
            zx_flip_array = inputs['zx_flip_array'].unsqueeze(-1)
            gt_center[:, :, :, 1] = gt_center[:, :, :, 1] * zx_flip_array
        gt_center[:, :, :, 0] = gt_center[:, :, :, 0] * flip_array  # [:, :2, :]
        # back to original image
        # center_2dpoints = torch.zeros((8, 64, 8, 2), device=self.device)
        # for idx_batch in range(center_2dpoints.shape[1]):
        #     center_2dpoints[:, idx_batch, :, :] = project_3dpoint_to_2dpoint_tensor(center_pos.to(torch.double)[:, idx_batch, :, :], K_tensor=K_tensor, Rtilt_tensor=Rtilt_tensor)
        center_2dpoints, depth = self.dataset_util.project_3dpoint_to_2dpoint_corners_tensor(gt_center.to(torch.double),
                                                                                             K_tensor=K_tensor,
                                                                                             Rtilt_tensor=Rtilt_tensor)
        # center_2dpoints, _ = project_3dpoint_to_2dpoint_corners_tensor(gt_center, gt_angles, gt_sizes, K_tensor=K_tensor,
        #                                                     Rtilt_tensor=Rtilt_tensor)

        # center_pos = inputs["gt_box_corners"] # shape: (8, 64, 8, 3) box_predictions["outputs"]['center_unnormalized']
        # center_pos = outputs['center_unnormalized']  # shape: (8, 128, 3)
        # K_tensor = inputs['K']
        # Rtilt_tensor = inputs['Rtilt']
        # flip_array = inputs['flip_array']#.unsqueeze(-1)
        # scale_array = inputs['scale_array']   #.unsqueeze(1)
        # rot_array = inputs['rot_array']       #.unsqueeze(1)
        # # back the augmentation for point cloud
        # center_pos = center_pos * scale_array  # [:, :2, :]
        # center_pos = torch.bmm(center_pos, rot_array)
        # center_pos[:, :,  0] = center_pos[:, :,  0] * flip_array  # [:, :2, :]
        # center_2dpoints = project_3dpoint_to_2dpoint_tensor(center_pos.to(torch.double), K_tensor=K_tensor,
        #                                                     Rtilt_tensor=Rtilt_tensor)

        # clip the point in the original size of the image
        for idx_batch in range(center_2dpoints.shape[0]):
            center_2dpoints[idx_batch, :, :, 0] = torch.clip(center_2dpoints[idx_batch, :, :, 0], min=0,
                                                             max=inputs["ori_width"][idx_batch] - 1)
            center_2dpoints[idx_batch, :, :, 1] = torch.clip(center_2dpoints[idx_batch, :, :, 1], min=0,
                                                             max=inputs["ori_height"][idx_batch] - 1)
            center_2dpoints[idx_batch, :, :, 0] += inputs['y_offset'][idx_batch].unsqueeze(-1)
            center_2dpoints[idx_batch, :, :, 1] += inputs['x_offset'][idx_batch].unsqueeze(
                -1)  # .unsqueeze(-1)  # short height
        # center_2dpoints[:, :, :, 0] = torch.maximum(
        #     torch.minimum(center_2dpoints[:, :, :, 0], (inputs["ori_width"].unsqueeze(-1) - 1)), self.zeros_tensor)
        # center_2dpoints[:, :, :, 1] = torch.maximum(
        #     torch.minimum(center_2dpoints[:, :, :, 1], (inputs["ori_height"].unsqueeze(-1) - 1)), self.zeros_tensor)

        # center_2dpoints[:, :, 0] = torch.clip(center_2dpoints[:, :, 0], min=)
        # do the padding in the datalayer

        # do the flip in the datalayer
        image_flip_array = inputs["image_flip_array"].unsqueeze(-1)
        flip_length = inputs["flip_length"].unsqueeze(-1).unsqueeze(-1)
        center_2dpoints[:, :, :, 0] = center_2dpoints[:, :, :, 0] * image_flip_array + (1 - image_flip_array) * (
                flip_length - 1 - center_2dpoints[:, :, :, 0])

        # outputs['gt_text_correlation_embedding'] = torch.zeros((center_2dpoints.shape[0], center_2dpoints.shape[1], outputs["text_correlation_embedding"].shape[2]), device=self.device)
        outputs['gt_text_correlation_embedding'] = torch.zeros(
            (center_2dpoints.shape[0], center_2dpoints.shape[1], 512),  # change version
            device=self.device)
        # outputs['gt_text_correlation_embedding'] = torch.zeros(
        #     (center_2dpoints.shape[0], center_2dpoints.shape[1], 768),
        #     device=self.device)
        outputs['gt_text_correlation_embedding_mask'] = torch.zeros(
            (center_2dpoints.shape[0], center_2dpoints.shape[1], 1),
            device=self.device)
        for idx_batch in range(center_2dpoints.shape[0]):
            proposal_feature_list = []
            idx_box_list = []
            # cv2.imwrite(os.path.join('./%s_3d_to_2d_box_image.png' % idx_batch), inputs['input_image'][idx_batch].cpu().numpy())
            # if_keep_box = torch.rand(size=(center_2dpoints.shape[1], 1), device=self.device, requires_grad=False)
            objectness_prob_this_batch = objectness_prob[idx_batch]
            if (not self.if_select_box_by_objectness) or curr_epoch < 540:
                select_list = np.random.choice(self.box_idx_list, self.distillation_box_num, replace=False)
            else:
                ob_list = torch.nonzero(objectness_prob_this_batch > 0.05)
                # locat_ob_list = torch.nonzero(objectness_prob_this_batch > 0.5)
                # locat_ob_list = locat_ob_list.cpu().numpy()
                ob_list = ob_list.cpu().numpy()
                if ob_list.shape[0] >= self.distillation_box_num:
                    select_list = ob_list[:, 0]
                else:
                    bg_list = torch.nonzero(objectness_prob_this_batch <= 0.05)
                    bg_list = bg_list.cpu().numpy()
                    left_len = self.distillation_box_num - ob_list.shape[0]
                    select_list = np.concatenate(
                        (ob_list[:, 0], np.random.choice(bg_list[:, 0], left_len, replace=False)))

            # select_list = np.random.choice(self.box_idx_list, 32, replace=False)
            # assert len(select_list) >= 30
            # print(select_list)
            # print(len(select_list))
            locat_idx_box_list = []
            for idx_box in select_list:
                img = inputs['input_image'][idx_batch]
                box_size = outputs["size_unnormalized"][idx_batch][idx_box]
                if torch.max(box_size) < 1e-16:  # when displaying gt, some of the box_size will be zeros
                    # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                    # idx_box_list.append(0)
                    continue
                corners_3d_image = center_2dpoints[idx_batch][idx_box]
                # corners_3d_image = center_2dpoints[idx_batch][0]
                xmin = int(torch.min(corners_3d_image[:, 0]))
                ymin = int(torch.min(corners_3d_image[:, 1]))
                xmax = int(torch.max(corners_3d_image[:, 0]))
                ymax = int(torch.max(corners_3d_image[:, 1]))

                if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                    img_crop = img[ymin:ymax, xmin:xmax]
                #     img_crop_show = img_crop.cpu().numpy()
                # # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
                #     cv2.imwrite('debug_show_2/' + os.path.basename(img_batch_name[idx_batch])[:-4] + '_' + str(
                #         idx_box) + '_2_predicted_embedding.png', img_crop_show)
                # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
                else:
                    continue
                if torch.min(depth[idx_batch][idx_box]) < 0:
                    continue

                idx_box_list.append(idx_box)
                if objectness_prob_this_batch[idx_box] > self.keep_objectness:
                    locat_idx_box_list.append(idx_box)

                w = ymax - ymin
                h = xmax - xmin

                if w > h:
                    max_edge = w
                else:
                    max_edge = h
                #
                img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8,
                                            device=self.device) * 255
                # part 2

                y_begin = (max_edge - w) // 2
                x_begin = (max_edge - h) // 2
                # try:
                # print(img_crop.shape)
                # print(x_begin)
                # print(y_begin)
                # print(x_begin + h)
                # print(y_begin + w)
                # print('============')
                img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
                # except:
                # print(1)
                # part 3
                img_crop = img_beckground.permute(2, 0, 1)
                img_crop_resized = self.resize(img_crop)  # change version
                # print(img_crop_resized.shape)
                # img_crop_resized = self.test_resize(img_crop)
                # cv2.imwrite(os.path.join('./show_outputs/no_aug/%s_%s_3d_to_2d_box_gt.png' % (idx_batch, idx_box)), img_crop_resized.permute(1, 2, 0).cpu().numpy())
                # cv2.imwrite(os.path.join('./show_outputs/no_aug/%s_%s_original.png' % (idx_batch, idx_box)), img.cpu().numpy())
                img_crop_resized_expand = img_crop_resized.unsqueeze(0)
                proposal_feature_list.append(img_crop_resized_expand)

            # if len(proposal_feature_list) < 1:
            #     continue
            try:
                img_proposal = torch.cat(proposal_feature_list, 0)
            except:
                print('No object on GT, too')
                continue
            # im_proposal_ref = np.load('img_proposal.npy')
            # image_proposal_batch_ref = np.load('image_proposal_batch.npy')
            # proposal_features_batch_ref = np.load('proposal_features_batch.npy')

            # img_proposal = torch.cat(proposal_feature_list, 0)

            image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]  # change version
            # image_proposal_batch_copy = image_proposal_batch.detach().clone()
            # proposal_features_batch = self.res_encoder(image_proposal_batch, if_pool=True, if_early_feat=False)
            proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)  # change version
            # proposal_features_batch_1 = self.clip_model_1.encode_image(image_proposal_batch_copy)  # change version
            # assert torch.equal(proposal_features_batch, proposal_features_batch_0)
            # print(idx_box_list)
            # print(len(idx_box_list))
            # print('==================')save_seen_feat_only
            # print(image_proposal_batch.shape)
            if isinstance(proposal_features_batch, tuple):
                proposal_features_batch = proposal_features_batch[0]
            proposal_features_batch = proposal_features_batch.to(torch.float32)
            # tmp = outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list]
            outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list] = proposal_features_batch
            outputs['gt_text_correlation_embedding_mask'][idx_batch][idx_box_list] = 1

            # ob_idx_box_list = locat_ob_list
            # print(1)
            if self.if_keep_box and curr_epoch >= 540:
                if len(locat_idx_box_list) < 1:
                    continue
                ob_proposal_features_batch = outputs['gt_text_correlation_embedding'][idx_batch][locat_idx_box_list]
                text_features_clip = outputs["text_features_clip"][idx_batch].to(torch.float32)
                # print(text_features_clip.shape)
                temperature_param = outputs["logit_scale"]
                # text_correlation_embedding = outputs["text_correlation_embedding"]
                text_correlation_embedding = ob_proposal_features_batch / (
                        ob_proposal_features_batch.norm(dim=-1, keepdim=True) + 1e-32)
                correlation_map = torch.mm(text_correlation_embedding,
                                           text_features_clip.permute(1, 0)) * temperature_param
                scores = torch.nn.functional.softmax(correlation_map, dim=-1)
                max_score, max_id = torch.max(scores, dim=-1)

                novel_list = torch.nonzero(torch.logical_and(max_score > 0.5, max_id > 9))
                novel_list = novel_list.cpu().numpy()[:, 0]
                novel_idx_box_list = np.array(locat_idx_box_list)[novel_list]
                # print(1)
                begin_idx = int(torch.sum(inputs["gt_box_present"][idx_batch]).item())
                for novel_idx in novel_idx_box_list:
                    if begin_idx > 63:
                        print('Final gt boxs are more than 64')
                        break
                    # print('Have new boxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:')
                    # print(inputs["im_name"][idx_batch])
                    inputs["gt_box_present"][idx_batch][begin_idx] = 1
                    this_novel_scores = outputs["angle_logits"][idx_batch][novel_idx].softmax(dim=-1)
                    max_this_novel_scores, max_idx_this_novel_scores = torch.max(this_novel_scores, dim=-1)
                    inputs["gt_angle_class_label"][idx_batch][begin_idx] = max_idx_this_novel_scores
                    inputs["gt_angle_residual_label"][idx_batch][begin_idx] = \
                        outputs["angle_residual"][idx_batch][novel_idx][max_idx_this_novel_scores]
                    inputs["gt_box_sizes_normalized"][idx_batch][begin_idx] = outputs["size_normalized"][idx_batch][
                        novel_idx]
                    inputs["gt_box_sizes"][idx_batch][begin_idx] = outputs["size_unnormalized"][idx_batch][novel_idx]
                    inputs["gt_box_corners"][idx_batch][begin_idx] = outputs["box_corners"][idx_batch][novel_idx]
                    inputs["gt_box_corners_xyz"][idx_batch][begin_idx] = outputs["box_corners_xyz"][idx_batch][
                        novel_idx]
                    inputs["gt_box_angles"][idx_batch][begin_idx] = outputs["angle_continuous"][idx_batch][novel_idx]
                    inputs["gt_box_centers_normalized"][idx_batch][begin_idx] = outputs["center_normalized"][idx_batch][
                        novel_idx]
                    inputs["gt_box_centers"][idx_batch][begin_idx] = outputs["center_unnormalized"][idx_batch][
                        novel_idx]
                    # a = 1
                    begin_idx += 1

        if self.if_clip_weak_labels:
            text_features_clip = outputs["text_features_clip"].to(torch.float32)
            # print(text_features_clip.shape)
            temperature_param = outputs["logit_scale"]
            gt_image_embedding = outputs['gt_text_correlation_embedding']
            gt_image_embedding_mask = outputs['gt_text_correlation_embedding_mask']
            gt_image_embedding = gt_image_embedding / (
                    gt_image_embedding.norm(dim=-1, keepdim=True) + 1e-32)
            correlation_map = torch.bmm(gt_image_embedding,
                                        text_features_clip.permute(0, 2, 1)) * temperature_param
            box_scores = torch.nn.functional.softmax(correlation_map, dim=-1)
            max_score, max_id = torch.max(box_scores, dim=-1)
            # bg_id = correlation_map.shape[-1]
            weak_confidence_weight = max_score
            weak_box_cate_label = max_id
            # weak_box_cate_label[gt_image_embedding_mask[:,:,0] < 1] = bg_id # if cann't project to find a 2d box and also cann't assign in the loss, it is taken as a wrong box
            weak_confidence_weight[gt_image_embedding_mask[:, :,
                                   0] < 1] = 0.0  # confience_weight for bg object is set to 0, because it is no
            outputs["weak_box_cate_label"] = weak_box_cate_label
            outputs["weak_confidence_weight"] = weak_confidence_weight
        else:

            outputs["weak_box_cate_label"] = torch.zeros(
                (center_2dpoints.shape[0], center_2dpoints.shape[1]),
                device=self.device, dtype=torch.int64)
            outputs["weak_confidence_weight"] = torch.zeros(
                (center_2dpoints.shape[0], center_2dpoints.shape[1]),
                device=self.device)
            # print(1)
        # just make sure the corners are right
        # for idx_batch in range(center_2dpoints.shape[0]):
        #     img_draw2d_box = inputs['input_image'][idx_batch].cpu().numpy()
        #     center_2dpoints_np = center_2dpoints.cpu().numpy()
        #     for idx_box in range(center_2dpoints.shape[1]):
        #
        #         # img_draw2d_box = sunrgbd_utils.draw_projected_box3d(img_draw2d_box, center_2dpoints_np[idx_batch][idx_box])
        #         img_draw2d_boximg_draw2d_box = img_draw2d_box
        #         cv2.imwrite(os.path.join('./debug_show/%s_%s_3d_to_2d_box_im_gt.png' % (idx_batch, idx_box)), img_draw2d_box)

        # # ################### add image embedding
        # img = inputs['input_image']
        # max_img_h = max(img.shape[1], img.shape[2])
        # full_img_beckground = torch.ones(img.shape[0], max_img_h, max_img_h, img.shape[3], dtype=torch.uint8, device=self.device) * 255
        # y_begin = (max_img_h - img.shape[1]) // 2
        # x_begin = (max_img_h - img.shape[2]) // 2
        # full_img_beckground[:, y_begin:y_begin + img.shape[1], x_begin:x_begin + img.shape[2], :] = img
        # full_img = full_img_beckground.permute(0, 3, 1, 2)
        # full_img_crop = self.resize(full_img)  #.unsqueeze(0)
        # full_image_proposal_batch = self.preprocess_for_tensor(full_img_crop).unsqueeze(0)[0]
        # full_image_proposal_features_batch = self.clip_model.encode_image(full_image_proposal_batch)
        # if isinstance(full_image_proposal_features_batch, tuple):
        #     full_image_proposal_features_batch = full_image_proposal_features_batch[0]
        # full_image_proposal_features_batch = full_image_proposal_features_batch.to(torch.float32)
        # outputs['full_image_embedding'] = full_image_proposal_features_batch
        # # ################### add image embedding
        # full_image_proposal_features_batch = full_image_proposal_features_batch / full_image_proposal_features_batch.norm(dim=1, keepdim=True)

        return outputs

    def get_predicted_box_clip_embedding_nms_iou_save_keep_clip_driven_with_cate_confidence(self, inputs, outputs, thres_obj=0.05,
                                                           if_use_gt_box=False,
                                                           if_expand_box=False,
                                                           if_padding_input=True,
                                                           test=False,
                                                           curr_epoch=-1,
                                                           if_test=False):  # the max number of classes are 18(train) and 14(val)
        gt_center_ori = outputs[
            "box_corners_xyz"]  # .clone().detach() #inputs["gt_box_corners_xyz"]  # shape: (8, 64, 8, 3) box_predictions["outputs"]['center_unnormalized']
        objectness_prob = outputs["objectness_prob"].detach().clone()
        K_tensor = inputs['K']
        Rtilt_tensor = inputs['Rtilt']
        flip_array = inputs['flip_array'].unsqueeze(-1)
        scale_array = inputs['scale_array'].unsqueeze(1)
        rot_array = inputs['rot_array'].unsqueeze(1)
        rot_angle = inputs['rot_angle'].unsqueeze(1)

        # back the augmentation for point cloud
        gt_center = gt_center_ori * scale_array  # [:, :2, :]
        gt_center = torch.matmul(gt_center, rot_array)
        if 'zx_flip_array' in inputs.keys():
            zx_flip_array = inputs['zx_flip_array'].unsqueeze(-1)
            gt_center[:, :, :, 1] = gt_center[:, :, :, 1] * zx_flip_array
        gt_center[:, :, :, 0] = gt_center[:, :, :, 0] * flip_array  # [:, :2, :]

        if (not if_test) and (
                curr_epoch % self.online_nms_update_save_epoch == 0):  # and curr_epoch >= 540:
            ori_center_unnormalized = outputs["center_unnormalized"].detach().clone() * inputs['scale_array']
            ori_size_unnormalized = outputs["size_unnormalized"].detach().clone() * inputs['scale_array']

            ori_center_unnormalized = torch.matmul(ori_center_unnormalized, inputs['rot_array'])
            ori_angle_continuous = outputs["angle_continuous"].detach().clone() + inputs['rot_angle'].unsqueeze(-1)

            if 'zx_flip_array' in inputs.keys():
                ori_center_unnormalized[:, :, 1] = ori_center_unnormalized[:, :, 1] * inputs['zx_flip_array']
                ori_angle_continuous = torch.where(inputs['zx_flip_array'] < 0, (math.pi - ori_angle_continuous),
                                                   ori_angle_continuous)

            ori_center_unnormalized[:, :, 0] = ori_center_unnormalized[:, :, 0] * inputs['flip_array']
            ori_angle_continuous = torch.where(inputs['flip_array'] < 0, (math.pi - ori_angle_continuous),
                                               ori_angle_continuous)


        center_2dpoints, depth = self.dataset_util.project_3dpoint_to_2dpoint_corners_tensor(gt_center.to(torch.double), # project the 3d boxes to 2d axis
                                                                                     K_tensor=K_tensor,
                                                                                     Rtilt_tensor=Rtilt_tensor)


        # clip the point in the original size of the image
        text_features_clip = outputs["maybe_novel_text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = outputs["logit_scale"]
        for idx_batch in range(center_2dpoints.shape[0]):
            center_2dpoints[idx_batch, :, :, 0] = torch.clip(center_2dpoints[idx_batch, :, :, 0], min=0,
                                                             max=inputs["ori_width"][idx_batch] - 1)
            center_2dpoints[idx_batch, :, :, 1] = torch.clip(center_2dpoints[idx_batch, :, :, 1], min=0,
                                                             max=inputs["ori_height"][idx_batch] - 1)
            center_2dpoints[idx_batch, :, :, 0] += inputs['y_offset'][idx_batch].unsqueeze(-1)
            center_2dpoints[idx_batch, :, :, 1] += inputs['x_offset'][idx_batch].unsqueeze(
                -1)  # .unsqueeze(-1)  # short height

        # do the flip in the datalayer
        image_flip_array = inputs["image_flip_array"].unsqueeze(-1)
        flip_length = inputs["flip_length"].unsqueeze(-1).unsqueeze(-1)
        center_2dpoints[:, :, :, 0] = center_2dpoints[:, :, :, 0] * image_flip_array + (1 - image_flip_array) * (
                flip_length - 1 - center_2dpoints[:, :, :, 0])

        gt_box_corners = inputs["gt_box_corners"].detach().clone()
        gt_box_present = inputs["gt_box_present"].detach().clone()
        pred_box_corneres = outputs["box_corners"].detach().clone()
        box_keep = torch.zeros((center_2dpoints.shape[0], center_2dpoints.shape[1]), device=self.device)
        box_save = torch.zeros((center_2dpoints.shape[0], center_2dpoints.shape[1]), device=self.device)

        outputs['gt_text_correlation_embedding'] = torch.zeros(
            (center_2dpoints.shape[0], center_2dpoints.shape[1], 512),
            device=self.device)

        outputs['gt_text_correlation_embedding_mask'] = torch.zeros(
            (center_2dpoints.shape[0], center_2dpoints.shape[1], 1),
            device=self.device)
        if (not if_test) and (
                curr_epoch % self.online_nms_update_save_epoch == 0):  # summarize the box information
            save_box_info = torch.zeros(
                (outputs["center_unnormalized"].shape[0], outputs["center_unnormalized"].shape[1], 10), #(center(3), size(3), angle, class, cate_prob, objectness)
                device=self.device)
            save_box_info_tmp = torch.cat(
                [ori_center_unnormalized, ori_size_unnormalized, ori_angle_continuous.unsqueeze(-1)], dim=-1)
            save_box_info[:, :, :7] = save_box_info_tmp
            save_box_info[:, :, 7] = -1
            save_box_info[:, :, 8] = 0
            save_box_info[:, :, 9] = 0
            save_box_info = save_box_info.detach().cpu().numpy()

        for idx_batch in range(center_2dpoints.shape[0]):
            if (not if_test) and (curr_epoch % self.online_nms_update_save_epoch == 0):  # and curr_epoch >= 540:
                box2d_thisbatch = torch.zeros((center_2dpoints.shape[1], 4), device=self.device)
                scores = objectness_prob[idx_batch]
                del_box_list = []
                for idx_box in range(center_2dpoints.shape[1]):
                    box_size = outputs["size_unnormalized"][idx_batch][idx_box]
                    if torch.max(box_size) < 1e-16:
                        # give the box up
                        scores[idx_box] = -1
                        del_box_list.append(idx_box)
                        box2d_thisbatch[idx_box][0] = 0
                        box2d_thisbatch[idx_box][1] = 0
                        box2d_thisbatch[idx_box][2] = 2
                        box2d_thisbatch[idx_box][3] = 2
                        continue
                    corners_3d_image = center_2dpoints[idx_batch][idx_box]
                    xmin = int(torch.min(corners_3d_image[:, 0]))
                    ymin = int(torch.min(corners_3d_image[:, 1]))
                    xmax = int(torch.max(corners_3d_image[:, 0]))
                    ymax = int(torch.max(corners_3d_image[:, 1]))
                    if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                        # give the box up
                        scores[idx_box] = -1
                        del_box_list.append(idx_box)
                        box2d_thisbatch[idx_box][0] = 0
                        box2d_thisbatch[idx_box][1] = 0
                        box2d_thisbatch[idx_box][2] = 2
                        box2d_thisbatch[idx_box][3] = 2
                        continue
                    if torch.min(depth[idx_batch][idx_box]) < 0:
                        scores[idx_box] = -1
                        del_box_list.append(idx_box)
                        box2d_thisbatch[idx_box][0] = 0
                        box2d_thisbatch[idx_box][1] = 0
                        box2d_thisbatch[idx_box][2] = 2
                        box2d_thisbatch[idx_box][3] = 2
                        continue
                    box2d_thisbatch[idx_box][0] = ymin
                    box2d_thisbatch[idx_box][1] = xmin
                    box2d_thisbatch[idx_box][2] = ymax
                    box2d_thisbatch[idx_box][3] = xmax

                nms_after = torchvision.ops.nms(box2d_thisbatch, scores,
                                                iou_threshold=0.25)  # adopt the nms to decrease the number of boxes
                pred_box_corneres_nms_select = torch.index_select(pred_box_corneres[idx_batch], dim=0, index=nms_after)
                gt_box_corners_nms_select = torch.index_select(gt_box_corners[idx_batch], dim=0,
                                                               index=torch.nonzero(gt_box_present[idx_batch])[:, 0])

                if gt_box_corners_nms_select.shape[0] > 0:
                    pred_box_corneres_nms_select_points = torch.zeros((pred_box_corneres_nms_select.shape[0], 6),
                                                                      device=self.device)
                    pred_box_corneres_nms_select_points[:, 0] = torch.min(pred_box_corneres_nms_select[:, :, 0], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 1] = torch.min(pred_box_corneres_nms_select[:, :, 1], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 2] = torch.min(pred_box_corneres_nms_select[:, :, 2], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 3] = torch.max(pred_box_corneres_nms_select[:, :, 0], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 4] = torch.max(pred_box_corneres_nms_select[:, :, 1], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 5] = torch.max(pred_box_corneres_nms_select[:, :, 2], dim=1)[
                        0]

                    gt_box_corners_nms_select_points = torch.zeros((gt_box_corners_nms_select.shape[0], 6),
                                                                   device=self.device)
                    gt_box_corners_nms_select_points[:, 0] = torch.min(gt_box_corners_nms_select[:, :, 0], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 1] = torch.min(gt_box_corners_nms_select[:, :, 1], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 2] = torch.min(gt_box_corners_nms_select[:, :, 2], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 3] = torch.max(gt_box_corners_nms_select[:, :, 0], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 4] = torch.max(gt_box_corners_nms_select[:, :, 1], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 5] = torch.max(gt_box_corners_nms_select[:, :, 2], dim=1)[0]

                    select_boxes_idx = []
                    assert nms_after.shape[0] == pred_box_corneres_nms_select.shape[0]
                    for idx_box_pred in range(pred_box_corneres_nms_select_points.shape[0]):
                        match_gt = 0
                        pred_box_select = pred_box_corneres_nms_select_points[idx_box_pred]
                        for idx_box_gt in range(gt_box_corners_nms_select_points.shape[0]):

                            gt_box_select = gt_box_corners_nms_select_points[idx_box_gt]
                            iou_pred_gt = self.cal_iou(pred_box_select, gt_box_select)
                            if iou_pred_gt > 0.25:
                                match_gt = 1
                                break
                        if match_gt == 0:
                            select_boxes_idx.append(idx_box_pred)

                    nms_after_iou_select = nms_after[select_boxes_idx]
                else:
                    nms_after_iou_select = nms_after

                box_keep[idx_batch] = box_keep[idx_batch].index_fill(dim=0, index=nms_after_iou_select, value=1)
                box_keep[idx_batch][
                    scores < self.keep_objectness] = 0  # set the objectness to keep, only keep the box which have larger objectness
                for idx_box in del_box_list:
                    box_keep[idx_batch][idx_box] = 0

                novel_idx_list = nms_after_iou_select.cpu().numpy()
                novel_idx_list_final = []
                for novel_idx in novel_idx_list:
                    if box_keep[idx_batch][novel_idx] == 0:
                        continue
                    novel_idx_list_final.append(novel_idx)

                box_save[idx_batch] = box_save[idx_batch].index_fill(dim=0, index=nms_after_iou_select, value=1)
                box_save[idx_batch][
                    scores < self.save_objectness] = 0  # set the objectness to keep, only keep the box which have larger objectness
                for idx_box in del_box_list:
                    box_save[idx_batch][idx_box] = 0
                # for idx_nms in range(nms_after.shape[0]):
                #     idx_box = nms_after[idx_nms]
                #     for gt_idx_box in range(gt_box_corners.shape[1]):
                #
                #         print(1)
                novel_idx_list = nms_after_iou_select.cpu().numpy()
                novel_idx_save_final = []
                for novel_idx in novel_idx_list:
                    if box_save[idx_batch][novel_idx] == 0:
                        continue
                    novel_idx_save_final.append(novel_idx) # select the box idx for saving

                idx_maybe_novel_box_list = []
                proposal_maybe_novel_feature_list = []
                for idx_box in novel_idx_save_final:
                    img = inputs['input_image'][idx_batch]
                    box_size = outputs["size_unnormalized"][idx_batch][idx_box]
                    if torch.max(box_size) < 1e-16:  # when displaying gt, some of the box_size will be zeros
                        continue
                    corners_3d_image = center_2dpoints[idx_batch][idx_box]
                    xmin = int(torch.min(corners_3d_image[:, 0]))
                    ymin = int(torch.min(corners_3d_image[:, 1]))
                    xmax = int(torch.max(corners_3d_image[:, 0]))
                    ymax = int(torch.max(corners_3d_image[:, 1]))

                    if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                        img_crop = img[ymin:ymax, xmin:xmax]
                    else:
                        continue

                    if torch.min(depth[idx_batch][idx_box])<0:
                        continue

                    idx_maybe_novel_box_list.append(idx_box)

                    w = ymax - ymin
                    h = xmax - xmin

                    if w > h:
                        max_edge = w
                    else:
                        max_edge = h
                    #
                    img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8,
                                                device=self.device) * 255
                    # part 2

                    y_begin = (max_edge - w) // 2
                    x_begin = (max_edge - h) // 2
                    # try:
                    # print(img_crop.shape)
                    # print(x_begin)
                    # print(y_begin)
                    # print(x_begin + h)
                    # print(y_begin + w)
                    # print('============')
                    img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
                    # except:
                    # print(1)
                    # part 3
                    img_crop = img_beckground.permute(2, 0, 1)
                    img_crop_resized = self.resize(img_crop)  # change version
                    # print(img_crop_resized.shape)
                    # img_crop_resized = self.test_resize(img_crop)
                    # cv2.imwrite(os.path.join('./%s_%s_3d_to_2d_box_gt.png' % (idx_batch, idx_box)), img_crop_resized.permute(1, 2, 0).cpu().numpy())
                    img_crop_resized_expand = img_crop_resized.unsqueeze(0)
                    proposal_maybe_novel_feature_list.append(img_crop_resized_expand)

                # if len(proposal_feature_list) < 1:
                #     continue
                if len(proposal_maybe_novel_feature_list) > 0:
                    img_proposal = torch.cat(proposal_maybe_novel_feature_list, 0)

                    image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]  # change version
                    # image_proposal_batch_copy = image_proposal_batch.detach().clone()
                    # proposal_features_batch = self.res_encoder(image_proposal_batch, if_pool=True, if_early_feat=False)
                    proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)  # change version
                    if isinstance(proposal_features_batch, tuple):
                        proposal_features_batch = proposal_features_batch[0]
                    proposal_features_batch = proposal_features_batch.to(torch.float32)

                    text_correlation_embedding = proposal_features_batch / (
                            proposal_features_batch.norm(dim=-1, keepdim=True) + 1e-32)
                    maybe_novel_correlation_map = torch.mm(text_correlation_embedding,
                                                           text_features_clip.permute(1, 0)) * temperature_param
                    maybe_novel_scores = torch.nn.functional.softmax(maybe_novel_correlation_map, dim=-1)[:, :]
                    maybe_max_score, maybe_max_idx = torch.max(maybe_novel_scores, dim=-1)
                    novel_idx_list_final_select = []
                    maybe_condition = torch.logical_and(maybe_max_score > self.clip_driven_keep_thres,
                                                        maybe_max_idx >= self.train_range_max)
                    if (curr_epoch % self.online_nms_update_save_epoch == 0):
                        for maybe_idx in range(maybe_condition.shape[0]):
                            maybe_item = maybe_condition[maybe_idx]
                            if maybe_item:
                                novel_idx_list_final_select.append(idx_maybe_novel_box_list[maybe_idx])
                                save_box_info[idx_batch][idx_maybe_novel_box_list[maybe_idx]][7] = maybe_max_idx[maybe_idx]
                                save_box_info[idx_batch][idx_maybe_novel_box_list[maybe_idx]][8] = maybe_max_score[maybe_idx]
                                save_box_info[idx_batch][idx_maybe_novel_box_list[maybe_idx]][9] = scores[idx_maybe_novel_box_list[maybe_idx]] # update the novel box info

                    if (not if_test) and (
                            curr_epoch % self.online_nms_update_save_epoch == 0):
                        begin_idx = int((inputs["gt_ori_box_num"][idx_batch]).item())
                        novel_box_save_list = []
                        for novel_idx in novel_idx_list_final_select:
                            if begin_idx > 63:
                                print('Final gt boxs are more than 64')
                                break
                            if (curr_epoch % self.online_nms_update_save_epoch == 0):
                                novel_box_save_list.append(save_box_info[idx_batch][novel_idx])

                        if (curr_epoch % self.online_nms_update_save_epoch == 0) and len(
                                novel_box_save_list) > 0:
                            novel_box_path = inputs['pseudo_box_path'][idx_batch]
                            if not self.if_accumulate_former_pseudo_labels:
                                # novel_box_former = np.load(novel_box_path)
                                novel_box_save = np.array(novel_box_save_list)
                                # novel_box_final = np.concatenate(novel_box_save, axis=0)
                                np.save(novel_box_path, novel_box_save)
                            else:
                                novel_box_former = np.load(novel_box_path)
                                novel_box_later = np.array(novel_box_save_list)
                                if novel_box_former.shape[0] == 0: # no former boxes
                                    novel_box_final = novel_box_later
                                else: # have former boxes
                                    novel_box_final = np.concatenate((novel_box_former, novel_box_later), axis=0)
                                np.save(novel_box_path, novel_box_final) # accumulate the novel boxes

            proposal_feature_list = []
            idx_box_list = []
            # cv2.imwrite(os.path.join('./%s_3d_to_2d_box_image.png' % idx_batch), inputs['input_image'][idx_batch].cpu().numpy())
            # if_keep_box = torch.rand(size=(center_2dpoints.shape[1], 1), device=self.device, requires_grad=False)
            select_list = np.random.choice(self.box_idx_list, self.distillation_box_num, replace=False) # random select N boxes for distillation

            for idx_box in select_list:
                img = inputs['input_image'][idx_batch]
                box_size = outputs["size_unnormalized"][idx_batch][idx_box]
                if torch.max(box_size) < 1e-16:  # when displaying gt, some of the box_size will be zeros
                    # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                    # idx_box_list.append(0)
                    continue
                corners_3d_image = center_2dpoints[idx_batch][idx_box]
                xmin = int(torch.min(corners_3d_image[:, 0]))
                ymin = int(torch.min(corners_3d_image[:, 1]))
                xmax = int(torch.max(corners_3d_image[:, 0]))
                ymax = int(torch.max(corners_3d_image[:, 1]))

                if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                    img_crop = img[ymin:ymax, xmin:xmax]
                else:
                    continue
                if torch.min(depth[idx_batch][idx_box]) < 0:
                    continue

                idx_box_list.append(idx_box)

                w = ymax - ymin
                h = xmax - xmin

                if w > h:
                    max_edge = w
                else:
                    max_edge = h
                #
                img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8,
                                            device=self.device) * 255
                # part 2

                y_begin = (max_edge - w) // 2
                x_begin = (max_edge - h) // 2

                img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
                # except:
                # print(1)
                # part 3
                img_crop = img_beckground.permute(2, 0, 1)
                img_crop_resized = self.resize(img_crop)  # change version
                img_crop_resized_expand = img_crop_resized.unsqueeze(0)
                proposal_feature_list.append(img_crop_resized_expand)

            # if len(proposal_feature_list) < 1:
            #     continue
            try:
                img_proposal = torch.cat(proposal_feature_list, 0)
            except:
                print('No object on GT, too')
                continue

            image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]

            proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)  # get the 2D clip color image embedding

            if isinstance(proposal_features_batch, tuple):
                proposal_features_batch = proposal_features_batch[0]
            proposal_features_batch = proposal_features_batch.to(torch.float32)
            # tmp = outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list]
            outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list] = proposal_features_batch
            outputs['gt_text_correlation_embedding_mask'][idx_batch][idx_box_list] = 1

        if self.if_clip_weak_labels:
            text_features_clip = outputs["text_features_clip"].to(torch.float32)
            temperature_param = outputs["logit_scale"]
            gt_image_embedding = outputs['gt_text_correlation_embedding']
            gt_image_embedding_mask = outputs['gt_text_correlation_embedding_mask']
            gt_image_embedding = gt_image_embedding / (
                    gt_image_embedding.norm(dim=-1, keepdim=True) + 1e-32)
            correlation_map = torch.bmm(gt_image_embedding,
                                        text_features_clip.permute(0, 2, 1)) * temperature_param
            box_scores = torch.nn.functional.softmax(correlation_map, dim=-1)
            max_score, max_id = torch.max(box_scores, dim=-1)
            weak_confidence_weight = max_score
            weak_box_cate_label = max_id
            weak_confidence_weight[
                gt_image_embedding_mask[:, :,
                0] < 1] = 0.0  # confience_weight for bg object is set to 0, because it is no
            outputs["weak_box_cate_label"] = weak_box_cate_label
            outputs["weak_confidence_weight"] = weak_confidence_weight
        return outputs

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features, point_clouds, inputs):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        # seen_cls_logits = self.mlp_heads["seen_sem_cls_head"](box_features).transpose(1, 2)
        text_correlation_embedding = self.mlp_heads["text_correlation_head"](box_features).transpose(1, 2)
        center_offset = (
                self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        # seen_cls_logits = seen_cls_logits.reshape(num_layers, batch, num_queries, -1)
        text_correlation_embedding = text_correlation_embedding.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
                np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            box_corners_xyz = self.box_processor.box_parametrization_to_corners_xyz(
                center_unnormalized, size_unnormalized, angle_continuous
            )
            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                # "seen_sem_cls_logits": seen_cls_logits[l],
                "text_correlation_embedding": text_correlation_embedding[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
                "box_corners_xyz": box_corners_xyz,
                "point_clouds": point_clouds
            }
            # print(objectness_prob.shape)
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def get_class_scores(self, box_predictions):
        '''
        calculate multi-class scores by clip-text-embedding
        '''
        if self.eval_layer_id != -1:
            for tmp_key in box_predictions["aux_outputs"][self.eval_layer_id].keys():
                box_predictions["outputs"][tmp_key] = box_predictions["aux_outputs"][self.eval_layer_id][tmp_key]
        outputs = box_predictions["outputs"]

        text_features_clip = box_predictions["outputs"]["text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = box_predictions["outputs"]["logit_scale"]
        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (
                    text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1)) * temperature_param
        scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        box_predictions["outputs"]["sem_cls_prob"] = scores

        return box_predictions, box_predictions["outputs"]["sem_cls_prob"], box_predictions["outputs"][
            'objectness_prob']

    
    def forward(self, inputs, encoder_only=False, if_test=False, if_real_test=False, curr_epoch=-1, if_cmp_class=False):
        point_clouds = inputs["point_clouds"]
        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)

        if encoder_only:
            return enc_xyz, enc_features.transpose(0, 1)
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)

        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        # combined_enc_pos[:enc_pos.shape[0],:,:] = enc_pos
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features, point_clouds, inputs
        )

        box_predictions["outputs"]["logit_scale"] = torch.clip(self.logit_scale.exp(), min=None, max=100)


        if (not if_real_test) and (not if_cmp_class) and (not if_test): # do not calculate when testing
            if self.if_clip_superset: # if adopt lvis superset
                box_predictions["outputs"]["text_features_clip"] = self.superset_text_features_fg_norm.unsqueeze(
                    0).repeat(point_clouds.shape[0], 1, 1)  # text_features_clip
            else:
                box_predictions["outputs"]["text_features_clip"] = self.text_features_fg_norm[:self.train_range_max, :].unsqueeze(0).repeat(point_clouds.shape[0], 1, 1)  # text_features_clip

            if self.online_nms_update_save_novel_label_clip_driven_with_cate_confidence:
                if self.if_clip_superset:
                    box_predictions["outputs"]["maybe_novel_text_features_clip"] = self.superset_text_features_fg_norm
                else:
                    box_predictions["outputs"]["maybe_novel_text_features_clip"] = self.text_features_fg_norm[:self.test_range_max, :]  # text_features_clip
                box_predictions["outputs"] = self.get_predicted_box_clip_embedding_nms_iou_save_keep_clip_driven_with_cate_confidence(inputs,
                                                                                                box_predictions["outputs"],
                                                                                                curr_epoch=curr_epoch,
                                                                                                if_test=if_test)
            else:
                box_predictions["outputs"] = self.get_predicted_box_clip_embedding(inputs, box_predictions["outputs"],
                                                                                   curr_epoch=curr_epoch)
        if if_real_test:
            box_predictions["outputs"]["text_features_clip"] = self.text_features_fg_norm.unsqueeze(0).repeat(
                    point_clouds.shape[0], 1, 1)  # text_features_clip
            if self.if_with_clip:
                box_predictions["outputs"] = self.clip_to_class_training(inputs, box_predictions["outputs"],
                                                                if_use_gt_box=self.if_use_gt_box,
                                                                if_expand_box=self.if_expand_box)
            else:
                if self.if_with_fake_classes:
                    box_predictions, scores2, prob2 = self.get_class_scores_from_three(box_predictions)
                else:
                    # box_predictions, scores2, prob2 = self.get_class_scores_sigmoid(box_predictions)
                    box_predictions, scores2, prob2 = self.get_class_scores(box_predictions)


        return box_predictions




class Model3DETRMultiClassHead(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        if_with_clip=False,
        if_use_gt_box=False,
        if_expand_box=False,
        if_with_clip_embed=False,
        args=None
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.if_use_gt_box = if_use_gt_box
        self.if_expand_box = if_expand_box
        print('================If use GT box================')
        print(self.if_use_gt_box)
        print('================If expand box================')
        print(self.if_expand_box)
        self.if_second_text = False
        self.K = 20
        print('================If second test================')
        print(self.if_second_text)
        print(self.K)
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=decoder_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=decoder_dim, pos_type=position_embedding, normalize=True
        )
        self.query_projection = GenericMLP(
            input_dim=decoder_dim,
            hidden_dims=[decoder_dim],
            output_dim=decoder_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)
        self.if_with_clip = if_with_clip
        self.if_with_clip_embed = if_with_clip_embed
        self.if_adopt_region_embed = False
        # self.if_with_clip_train = if_with_clip_train
        self.args = args
        print('=============Adopt CLIP model to train the model=============')
        if args.dataset_name.find('scannet') != -1:
            class_id_to_idx = {2: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 13: 9, 14: 10, 15: 11,
                               16: 12,
                               17: 13, 18: 14, 19: 15, 21: 16, 22: 17, 23: 18, 24: 19, 26: 20, 27: 21, 28: 22,
                               29: 23,
                               31: 24, 32: 25, 33: 26, 34: 27, 35: 28, 36: 29, 38: 30, 39: 31, 40: 32, 41: 33,
                               42: 34,
                               44: 35, 45: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 54: 44,
                               55: 45,
                               56: 46, 57: 47, 58: 48, 59: 49, 62: 50, 63: 51, 64: 52, 65: 53, 66: 54, 67: 55,
                               68: 56,
                               69: 57, 70: 58, 71: 59, 72: 60, 73: 61, 74: 62, 75: 63, 76: 64, 77: 65, 78: 66,
                               79: 67,
                               80: 68, 82: 69, 84: 70, 86: 71, 87: 72, 88: 73, 89: 74, 90: 75, 93: 76, 95: 77,
                               96: 78,
                               97: 79, 98: 80, 99: 81, 100: 82, 101: 83, 102: 84, 103: 85, 104: 86, 105: 87,
                               106: 88,
                               107: 89, 110: 90, 112: 91, 115: 92, 116: 93, 118: 94, 120: 95, 121: 96, 122: 97,
                               125: 98, 128: 99, 130: 100, 131: 101, 132: 102, 134: 103, 136: 104, 138: 105,
                               139: 106,
                               140: 107, 141: 108, 145: 109, 148: 110, 154: 111, 155: 112, 156: 113, 157: 114,
                               159: 115, 161: 116, 163: 117, 165: 118, 166: 119, 168: 120, 169: 121, 170: 122,
                               177: 123, 180: 124, 185: 125, 188: 126, 191: 127, 193: 128, 195: 129, 202: 130,
                               208: 131, 213: 132, 214: 133, 221: 134, 229: 135, 230: 136, 232: 137, 233: 138,
                               242: 139, 250: 140, 261: 141, 264: 142, 276: 143, 283: 144, 286: 145, 300: 146,
                               304: 147, 312: 148, 323: 149, 325: 150, 331: 151, 342: 152, 356: 153, 370: 154,
                               392: 155, 395: 156, 399: 157, 408: 158, 417: 159, 488: 160, 540: 161, 562: 162,
                               570: 163, 572: 164, 581: 165, 609: 166, 748: 167, 776: 168, 1156: 169, 1163: 170,
                               1164: 171, 1165: 172, 1166: 173, 1167: 174, 1168: 175, 1169: 176, 1170: 177,
                               1171: 178,
                               1172: 179, 1173: 180, 1174: 181, 1175: 182, 1176: 183, 1178: 184, 1179: 185,
                               1180: 186,
                               1181: 187, 1182: 188, 1183: 189, 1184: 190, 1185: 191, 1186: 192, 1187: 193,
                               1188: 194,
                               1189: 195, 1190: 196, 1191: 197}
            train_idx_list = [class_id_to_idx[i] for i in args.train_range_list]
            tmp_test_range_list = args.train_range_list.copy()
            tmp_cnt = 0
            for tmp_class_id in args.test_range_list:
                if tmp_class_id in args.train_range_list:
                    continue
                tmp_test_range_list.append(tmp_class_id)
                tmp_cnt += 1
                if tmp_cnt >= args.reset_scannet_num:
                    break
            test_class_list = sorted(tmp_test_range_list)
            test_idx_list = [class_id_to_idx[i] for i in test_class_list]
            all_classes_keys = np.load(ALL_CLASS_PATH_SCANNET)
            all_cmp_classes = np.load(
                ALL_CMP_CLASS_PATH_SCANNET)  # [:args.test_range_max]all_classes_keys # waiting to change this part
            select_class_path = ALL_CLASS_PATH_SCANNET
            all_classes_keys = all_classes_keys[test_idx_list]
            self.class_num = len(all_classes_keys)
        else:
            if self.args.if_use_v1:
                all_classes = np.load(ALL_CLASS_PATH_V1, allow_pickle=True).item()
            else:
                all_classes = np.load(ALL_CLASS_PATH_V2, allow_pickle=True).item()
            all_cmp_classes = np.load(ALL_CMP_CLASS_PATH, allow_pickle=True)

        # self.bg_embed = torch.nn.Parameter(torch.zeros([1, 768]))
        # self.test_prob = torch.nn.Parameter(torch.zeros([1, dataset_config.num_semcls+1]))
            if args.if_only_novel_prompt:
                all_classes_keys = list(all_classes.keys())[10:37]
            else:
                all_classes_keys = list(all_classes.keys())[:args.test_range_max]#[:dataset_config.num_semcls]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        all_cmp_classes_keys = list(all_cmp_classes)
        self.test_range_max =  args.test_range_max
        print(len(all_classes_keys))

        print('========compare with cvpr paper=========')
        print(len(all_cmp_classes_keys))
        print(all_cmp_classes_keys)

        self.all_classes_keys = ['a photo of a '+i.replace('_', ' ').lower()+' in the scene' for i in all_classes_keys]
        self.all_cmp_classes_keys = ['a photo of a ' + i.replace('_', ' ').lower() + ' in the scene' for i in
                                 all_cmp_classes_keys]
        

        # self.all_classes_keys = [i.replace('_', ' ').lower() for i in all_classes_keys]
        # all_classes_keys_test = list(all_classes.keys())
        # print(len(all_classes_keys))
        # self.all_classes_keys_test = ['a photo of a '+i.replace('_', ' ').lower()+' in the scene' for i in all_classes_keys_test]
        # self.text = clip.tokenize(self.all_classes_keys).to(self.device)
        # self.clip_model, self.preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-B-32.pt",
        #                                                         device=self.device,
        #                                                         download_root='./CLIP/pretrain_models/',
        #                                                         if_transform_tensor=True)

        #######################ViT-L-14#########################
        # self.clip_model, self.preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-L-14.pt", device=self.device, download_root='./CLIP/pretrain_models/', if_transform_tensor=True)
        # # self.clip_model.to(self.device)
        # for param in self.clip_model.parameters():
        #     param.requires_grad = False
        # self.logit_scale = self.clip_model.logit_scale.exp()
        # # self.clip_model = self.clip_model.eval()
        # # with torch.no_grad():
        # # with torch.no_grad():
        # self.text = clip.tokenize(self.all_classes_keys).to(self.device)
        # # self.text_features_bg = torch.from_numpy(np.load('bg_embedding.npy')).to(self.device).unsqueeze(0)
        # self.text_features_fg = self.clip_model.encode_text(self.text)
        # self.resize = Resize(self.clip_model.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        #######################ViT-L-14#########################

        self.if_clip_superset = args.if_clip_superset
        print('====================if clip superset====================')
        print(self.if_clip_superset)
        if self.if_clip_superset:
            superset_all_classes = np.load(ALL_SUPERCLASS_PATH, allow_pickle=True)
            superset_all_classes_keys = list(superset_all_classes)[1:]  # the  first one is "name"

            tmp_superset = ['a photo of a ' + i.replace('_', ' ').lower() + ' in the scene' for i in
                            superset_all_classes_keys]

            self.superset_all_classes_keys = []

            cnt_in = 0
            if args.dataset_name.find('scannet') != -1:
                for idx_class in dataset_config.seen_idx_list:
                    i = self.all_classes_keys[idx_class]  # not wrong for scannet
                    # for i in self.all_classes_keys[:10]: # only the seen classes
                    if i not in self.superset_all_classes_keys:
                        # print(i)
                        self.superset_all_classes_keys.append(i)
                        cnt_in += 1
            else:
                for i in self.all_classes_keys[:10]:  # only the seen classes
                    if i not in self.superset_all_classes_keys:
                        # print(i)
                        self.superset_all_classes_keys.append(i)
                        cnt_in += 1

            # assert len(self.superset_all_classes_keys) == 37

            cnt_same = 0
            for i in tmp_superset:
                #     # text_item = 'a photo of a ' + i.replace('_', ' ').lower() + ' in the scene'
                if i not in self.superset_all_classes_keys:
                    # print(i)
                    self.superset_all_classes_keys.append(i)
                else:
                    cnt_same += 1

            print('=====================class superset======================')
            # save_dic = {item: idx for idx, item in enumerate(self.superset_all_classes_keys)}
            assert len(self.superset_all_classes_keys) == 1201 or len(self.superset_all_classes_keys) == 1216 or len(
                self.superset_all_classes_keys) == 232 or len(self.superset_all_classes_keys) == 1203
            print(self.superset_all_classes_keys)
            print(len(self.superset_all_classes_keys))


        # # #########################ResNet#########################
        self.clip_model, self.preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-B-16.pt", #"./CLIP/pretrain_models/RN50x64.pt", #"./CLIP/pretrain_models/ViT-L-14.pt",
                                                                device=self.device,
                                                                download_root='./CLIP/pretrain_models/',
                                                                if_transform_tensor=True)
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        self.logit_scale = self.clip_model.logit_scale.exp()
        # self.clip_model = self.clip_model.eval()
        # with torch.no_grad():
        # with torch.no_grad():
        self.text = clip.tokenize(self.all_classes_keys).to(self.device)
        # self.text_features_bg = torch.from_numpy(np.load('bg_embedding.npy')).to(self.device).unsqueeze(0)
        self.text_features_fg = self.clip_model.encode_text(self.text)
        self.text_features_fg_norm = self.text_features_fg / self.text_features_fg.norm(dim=1, keepdim=True)
        self.text_features_fg_norm = self.text_features_fg_norm.to(torch.float32)

        self.cmp_text = clip.tokenize(self.all_cmp_classes_keys).to(self.device)
        # self.text_features_bg = torch.from_numpy(np.load('bg_embedding.npy')).to(self.device).unsqueeze(0)
        self.cmp_text_features_fg = self.clip_model.encode_text(self.cmp_text)
        self.cmp_text_features_fg_norm = self.cmp_text_features_fg / self.cmp_text_features_fg.norm(dim=1, keepdim=True)

        if self.if_clip_superset:
            self.superset_text = clip.tokenize(self.superset_all_classes_keys).to(self.device)
            self.superset_text_features_fg = self.clip_model.encode_text(self.superset_text).to(torch.float32)
            self.superset_text_features_fg_norm = self.superset_text_features_fg / self.superset_text_features_fg.norm(
                dim=1, keepdim=True)
            self.superset_text_features_fg_norm = self.superset_text_features_fg_norm.to(torch.float32)

        self.resize = Resize(self.clip_model.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        # # #########################ResNet#########################

        # self.text_features_fg.requires_grad = True

        # self.text_test = clip.tokenize(self.all_classes_keys_test).to(self.device)
        # self.text_features_fg_test = self.clip_model.encode_text(self.text_test)
        # self.text_features_fg_test.requires_grad = True
        # self.clip_model, self.preprocess_for_tensor = clip.load("./CLIP/pretrain_models/ViT-L-14-336px.pt",
        #                                                         device=self.device,
        #                                                         download_root='./CLIP/pretrain_models/',
        #                                                         if_transform_tensor=True)
        # self.clip_model, self.preprocess = clip.load("./CLIP/pretrain_models/ViT-L-14.pt", device=self.device, download_root='./CLIP/pretrain_models/',
        #                                                         if_transform_tensor=False)
        print('All the class for CLIP:')
        print(self.all_classes_keys)
        #if self.if_with_clip_embed:
        print('if_with_clip_embed:')
        print(self.if_with_clip_embed)
        self.train_range_max = args.train_range_max
        print('====================dataset name=========================')
        if args.dataset_name.find('scannet') != -1:
            self.dataset_name = 'scannet'
            self.dataset_util = scannet_utils
        else:
            self.dataset_name = 'sunrgbd'
            self.dataset_util = sunrgbd_utils
        print(self.dataset_name)
        print('=========================if online_nms_update_save_novel_label_clip_driven_with_cate_confidence')
        self.online_nms_update_save_novel_label_clip_driven_with_cate_confidence = args.online_nms_update_save_novel_label_clip_driven_with_cate_confidence
        self.save_objectness = args.save_objectness
        self.clip_driven_keep_thres = args.clip_driven_keep_thres
        print(self.online_nms_update_save_novel_label_clip_driven_with_cate_confidence)
        print(self.save_objectness)
        self.online_nms_update_save_epoch = args.online_nms_update_save_epoch
        print(self.online_nms_update_save_epoch)
        print(self.clip_driven_keep_thres)
        print('=====================if accumulate former pseudo labels========================')
        self.if_accumulate_former_pseudo_labels = args.if_accumulate_former_pseudo_labels
        print(self.if_accumulate_former_pseudo_labels)

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        print('------------Predict %d classes(including background/not-an-object class)------------' % (dataset_config.num_semcls + 1))
        semcls_head = mlp_func(output_dim=1 + 1)

        # geometry of the box
        center_head = mlp_func(output_dim=3)
        size_head = mlp_func(output_dim=3)
        angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
        angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

        # seen_semcls_head = mlp_func(output_dim=37 + 1)
        mlp_heads = [
            ("sem_cls_head", semcls_head),
            # ("seen_sem_cls_head", seen_semcls_head),
            ("center_head", center_head),
            ("size_head", size_head),
            ("angle_cls_head", angle_cls_head),
            ("angle_residual_head", angle_reg_head),
        ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds)
        return enc_xyz, enc_features, enc_inds

    # def _affine_transform(self, points, matrix):
    #     """Affine transform bbox points to input image.
    #     Args:
    #         points (np.ndarray): Points to be transformed.
    #             shape: (N, 2)
    #         matrix (np.ndarray): Affine transform matrix.
    #             shape: (3, 3)
    #     Returns:
    #         np.ndarray: Transformed points.
    #     """
    #     num_points = points.shape[0]
    #     hom_points_2d = np.concatenate((points, np.ones((num_points, 1))),
    #                                    axis=1)
    #     hom_points_2d = hom_points_2d.T
    #     affined_points = np.matmul(matrix, hom_points_2d).T
    #     return affined_points[:, :2]
    def my_entropy(self, probs):
        # print(torch.sum(probs))
        # print(probs)
        probs_log = torch.log(probs+1e-7) # prevent inf value
        # print(probs_log)
        # print(probs_log.shape)
        entro_res = torch.sum(-1 * probs_log * probs)
        return entro_res

    def clip_to_class_embedding(self, inputs, outputs, thres_obj=0.05, if_use_gt_box=False, if_expand_box=False, if_padding_input=True): # the max number of classes are 18(train) and 14(val)


        if if_use_gt_box:
        # ceiling experiment

            box_centers_batch = inputs['gt_box_centers'].cpu().numpy()
            box_angle_batch = inputs['gt_box_angles'].cpu().numpy()
            box_size_batch = inputs['gt_box_sizes'].cpu().numpy()
            batch_size = box_size_batch.shape[0]
            # print(batch_size)
            outputs['center_unnormalized'] = inputs['gt_box_centers']
            outputs['angle_continuous'] = inputs['gt_box_angles']
            outputs['size_unnormalized'] = inputs['gt_box_sizes']
            outputs["box_corners"] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, 690)).to('cuda')
            outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, 238)).to('cuda')
            outputs['objectness_prob'] = inputs["gt_box_present"].clone().detach() #torch.ones((8, 64)).to('cuda')

            # print(outputs['objectness_prob'].shape)
            # outputs["sem_cls_logits"] =  torch.zeros((8, 64, 690)).to('cuda')
            x_offset_batch = inputs['x_offset'].cpu().numpy()
            y_offset_batch = inputs['y_offset'].cpu().numpy()
            ori_width_batch = inputs['ori_width'].cpu().numpy()
            ori_height_batch = inputs['ori_height'].cpu().numpy()
            # outputs["center_normalized"] = inputs["gt_box_centers_normalized"] + 1
            # outputs["size_normalized"] = inputs["gt_box_sizes_normalized"]
            # outputs["angle_logits"] = inputs["gt_box_angles"]
            # outputs["angle_residual"] = inputs["gt_angle_residual_label"]
            # # outputs["angle_residual_normalized"] angle_residual_normalized[l],
            # # outputs["objectness_prob"] objectness_prob,
            # outputs["point_clouds"] = inputs["point_clouds"]
            # outputs['center_unnormalized'][:,  :64] = box_centers_batch
            # # print(box_centers_batch)
            # outputs['angle_continuous'][:,  :64] = box_angle_batch
            # outputs['size_unnormalized'][:,  :64] = box_size_batch
            # # print(box_size_batch)
            # outputs["box_corners"][:,  :64] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'][:,  :64] = torch.zeros((8, 64, 690)).to('cuda')
            # outputs['objectness_prob'][:,  :64] = inputs["gt_box_present"]

            # outputs['center_unnormalized'][:, 64:] = box_centers_batch
            # outputs['angle_continuous'][:, 64:] = box_angle_batch
            # outputs['size_unnormalized'][:, 64:] = box_size_batch
            # outputs["box_corners"][:, 64:] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'][:, 64:] = torch.zeros((8, 64, 690)).to('cuda')
            # outputs['objectness_prob'][:, 64:] = inputs["gt_box_present"]
        # ceiling experiment
        else:
            box_centers_batch = outputs[
                'center_unnormalized'].cpu().numpy()  # .cpu().numpy()#inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
            # print(box_centers.shape)
            box_angle_batch = outputs[
                'angle_continuous'].cpu().numpy()  # .cpu().numpy() #inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
            # print(box_angle.shape)
            box_size_batch = outputs[
                'size_unnormalized'].cpu().numpy()  # .cpu().numpy() #inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
            x_offset_batch = inputs['x_offset'].cpu().numpy()
            y_offset_batch = inputs['y_offset'].cpu().numpy()
            ori_width_batch = inputs['ori_width'].cpu().numpy()
            ori_height_batch = inputs['ori_height'].cpu().numpy()

        max_entropy = np.log(outputs['sem_cls_prob'].shape[-1])

        # print('=============inputs key=======================')
        # print(inputs.keys())
        img_batch_name = inputs['im_name']
        calib_name_list = inputs['calib_name']
        trans_mtx_batch = inputs['trans_mtx'].cpu().numpy()
        # print(trans_mtx_batch.shape)
        input_image_batch = inputs["input_image"]

        for idx_batch in range(len(calib_name_list)):
            # box_center_list = box_centers_batch[idx_batch]
            # print(box_angle.shape)


            calib_name = calib_name_list[idx_batch]
            img = input_image_batch[idx_batch]
            # img = input_image.permute(2, 0, 1)

            # input_im = Image.fromarray(input_image[:,:,::-1])
            # input_im_ = np.array(input_im)
            # img_ = Image.open(img_batch_name[idx_batch])
            # img_ = np.array(img)
            # if_seem = np.all(np.equal(img_, input_im_))
            # img_ = cv2.imread(img_batch_name[idx_batch])
            calib = sunrgbd_utils.SUNRGBD_Calibration(calib_name)
            box_center_list = box_centers_batch[idx_batch]
            box_angle_list = box_angle_batch[idx_batch]
            box_size_list = box_size_batch[idx_batch]
            trans_mtx = trans_mtx_batch[idx_batch]
            x_offset = x_offset_batch[idx_batch]
            y_offset = y_offset_batch[idx_batch]

            for idx_box in range(len(box_center_list)):
                prob_list = []
                entropy_list = []
                ori_width = ori_width_batch[idx_batch]
                ori_height = ori_height_batch[idx_batch]
                if outputs['objectness_prob'][idx_batch, idx_box] < thres_obj:
                    continue
                box_center = box_center_list[idx_box]
                box_angle = box_angle_list[idx_box]
                box_size = box_size_list[idx_box]
                # trans_mtx = trans_mtx_list[idx_box]

                if np.max(box_size)<1e-16: # when displaying gt, some of the box_size will be zeros
                    continue
                # print(idx_box)
                if if_padding_input:
                    corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_offset(box_center, box_angle, box_size,
                                                                                       calib, x_offset, y_offset)
                    xmin_ = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    ymin_ = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                    xmax_ = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    ymax_ = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                else:
                    corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                       calib, trans_mtx)
                    xmin_ = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymin_ = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                    xmax_ = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymax_ = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))
                # print(corners_3d_image.shape) # 8, 2
                # corners_3d_image = self._affine_transform(corners_3d_image_, trans_affine)
                # print(img.size)
                # print(img_.shape)
                # xmin_ = int(min(max(np.min(corners_3d_image_[:,0]), 0), img_.size[1])) # the w, h of Iamge.open is diff from cv2.imread
                # ymin_ = int(min(max(np.min(corners_3d_image_[:,1]), 0), img_.size[0]))
                # xmax_ = int(min(max(np.max(corners_3d_image_[:,0]), 0), img_.size[1]))
                # ymax_ = int(min(max(np.max(corners_3d_image_[:,1]), 0), img_.size[0]))
                # print(corners_3d_image[:,0])
                # print(img.shape)
                # xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                # ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                # xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                # ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))

                if if_expand_box:
                    w = ymax - ymin
                    h = xmax - xmin
                    if w > h:
                        xmin = xmin - (w-h)//2
                        xmax = xmax + (w-h)//2
                    else:
                        ymin = ymin - (h-w)//2
                        ymax = ymax + (h-w)//2
                    xmin = int(min(max(xmin, 0), img.shape[1]))
                    ymin = int(min(max(ymin, 0), img.shape[0]))
                    xmax = int(min(max(xmax, 0), img.shape[1]))
                    ymax = int(min(max(ymax, 0), img.shape[0]))


                if (xmax_-xmin_)<=0 and (ymax_-ymin_)<=0:
                    print('Box size is less than 0')
                    continue

                w = ymax_ - ymin_
                h = xmax_ - xmin_
                shift_scale = 1.0/4
                for xmin_shift_off in [0, 1, -1]:
                    for xmax_shift_off in [0]: #[0, 1, -1]:
                        for ymin_shift_off in [0, 1, -1]:
                            for ymax_shift_off in [0]: #in [0, 1, -1]:  #  too slow
                                # xmin = xmin + shift_scale * xmin_shift_off * h
                                # ymin = ymin + shift_scale * ymin_shift_off * w
                                # xmax = xmax + shift_scale * xmax_shift_off * h
                                # ymax = ymax + shift_scale * ymax_shift_off * w

                                xmin = xmin_ + shift_scale * xmin_shift_off * h
                                ymin = ymin_ + shift_scale * ymin_shift_off * w
                                xmax = xmax_ + shift_scale * xmin_shift_off * h
                                ymax = ymax_ + shift_scale * ymin_shift_off * w

                                xmin = int(min(max(xmin, 0), img.shape[1]))
                                ymin = int(min(max(ymin, 0), img.shape[0]))
                                xmax = int(min(max(xmax, 0), img.shape[1]))
                                ymax = int(min(max(ymax, 0), img.shape[0]))

                                img_crop = img[ymin:ymax, xmin:xmax]
                #####exp1: blank background#####

                                w_shift = ymax - ymin
                                h_shift = xmax - xmin
                                if w_shift > h_shift:
                                    max_edge = w_shift
                                else:
                                    max_edge = h_shift

                                img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8) * 255
                # print(torch.max(img_beckground))
                                y_begin = (max_edge - w_shift) // 2
                                x_begin = (max_edge - h_shift) // 2
                                img_beckground[y_begin:y_begin+w_shift, x_begin:x_begin+h_shift, :] = img_crop

                # cv2.imwrite(os.path.basename(img_batch_name[idx_batch])[:-4] + '_' + str(idx_box) + '_2.png', img_crop_show)
                                img_crop = img_beckground.permute(2, 0, 1)
                # img_crop = img_crop.permute(2, 0, 1)
                # print(torch.max(img_crop))
                # print(np.max(np.array(img_crop_.convert('RGB'))))
                                image = self.preprocess_for_tensor(img_crop).unsqueeze(0).to(self.device)
                # image_ = self.preprocess(img_crop_).unsqueeze(0).to(self.device)
                # print(torch.max(image))
                # print(torch.max(image_))

                # image_features = clip_model.encode_image(image)
                # text_features = clip_model.encode_text(text)
                # print(torch.max(image))
                                logits_per_image, logits_per_text = self.clip_model(image, self.text)
                                probs = logits_per_image.softmax(dim=-1)#.cpu().numpy()
                                # print(torch.max(probs)) # [1, 238]
                                # print((probs))  # [1, 238]
                                # entropy_item = Categorical(probs = (probs+1e-16)).entropy() #output value range: [0, log(n)], input value range (0,
                                entropy_item = self.my_entropy(probs)
                                #print(entropy_item.shape) # [1]
                                entropy_list.append(entropy_item.unsqueeze(0))
                                prob_list.append(probs)
                                # print(torch.max(entropy_item))
                                # print(np.log(238))
                              #  print(max_entropy)

                                # print(torch.min(entropy_item))
                # print(entropy_list[0].shape)
                # print(entropy_list[0])
                entropy_list_tensor = torch.cat(entropy_list, dim=0).unsqueeze(-1)
                prob_list_tensor = torch.cat(prob_list, dim=0)
                # print(entropy_list_tensor.shape) # (81, 1)
                # print(prob_list_tensor.shape) # (81, 238)

                weight_tensor = max_entropy - entropy_list_tensor #value range: [0, log(n)], less entropy, bigger weights
                # print(torch.sum(weight_tensor))
                weighted_prob = torch.sum(weight_tensor * prob_list_tensor, dim=0) / torch.sum(weight_tensor)
                # print(weighted_prob.shape)
                # print(torch.sum(weighted_prob))
                # print(torch.max(weighted_prob))
                # print(torch.min(weighted_prob))

                # print(probs.shape)
                # print(torch.max(probs, dim=1)[0])
                # print(torch.max(probs, dim=1)[1])
                # print(probs[0])
                # print(probs[0].shape)
                # print()
                # logits_per_image, logits_per_text = self.clip_model(image_, self.text)
                # probs = logits_per_image.softmax(dim=-1)#.cpu().numpy()
                # print(probs.shape)
                # print(torch.max(probs, dim=1)[0])
                # print(torch.max(probs, dim=1)[1])
                # print('======================================================')
                # print(probs[0])
                # print(probs[0].shape)
                # print(len(box_center_list))
                # print(outputs['sem_cls_prob'].shape)
                # print(outputs['sem_cls_prob'][idx_batch, idx_box].shape)
                # print(probs[0].shape)
                if self.if_second_text:
                    top_k_value = torch.topk(probs[0], self.K, dim=0, largest=True, sorted=False)
                    # top_k_value_pro = top_k_value.values
                    top_k_value_indices = top_k_value.indices

                    all_classes_keys_second_order = []
                    for idx in top_k_value_indices:
                        all_classes_keys_second_order.append(self.all_classes_keys[idx])

                    text_second_order = clip.tokenize(all_classes_keys_second_order).to(self.device)
                    logits_per_image, logits_per_text = self.clip_model(image, text_second_order)
                    probs_second = torch.zeros_like(probs)
                    probs_second[0][top_k_value_indices] = logits_per_image.softmax(dim=-1)
                    probs = probs_second
                    # print(probs[0])
                    # print(probs[0][top_k_value_indices])
                    # print(torch.sum(probs[0][top_k_value_indices]))
                    # print(probs_second.shape)
                    # print(probs_second[0])
                    # print(probs_second[0][top_k_value_indices])
                    # print(torch.sum(probs_second[0][top_k_value_indices]))
                # print(top_k_value_pro.shape)
                # print(top_k_value_indices.shape)
                outputs['sem_cls_prob'][idx_batch, idx_box] = weighted_prob
                # print(outputs['sem_cls_prob'][idx_batch, idx_box])
                # print(torch.argmax(outputs['sem_cls_prob'][idx_batch, idx_box], dim=-1))
        return outputs

    def clip_to_class(self, inputs, outputs, thres_obj=0.05, if_use_gt_box=False, if_expand_box=False, if_padding_input=True): # the max number of classes are 18(train) and 14(val)


        if if_use_gt_box:
        # ceiling experiment

            box_centers_batch = inputs['gt_box_centers'].cpu().numpy()
            box_angle_batch = inputs['gt_box_angles'].cpu().numpy()
            box_size_batch = inputs['gt_box_sizes'].cpu().numpy()
            batch_size = box_size_batch.shape[0]
            # print(batch_size)
            outputs['center_unnormalized'] = inputs['gt_box_centers']
            outputs['angle_continuous'] = inputs['gt_box_angles']
            outputs['size_unnormalized'] = inputs['gt_box_sizes']
            outputs["box_corners"] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, 690)).to('cuda')
            outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, 238)).to('cuda')
            outputs['objectness_prob'] = inputs["gt_box_present"].clone().detach() #torch.ones((8, 64)).to('cuda')
            x_offset_batch = inputs['x_offset'].cpu().numpy()
            y_offset_batch = inputs['y_offset'].cpu().numpy()
            ori_width_batch = inputs['ori_width'].cpu().numpy()
            ori_height_batch = inputs['ori_height'].cpu().numpy()
            # print(outputs['objectness_prob'].shape)
            # outputs["sem_cls_logits"] =  torch.zeros((8, 64, 690)).to('cuda')
            # outputs["center_normalized"] = inputs["gt_box_centers_normalized"] + 1
            # outputs["size_normalized"] = inputs["gt_box_sizes_normalized"]
            # outputs["angle_logits"] = inputs["gt_box_angles"]
            # outputs["angle_residual"] = inputs["gt_angle_residual_label"]
            # # outputs["angle_residual_normalized"] angle_residual_normalized[l],
            # # outputs["objectness_prob"] objectness_prob,
            # outputs["point_clouds"] = inputs["point_clouds"]
            # outputs['center_unnormalized'][:,  :64] = box_centers_batch
            # # print(box_centers_batch)
            # outputs['angle_continuous'][:,  :64] = box_angle_batch
            # outputs['size_unnormalized'][:,  :64] = box_size_batch
            # # print(box_size_batch)
            # outputs["box_corners"][:,  :64] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'][:,  :64] = torch.zeros((8, 64, 690)).to('cuda')
            # outputs['objectness_prob'][:,  :64] = inputs["gt_box_present"]

            # outputs['center_unnormalized'][:, 64:] = box_centers_batch
            # outputs['angle_continuous'][:, 64:] = box_angle_batch
            # outputs['size_unnormalized'][:, 64:] = box_size_batch
            # outputs["box_corners"][:, 64:] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'][:, 64:] = torch.zeros((8, 64, 690)).to('cuda')
            # outputs['objectness_prob'][:, 64:] = inputs["gt_box_present"]
        # ceiling experiment
        else:
            box_centers_batch = outputs[
                'center_unnormalized'].cpu().numpy()  # .cpu().numpy()#inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
            # print(box_centers.shape)
            box_angle_batch = outputs[
                'angle_continuous'].cpu().numpy()  # .cpu().numpy() #inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
            # print(box_angle.shape)
            box_size_batch = outputs[
                'size_unnormalized'].cpu().numpy()  # .cpu().numpy() #inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
            x_offset_batch = inputs['x_offset'].cpu().numpy()
            y_offset_batch = inputs['y_offset'].cpu().numpy()
            ori_width_batch = inputs['ori_width'].cpu().numpy()
            ori_height_batch = inputs['ori_height'].cpu().numpy()


        # print('=============inputs key=======================')
        # print(inputs.keys())
        img_batch_name = inputs['im_name']
        calib_name_list = inputs['calib_name']
        trans_mtx_batch = inputs['trans_mtx'].cpu().numpy()
        # print(trans_mtx_batch.shape)
        input_image_batch = inputs["input_image"]

        for idx_batch in range(len(calib_name_list)):
            # box_center_list = box_centers_batch[idx_batch]
            # print(box_angle.shape)

            x_offset = x_offset_batch[idx_batch]
            y_offset = y_offset_batch[idx_batch]

            calib_name = calib_name_list[idx_batch]
            img = input_image_batch[idx_batch]
            # img = input_image.permute(2, 0, 1)

            # input_im = Image.fromarray(input_image[:,:,::-1])
            # input_im_ = np.array(input_im)
            # img_ = Image.open(img_batch_name[idx_batch])
            # img_ = np.array(img)
            # if_seem = np.all(np.equal(img_, input_im_))
            # img_ = cv2.imread(img_batch_name[idx_batch])
            calib = sunrgbd_utils.SUNRGBD_Calibration(calib_name)
            box_center_list = box_centers_batch[idx_batch]
            box_angle_list = box_angle_batch[idx_batch]
            box_size_list = box_size_batch[idx_batch]
            trans_mtx = trans_mtx_batch[idx_batch]
            ori_width = ori_width_batch[idx_batch]
            ori_height = ori_height_batch[idx_batch]

            for idx_box in range(len(box_center_list)):
                if outputs['objectness_prob'][idx_batch, idx_box] < thres_obj:
                    continue
                box_center = box_center_list[idx_box]
                box_angle = box_angle_list[idx_box]
                box_size = box_size_list[idx_box]
                # trans_mtx = trans_mtx_list[idx_box]

                if np.max(box_size)<1e-16: # when displaying gt, some of the box_size will be zeros
                    continue
                # print(idx_box)
                if if_padding_input:
                    corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_offset(box_center, box_angle, box_size,
                                                                                       calib, x_offset, y_offset)
                    xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                    xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                else:
                    corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                       calib, trans_mtx)
                    xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                    xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))
              #  corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size, calib, trans_mtx)
                # print(corners_3d_image.shape) # 8, 2
                # corners_3d_image = self._affine_transform(corners_3d_image_, trans_affine)
                # print(img.shape)
                # print(img_.size)
                # corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d(box_center, box_angle, box_size,
                #                                                                        calib)

                # print(img_.size)
                # xmin_ = int(min(max(np.min(corners_3d_image[:,0]), 0), img_.size[0])) # the w, h of Iamge.open is diff from cv2.imread
                # ymin_ = int(min(max(np.min(corners_3d_image[:,1]), 0), img_.size[1]))
                # xmax_ = int(min(max(np.max(corners_3d_image[:,0]), 0), img_.size[0]))
                # ymax_ = int(min(max(np.max(corners_3d_image[:,1]), 0), img_.size[1]))
                # print(xmin_)
                # print(xmax_)
                # print(ymin_)
                # print(ymax_)
                # print(corners_3d_image[:,0])
                # print(img.shape)


                if if_expand_box:
                    w = ymax-ymin
                    h = xmax-xmin
                    if w > h:
                        xmin = xmin - (w-h)//2
                        xmax = xmax + (w-h)//2
                    else:
                        ymin = ymin - (h-w)//2
                        ymax = ymax + (h-w)//2
                    xmin = int(min(max(xmin, 0), img.shape[1]))
                    ymin = int(min(max(ymin, 0), img.shape[0]))
                    xmax = int(min(max(xmax, 0), img.shape[1]))
                    ymax = int(min(max(ymax, 0), img.shape[0]))

                if (xmax-xmin)>0 and (ymax-ymin)>0:
                    img_crop = img[ymin:ymax, xmin:xmax]
                    # img_crop_show = img_crop.cpu().numpy()
                    # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
                    # cv2.imwrite(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'_2.png', img_crop_show)
                    # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
                    # continue
                else:
                    print('Box size is less than 0')
                    continue

                # if  (xmax_-xmin_)>5 and (ymax_-ymin_)>5:
                #     # img_crop = img[ymin:ymax, xmin:xmax]
                #     # img_crop_show = img_crop.cpu().numpy()
                #     # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
                #     img_crop_ = img_.crop(
                #         (xmin_, ymin_, xmax_, ymax_))  # the w, h of Iamge.open is diff from cv2.imread
                #     # cv2.imwrite(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'_2.png', img_crop_show)
                #     # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
                #     # continue
                # else:
                #     print('Image.open Box size is less than 5')
                #     continue

                #####exp1: blank background#####
                # img_crop_ = img_crop.clone()
                w = ymax - ymin
                h = xmax - xmin
                if w > h:
                    max_edge = w
                else:
                    max_edge = h
                # max_edge = max_edge+20
                # print(img_crop.shape)
                # print(torch.max(img_crop))
                # print(torch.min(img_crop))
                img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8) * 255
                # print(torch.max(img_beckground))
                y_begin = (max_edge - w) // 2
                x_begin = (max_edge - h) // 2
                img_beckground[y_begin:y_begin+w, x_begin:x_begin+h, :] = img_crop
                # print(img_beckground.shape)
                # print(torch.max(img_beckground))
                # print(torch.min(img_beckground))
                ################################
                # img_crop_show = img_crop.cpu().numpy()
                # cv2.imwrite(os.path.basename(img_batch_name[idx_batch])[:-4] + '_' + str(idx_box) + '_2.png', img_crop_show)
                img_crop = img_beckground.permute(2, 0, 1)
                # img_crop = img_crop.permute(2, 0, 1)
                # print(torch.max(img_crop))
                # print(np.max(np.array(img_crop_.convert('RGB'))))
                image = self.preprocess_for_tensor(img_crop).unsqueeze(0).to(self.device)
                # print(img_crop_.size)

                # w_ = ymax_ - ymin_
                # h_ = xmax_ - xmin_
                # if w_ > h_:
                #     max_edge_ = w_
                # else:
                #     max_edge_ = h_
                # img_background_ = Image.new('RGB', (max_edge_, max_edge_), (255, 255, 255))
                # y_begin_ = (max_edge_ - w_) // 2
                # x_begin_ = (max_edge_ - h_) // 2
                # img_background_.paste(img_crop_, (x_begin_, y_begin_))
                # image_ = self.preprocess(img_background_).unsqueeze(0).to(self.device)
                # print(torch.max(image))
                # print(torch.max(image_))

                # image_features = clip_model.encode_image(image)
                # text_features = clip_model.encode_text(text)
                # print(torch.max(image))
                logits_per_image, logits_per_text = self.clip_model(image, self.text)
                probs = logits_per_image.softmax(dim=-1)#.cpu().numpy()
                # print(probs.shape)
                # print(torch.max(probs, dim=1)[0])
                # print(torch.max(probs, dim=1)[1])
                # print(probs[0])
                # print(probs[0].shape)
                # print()
                # logits_per_image, logits_per_text = self.clip_model(image_, self.text)
                # probs = logits_per_image.softmax(dim=-1)#.cpu().numpy()
                # print(probs.shape)
                # print(torch.max(probs, dim=1)[0])
                # print(torch.max(probs, dim=1)[1])
                # print('======================================================')
                # print(probs[0])
                # print(probs[0].shape)
                # print(len(box_center_list))
                # print(outputs['sem_cls_prob'].shape)
                # print(outputs['sem_cls_prob'][idx_batch, idx_box].shape)
                # print(probs[0].shape)
                if self.if_second_text:
                    top_k_value = torch.topk(probs[0], self.K, dim=0, largest=True, sorted=False)
                    # top_k_value_pro = top_k_value.values
                    top_k_value_indices = top_k_value.indices

                    all_classes_keys_second_order = []
                    for idx in top_k_value_indices:
                        all_classes_keys_second_order.append(self.all_classes_keys[idx])

                    text_second_order = clip.tokenize(all_classes_keys_second_order).to(self.device)
                    logits_per_image, logits_per_text = self.clip_model(image, text_second_order)
                    probs_second = torch.zeros_like(probs)
                    probs_second[0][top_k_value_indices] = logits_per_image.softmax(dim=-1)
                    probs = probs_second
                    # print(probs[0])
                    # print(probs[0][top_k_value_indices])
                    # print(torch.sum(probs[0][top_k_value_indices]))
                    # print(probs_second.shape)
                    # print(probs_second[0])
                    # print(probs_second[0][top_k_value_indices])
                    # print(torch.sum(probs_second[0][top_k_value_indices]))
                # print(top_k_value_pro.shape)
                # print(top_k_value_indices.shape)
                # print(outputs['objectness_prob'][idx_batch, idx_box])
                # print('----------------------')
                # print(outputs['sem_cls_prob'][idx_batch, idx_box].shape)
                # print(probs[0].shape)
                outputs['sem_cls_prob'][idx_batch, idx_box] = probs[0]
                # outputs['sem_cls_prob'][idx_batch, idx_box] = torch.sqrt(probs[0] * outputs['objectness_prob'][idx_batch, idx_box])
                # outputs['sem_cls_prob'][idx_batch, idx_box] = torch.sqrt(probs[0] * outputs['objectness_prob'][idx_batch, idx_box])

        return outputs

    def clip_to_class_training(self, inputs, outputs, thres_obj=0.05, if_use_gt_box=False, if_expand_box=False, if_padding_input=True, test=False, if_cmp_class=False): # the max number of classes are 18(train) and 14(val)
        if if_use_gt_box:
        # ceiling experiment

            box_centers_batch = inputs['gt_box_centers']
            box_angle_batch = inputs['gt_box_angles']
            box_size_batch = inputs['gt_box_sizes']
            batch_size = box_size_batch.shape[0]
            # print(batch_size)
            outputs['center_unnormalized'] = inputs['gt_box_centers']
            outputs['angle_continuous'] = inputs['gt_box_angles']
            outputs['size_unnormalized'] = inputs['gt_box_sizes']
            outputs["box_corners"] = inputs['gt_box_corners']
            # outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, 690)).to('cuda')
            if if_cmp_class:
                outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, len(self.all_cmp_classes_keys)), device=self.device)
            else:
                outputs['sem_cls_prob'] = torch.zeros((batch_size, 64, self.test_range_max), device=self.device)
            outputs['objectness_prob'] = inputs["gt_box_present"].clone().detach() #torch.ones((8, 64)).to('cuda')
            x_offset_batch = inputs['x_offset']
            y_offset_batch = inputs['y_offset']
            ori_width_batch = inputs['ori_width']
            ori_height_batch = inputs['ori_height']
        else:
            box_centers_batch = outputs[
                'center_unnormalized']  # .cpu().numpy()#inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
            # print(box_centers.shape)
            box_angle_batch = outputs[
                'angle_continuous']  # .cpu().numpy() #inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
            # print(box_angle.shape)
            box_size_batch = outputs[
                'size_unnormalized']  # .cpu().numpy() #inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
            x_offset_batch = inputs['x_offset']
            y_offset_batch = inputs['y_offset']
            ori_width_batch = inputs['ori_width']
            ori_height_batch = inputs['ori_height']
            if self.args.if_only_novel_prompt:
                outputs['sem_cls_prob'] = torch.zeros((box_centers_batch.shape[0], box_centers_batch.shape[1], 27), device=self.device) #.to('cuda')
                outputs['sem_cls_logits'] = torch.zeros((box_centers_batch.shape[0], box_centers_batch.shape[1], 27), device=self.device) #.to('cuda')
            else:
                if if_cmp_class:
                    outputs['sem_cls_prob'] = torch.zeros((box_centers_batch.shape[0], box_centers_batch.shape[1], len(self.all_cmp_classes_keys)), device=self.device) #.to('cuda')
                    outputs['sem_cls_logits'] = torch.zeros((box_centers_batch.shape[0], box_centers_batch.shape[1], len(self.all_cmp_classes_keys)), device=self.device) #.to('cuda')
                else:
                    outputs['sem_cls_prob'] = torch.zeros((box_centers_batch.shape[0], box_centers_batch.shape[1], self.test_range_max), device=self.device) #.to('cuda')
                    outputs['sem_cls_logits'] = torch.zeros((box_centers_batch.shape[0], box_centers_batch.shape[1], self.test_range_max), device=self.device) #.to('cuda')
            # outputs['gt_region_embedding'] = outputs['region_embedding'].clone().detach()
            # region_embedding_batch = outputs['region_embedding']
            # return outputs
            # if test:
        # return outputs
        # print(box_angle_batch.shape)
        # print('=============inputs key=======================')
        # print(inputs.keys())
        img_batch_name = inputs['im_name']
        calib_name_list = inputs['calib_name']
        trans_mtx_batch = inputs['trans_mtx']
        # print(trans_mtx_batch.shape)
        input_image_batch = inputs["input_image"]
        batch_size_data = box_size_batch.shape[0]
        box_num_data = box_size_batch.shape[1]
        # outputs['sem_cls_prob'] = torch.zeros((batch_size_data, batch_size_data, 232), device=self.device)
        # if test:
        #     outputs['sem_cls_prob'] = torch.zeros((batch_size_data, batch_size_data, 238))
        #     outputs['sem_cls_logits'] = torch.zeros((batch_size_data, batch_size_data, 239))
        #     outputs['objectness_prob'] = torch.zeros((batch_size_data, batch_size_data, 1))
        #     text_features = torch.cat((self.text_features_fg_test, self.bg_embed + 1e-16), dim=0)
        #     # normalized features
        #     text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
        # else:
        # text_features = torch.cat((self.text_features_fg, self.text_features_bg), dim=0)
        if if_cmp_class:
            text_features = self.cmp_text_features_fg
        else:
            text_features = self.text_features_fg
        # normalized features
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)

        for idx_batch in range(batch_size_data):
            # box_center_list = box_centers_batch[idx_batch]
            # print(box_angle.shape)
            # print(idx_batch)
            x_offset = x_offset_batch[idx_batch]
            y_offset = y_offset_batch[idx_batch]

            calib_name = calib_name_list[idx_batch]
            img = input_image_batch[idx_batch]
            # img = input_image.permute(2, 0, 1)

            # input_im = Image.fromarray(input_image[:,:,::-1])
            # input_im_ = np.array(input_im)
            # img_ = Image.open(img_batch_name[idx_batch])
            # img_ = np.array(img)
            # if_seem = np.all(np.equal(img_, input_im_))
            # img_ = cv2.imread(img_batch_name[idx_batch])
            if self.dataset_name == 'scannet':
                squence_name = inputs['squence_name'][idx_batch]
                calib = self.dataset_util.SCANNET_Calibration(calib_name, squence_name)
            if self.dataset_name == 'sunrgbd':
                calib = self.dataset_util.SUNRGBD_Calibration(calib_name)
            box_center_list = box_centers_batch[idx_batch]
            box_angle_list = box_angle_batch[idx_batch]
            box_size_list = box_size_batch[idx_batch]
            trans_mtx = trans_mtx_batch[idx_batch]
            ori_width = ori_width_batch[idx_batch]
            ori_height = ori_height_batch[idx_batch]

            proposal_feature_list = []
            idx_box_list = []
            for idx_box in range(box_num_data):
                box_center = box_center_list[idx_box]
                box_angle = box_angle_list[idx_box]
                box_size = box_size_list[idx_box]
                # region_embed = region_embedding_batch[idx_batch, idx_box]
                # trans_mtx = trans_mtx_list[idx_box]

                if torch.max(box_size)<1e-16: # when displaying gt, some of the box_size will be zeros
                    # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                    # idx_box_list.append(0)
                    continue

                # print(idx_box)
                if if_padding_input:
                    corners_3d_image, corners_3d, d = self.dataset_util.compute_box_3d_offset_tensor(box_center, box_angle, box_size,
                                                                                       calib, x_offset, y_offset)
                    xmin = int(min(max(torch.min(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    ymin = int(min(max(torch.min(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                    xmax = int(min(max(torch.max(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    ymax = int(min(max(torch.max(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                    # corners_3d_image_np, corners_3d_np = sunrgbd_utils.compute_box_3d_offset(box_center.detach().cpu().numpy(), box_angle.detach().cpu().numpy(), box_size.detach().cpu().numpy(),
                    #                                                                    calib, x_offset.detach().cpu().numpy(), y_offset.detach().cpu().numpy())
                    # print(int(np.max(corners_3d_image.detach().cpu().numpy()))==int(np.max(corners_3d_image_np)))
                    # print(int(np.min(corners_3d_image.detach().cpu().numpy())) == int(np.min(corners_3d_image_np)))
                    # print(np.sum(np.abs(corners_3d.detach().cpu().numpy() - corners_3d_np)))
                    # xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    # ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                    # xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width+y_offset))
                    # ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height+x_offset))
                else:
                    corners_3d_image, corners_3d = dataset_util.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                       calib, trans_mtx)
                    xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                    xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                    ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))

                if torch.min(d) < 0:
                    # print('d less than 0')
                    continue
                    
                if if_expand_box:
                    w = ymax-ymin
                    h = xmax-xmin
                    if w > h:
                        xmin = xmin - (w-h)//2
                        xmax = xmax + (w-h)//2
                    else:
                        ymin = ymin - (h-w)//2
                        ymax = ymax + (h-w)//2
                    xmin = int(min(max(xmin, 0), img.shape[1]))
                    ymin = int(min(max(ymin, 0), img.shape[0]))
                    xmax = int(min(max(xmax, 0), img.shape[1]))
                    ymax = int(min(max(ymax, 0), img.shape[0]))

                if (xmax-xmin)>0 and (ymax-ymin)>0:
                    img_crop = img[ymin:ymax, xmin:xmax]
                    # img_crop_show = img_crop.cpu().numpy()
                    # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
                    # cv2.imwrite('debug_show/' + os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'_2.png', img_crop_show)
                    # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
                    # continue
                else:
                    # print('Box size is less than 0')
                    # idx_box_list.append(0)
                    # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                    continue

                idx_box_list.append(idx_box)

                # #####exp1: blank background#####
                # part 1
                # img_crop_ = img_crop.clone()
                w = ymax - ymin
                h = xmax - xmin
                if w > h:
                    max_edge = w
                else:
                    max_edge = h
                #
                img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8, device=self.device) * 255
                # part 2

                y_begin = (max_edge - w) // 2
                x_begin = (max_edge - h) // 2
                img_beckground[y_begin:y_begin+w, x_begin:x_begin+h, :] = img_crop
                # part 3
                img_crop = img_beckground.permute(2, 0, 1)
                img_crop_resized = self.resize(img_crop)

                img_crop_resized_expand = img_crop_resized.unsqueeze(0)
                proposal_feature_list.append(img_crop_resized_expand)

            # print(self.clip_model.visual.input_resolution)
            # print(idx_box_list)
            try:
                img_proposal = torch.cat(proposal_feature_list, 0)
            except:
                print('No object on GT.')
                continue

            # ################### add image embedding
            # max_img_h = max(img.shape[0], img.shape[1])
            # full_img_beckground = torch.ones(max_img_h, max_img_h, img.shape[2], dtype=torch.uint8, device=self.device) * 255
            # y_begin = (max_img_h - img.shape[0]) // 2
            # x_begin = (max_img_h - img.shape[1]) // 2
            # full_img_beckground[y_begin:y_begin + img.shape[0], x_begin:x_begin + img.shape[1], :] = img
            # full_img = full_img_beckground.permute(2, 0, 1)
            # full_img_crop = self.resize(full_img).unsqueeze(0)
            # full_image_proposal_batch = self.preprocess_for_tensor(full_img_crop).unsqueeze(0)[0]
            # full_image_proposal_features_batch = self.clip_model.encode_image(full_image_proposal_batch)
            # if isinstance(full_image_proposal_features_batch, tuple):
            #     full_image_proposal_features_batch = full_image_proposal_features_batch[0]
            # full_image_proposal_features_batch = full_image_proposal_features_batch.to(torch.float32)
            # # full_image_proposal_features_batch = full_image_proposal_features_batch / full_image_proposal_features_batch.norm(dim=1, keepdim=True)
            # image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]
            # proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)
            # if isinstance(proposal_features_batch, tuple):
            #     proposal_features_batch = proposal_features_batch[0]
            # proposal_features_batch = proposal_features_batch.to(torch.float32)
            # proposal_features_batch_norm = proposal_features_batch / proposal_features_batch.norm(dim=1, keepdim=True)
            # logits_per_image = self.logit_scale * proposal_features_batch_norm @ (text_features_norm * full_image_proposal_features_batch).t().to(
            #     torch.float32)
            # # logits_per_image = self.logit_scale * proposal_features_batch_norm @ text_features_norm.t().to(torch.float32)
            # logits_per_image_ = self.logit_scale * full_image_proposal_features_batch @ ((text_features_norm).t().to(torch.float32) )
            # if not self.if_adopt_region_embed:
            #     probs_ = logits_per_image_.softmax(dim=-1)
            #     probs = logits_per_image.softmax(dim=-1)
            #     # print(1)
            #     outputs['sem_cls_prob'][idx_batch][idx_box_list] = probs #* probs_[:, :].detach() #torch.gather(probs[:, 
            # ################### add image embedding


            image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]
            proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)
            if isinstance(proposal_features_batch, tuple):
                proposal_features_batch = proposal_features_batch[0]
            proposal_features_batch = proposal_features_batch.to(torch.float32)
            proposal_features_batch_norm = proposal_features_batch / proposal_features_batch.norm(dim=1, keepdim=True)
            logits_per_image = self.logit_scale * proposal_features_batch_norm @ text_features_norm.t().to(torch.float32)
            # logits_per_image = self.logit_scale * proposal_features_batch_norm @ ((text_features_norm * full_image_proposal_features_batch).t().to(torch.float32) )
            # idx_tensor = torch.tensor(idx_box_list).to(self.device)
            # outputs['gt_region_embedding'][idx_batch][idx_box_list] =  proposal_features_batch #torch.gather(proposal_features_batch, dim=0, index=idx_tensor)
            if not self.if_adopt_region_embed:
                probs = logits_per_image.softmax(dim=-1)
                outputs['sem_cls_prob'][idx_batch][idx_box_list] = probs[:, :].detach() #torch.gather(probs[:, :-1].detach(), dim=-1, index=idx_tensor)
                # max_v, max_idx =  torch.max(probs, dim=-1)
                # print(max_idx)
                # print(max_v)
                # print(outputs['objectness_prob'][idx_batch][idx_box_list].shape)
                # outputs['objectness_prob'][idx_batch][idx_box_list] = torch.where(max_v>0.3, outputs['objectness_prob'][idx_batch][idx_box_list], torch.zeros_like(outputs['objectness_prob'][idx_batch][idx_box_list], device='cuda'))
                # outputs['sem_cls_logits'][idx_batch][idx_box_list] = logits_per_image #torch.gather(logits_per_image, dim=-1, index=idx_tensor)
                # outputs['objectness_prob'][idx_batch][idx_box_list] = 1.0-probs[:, -1] #torch.gather(1.0-probs[:, -1], dim=-1, index=idx_tensor)
                # print(1)


            # return outputs

        if self.if_adopt_region_embed:
            # region_embed_norm = region_embedding_batch / region_embedding_batch.norm(dim=-1, keepdim=True)
            logits_per_image = self.logit_scale * region_embed_norm @ text_features_norm.t()
            probs = logits_per_image.softmax(dim=-1)  # .cpu().numpy()
            outputs['sem_cls_prob'] = probs[:, :, :-1].detach()
            outputs['sem_cls_logits'] = logits_per_image
            outputs['objectness_prob'] = 1.0 - probs[:, :, -1]

        return outputs

    def get_predicted_box_clip_embedding_nms_iou_save_keep_clip_driven_with_cate_confidence(self, inputs, outputs, thres_obj=0.05,
                                                           if_use_gt_box=False,
                                                           if_expand_box=False,
                                                           if_padding_input=True,
                                                           test=False,
                                                           curr_epoch=-1,
                                                           if_test=False):  # the max number of classes are 18(train) and 14(val)
        gt_center_ori = outputs[
            "box_corners_xyz"]  # .clone().detach() #inputs["gt_box_corners_xyz"]  # shape: (8, 64, 8, 3) box_predictions["outputs"]['center_unnormalized']
        objectness_prob = outputs["objectness_prob"].detach().clone()
        # img_batch_name = inputs['im_name']
        # gt_angles = inputs["gt_box_angles"]
        # gt_sizes = inputs["gt_box_sizes"]
        # tmp = outputs['center_unnormalized']  # shape: (8, 128, 3)
        K_tensor = inputs['K']
        Rtilt_tensor = inputs['Rtilt']
        flip_array = inputs['flip_array'].unsqueeze(-1)
        scale_array = inputs['scale_array'].unsqueeze(1)
        rot_array = inputs['rot_array'].unsqueeze(1)
        rot_angle = inputs['rot_angle'].unsqueeze(1)
        # back the augmentation for point cloud
        gt_center = gt_center_ori * scale_array  # [:, :2, :]
        gt_center = torch.matmul(gt_center, rot_array)
        if 'zx_flip_array' in inputs.keys():
            zx_flip_array = inputs['zx_flip_array'].unsqueeze(-1)
            gt_center[:, :, :, 1] = gt_center[:, :, :, 1] * zx_flip_array
        gt_center[:, :, :, 0] = gt_center[:, :, :, 0] * flip_array  # [:, :2, :]

        # inputs["gt_box_centers"] = inputs["gt_box_centers"].detach().clone()  * inputs['scale_array']
        # inputs["gt_box_sizes"] = inputs["gt_box_sizes"].detach().clone()  * inputs['scale_array']
        #
        # inputs["gt_box_centers"] = torch.matmul(inputs["gt_box_centers"], inputs['rot_array'])
        # inputs["gt_box_angles"] = inputs["gt_box_angles"].detach().clone() + inputs['rot_angle'].unsqueeze(-1)
        #
        # inputs["gt_box_centers"][:, :,  0] = inputs["gt_box_centers"][:, :,  0] * inputs['flip_array']
        # inputs["gt_box_angles"] = torch.where(inputs['flip_array'] < 0, (math.pi-inputs["gt_box_angles"]), inputs["gt_box_angles"])
        #
        # outputs["center_unnormalized"] = outputs["center_unnormalized"].detach().clone()  * inputs['scale_array']
        # outputs["size_unnormalized"] = outputs["size_unnormalized"].detach().clone()  * inputs['scale_array']
        #
        # outputs["center_unnormalized"] = torch.matmul(outputs["center_unnormalized"], inputs['rot_array'])
        # outputs["angle_continuous"] = outputs["angle_continuous"].detach().clone() + inputs['rot_angle'].unsqueeze(-1)
        #
        # outputs["center_unnormalized"][:, :,  0] = outputs["center_unnormalized"][:, :,  0] * inputs['flip_array']
        # outputs["angle_continuous"] = torch.where(inputs['flip_array'] < 0, (math.pi-outputs["angle_continuous"]), outputs["angle_continuous"])
        if (not if_test) and (
                curr_epoch % self.online_nms_update_save_epoch == 0):  # and curr_epoch >= 540:
            ori_center_unnormalized = outputs["center_unnormalized"].detach().clone() * inputs['scale_array']
            ori_size_unnormalized = outputs["size_unnormalized"].detach().clone() * inputs['scale_array']

            ori_center_unnormalized = torch.matmul(ori_center_unnormalized, inputs['rot_array'])
            ori_angle_continuous = outputs["angle_continuous"].detach().clone() + inputs['rot_angle'].unsqueeze(-1)

            if 'zx_flip_array' in inputs.keys():
                ori_center_unnormalized[:, :, 1] = ori_center_unnormalized[:, :, 1] * inputs['zx_flip_array']
                ori_angle_continuous = torch.where(inputs['zx_flip_array'] < 0, (math.pi - ori_angle_continuous),
                                                   ori_angle_continuous)

            ori_center_unnormalized[:, :, 0] = ori_center_unnormalized[:, :, 0] * inputs['flip_array']
            ori_angle_continuous = torch.where(inputs['flip_array'] < 0, (math.pi - ori_angle_continuous),
                                               ori_angle_continuous)

        # ori_center_unnormalized = outputs["center_unnormalized"].detach().clone()
        # ori_size_unnormalized = outputs["size_unnormalized"].detach().clone()
        # ori_angle_continuous = outputs["angle_continuous"].detach().clone()

        # back to original image
        # center_2dpoints = torch.zeros((8, 64, 8, 2), device=self.device)
        # for idx_batch in range(center_2dpoints.shape[1]):
        #     center_2dpoints[:, idx_batch, :, :] = project_3dpoint_to_2dpoint_tensor(center_pos.to(torch.double)[:, idx_batch, :, :], K_tensor=K_tensor, Rtilt_tensor=Rtilt_tensor)
        # center_2dpoints, depth = project_3dpoint_to_2dpoint_corners_tensor(gt_center.to(torch.double), K_tensor=K_tensor,
        #                                                             Rtilt_tensor=Rtilt_tensor)
        center_2dpoints, depth = self.dataset_util.project_3dpoint_to_2dpoint_corners_tensor(gt_center.to(torch.double),
                                                                                     K_tensor=K_tensor,
                                                                                     Rtilt_tensor=Rtilt_tensor)
        # center_2dpoints, _ = project_3dpoint_to_2dpoint_corners_tensor(gt_center, gt_angles, gt_sizes, K_tensor=K_tensor,
        #                                                     Rtilt_tensor=Rtilt_tensor)

        # center_pos = inputs["gt_box_corners"] # shape: (8, 64, 8, 3) box_predictions["outputs"]['center_unnormalized']
        # center_pos = outputs['center_unnormalized']  # shape: (8, 128, 3)
        # K_tensor = inputs['K']
        # Rtilt_tensor = inputs['Rtilt']
        # flip_array = inputs['flip_array']#.unsqueeze(-1)
        # scale_array = inputs['scale_array']   #.unsqueeze(1)
        # rot_array = inputs['rot_array']       #.unsqueeze(1)
        # # back the augmentation for point cloud
        # center_pos = center_pos * scale_array  # [:, :2, :]
        # center_pos = torch.bmm(center_pos, rot_array)
        # center_pos[:, :,  0] = center_pos[:, :,  0] * flip_array  # [:, :2, :]
        # center_2dpoints = project_3dpoint_to_2dpoint_tensor(center_pos.to(torch.double), K_tensor=K_tensor,
        #                                                     Rtilt_tensor=Rtilt_tensor)

        # clip the point in the original size of the image
        text_features_clip = outputs["maybe_novel_text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = outputs["logit_scale"]
        for idx_batch in range(center_2dpoints.shape[0]):
            center_2dpoints[idx_batch, :, :, 0] = torch.clip(center_2dpoints[idx_batch, :, :, 0], min=0,
                                                             max=inputs["ori_width"][idx_batch] - 1)
            center_2dpoints[idx_batch, :, :, 1] = torch.clip(center_2dpoints[idx_batch, :, :, 1], min=0,
                                                             max=inputs["ori_height"][idx_batch] - 1)
            center_2dpoints[idx_batch, :, :, 0] += inputs['y_offset'][idx_batch].unsqueeze(-1)
            center_2dpoints[idx_batch, :, :, 1] += inputs['x_offset'][idx_batch].unsqueeze(
                -1)  # .unsqueeze(-1)  # short height
        # center_2dpoints[:, :, :, 0] = torch.maximum(
        #     torch.minimum(center_2dpoints[:, :, :, 0], (inputs["ori_width"].unsqueeze(-1) - 1)), self.zeros_tensor)
        # center_2dpoints[:, :, :, 1] = torch.maximum(
        #     torch.minimum(center_2dpoints[:, :, :, 1], (inputs["ori_height"].unsqueeze(-1) - 1)), self.zeros_tensor)

        # center_2dpoints[:, :, 0] = torch.clip(center_2dpoints[:, :, 0], min=)
        # do the padding in the datalayer

        # do the flip in the datalayer
        image_flip_array = inputs["image_flip_array"].unsqueeze(-1)
        flip_length = inputs["flip_length"].unsqueeze(-1).unsqueeze(-1)
        center_2dpoints[:, :, :, 0] = center_2dpoints[:, :, :, 0] * image_flip_array + (1 - image_flip_array) * (
                flip_length - 1 - center_2dpoints[:, :, :, 0])

        gt_box_corners = inputs["gt_box_corners"].detach().clone()
        gt_box_present = inputs["gt_box_present"].detach().clone()
        pred_box_corneres = outputs["box_corners"].detach().clone()
        box_keep = torch.zeros((center_2dpoints.shape[0], center_2dpoints.shape[1]), device=self.device)
        box_save = torch.zeros((center_2dpoints.shape[0], center_2dpoints.shape[1]), device=self.device)

        outputs['gt_text_correlation_embedding'] = torch.zeros(
            (center_2dpoints.shape[0], center_2dpoints.shape[1], 512),  # change version
            device=self.device)
        # outputs['gt_text_correlation_embedding'] = torch.zeros(
        #     (center_2dpoints.shape[0], center_2dpoints.shape[1], 768),
        #     device=self.device)
        outputs['gt_text_correlation_embedding_mask'] = torch.zeros(
            (center_2dpoints.shape[0], center_2dpoints.shape[1], 1),
            device=self.device)
        if (not if_test) and (
                curr_epoch % self.online_nms_update_save_epoch == 0):  # and curr_epoch >= 540:
            # center_unnormalized_outputs = outputs["center_unnormalized"].detach().cpu().numpy()
            # size_unnormalized_outputs = outputs["size_unnormalized"].detach().cpu().numpy()
            # angle_unnormalized_outputs = outputs["angle_continuous"].detach().cpu().numpy()
            save_box_info = torch.zeros(
                (outputs["center_unnormalized"].shape[0], outputs["center_unnormalized"].shape[1], 10), #(center(3), size(3), angle, class, cate_prob, objectness)
                device=self.device)
            save_box_info_tmp = torch.cat(
                [ori_center_unnormalized, ori_size_unnormalized, ori_angle_continuous.unsqueeze(-1)], dim=-1)
            save_box_info[:, :, :7] = save_box_info_tmp
            save_box_info[:, :, 7] = -1
            save_box_info[:, :, 8] = 0
            save_box_info[:, :, 9] = 0
            save_box_info = save_box_info.detach().cpu().numpy()
        for idx_batch in range(center_2dpoints.shape[0]):
            if (not if_test):  # and curr_epoch >= 540:
                box2d_thisbatch = torch.zeros((center_2dpoints.shape[1], 4), device=self.device)
                scores = objectness_prob[idx_batch]
                del_box_list = []
                for idx_box in range(center_2dpoints.shape[1]):
                    box_size = outputs["size_unnormalized"][idx_batch][idx_box]
                    if torch.max(box_size) < 1e-16:
                        # give the box up
                        scores[idx_box] = -1
                        del_box_list.append(idx_box)
                        box2d_thisbatch[idx_box][0] = 0
                        box2d_thisbatch[idx_box][1] = 0
                        box2d_thisbatch[idx_box][2] = 2
                        box2d_thisbatch[idx_box][3] = 2
                        continue
                    corners_3d_image = center_2dpoints[idx_batch][idx_box]
                    xmin = int(torch.min(corners_3d_image[:, 0]))
                    ymin = int(torch.min(corners_3d_image[:, 1]))
                    xmax = int(torch.max(corners_3d_image[:, 0]))
                    ymax = int(torch.max(corners_3d_image[:, 1]))
                    if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                        # give the box up
                        scores[idx_box] = -1
                        del_box_list.append(idx_box)
                        box2d_thisbatch[idx_box][0] = 0
                        box2d_thisbatch[idx_box][1] = 0
                        box2d_thisbatch[idx_box][2] = 2
                        box2d_thisbatch[idx_box][3] = 2
                        continue
                    if torch.min(depth[idx_batch][idx_box]) < 0:
                        scores[idx_box] = -1
                        del_box_list.append(idx_box)
                        box2d_thisbatch[idx_box][0] = 0
                        box2d_thisbatch[idx_box][1] = 0
                        box2d_thisbatch[idx_box][2] = 2
                        box2d_thisbatch[idx_box][3] = 2
                        continue
                    box2d_thisbatch[idx_box][0] = ymin
                    box2d_thisbatch[idx_box][1] = xmin
                    box2d_thisbatch[idx_box][2] = ymax
                    box2d_thisbatch[idx_box][3] = xmax

                nms_after = torchvision.ops.nms(box2d_thisbatch, scores,
                                                iou_threshold=0.25)  # adopt the nms to decrease the number of boxes
                pred_box_corneres_nms_select = torch.index_select(pred_box_corneres[idx_batch], dim=0, index=nms_after)
                gt_box_corners_nms_select = torch.index_select(gt_box_corners[idx_batch], dim=0,
                                                               index=torch.nonzero(gt_box_present[idx_batch])[:, 0])

                if gt_box_corners_nms_select.shape[0] > 0:
                    pred_box_corneres_nms_select_points = torch.zeros((pred_box_corneres_nms_select.shape[0], 6),
                                                                      device=self.device)
                    pred_box_corneres_nms_select_points[:, 0] = torch.min(pred_box_corneres_nms_select[:, :, 0], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 1] = torch.min(pred_box_corneres_nms_select[:, :, 1], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 2] = torch.min(pred_box_corneres_nms_select[:, :, 2], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 3] = torch.max(pred_box_corneres_nms_select[:, :, 0], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 4] = torch.max(pred_box_corneres_nms_select[:, :, 1], dim=1)[
                        0]
                    pred_box_corneres_nms_select_points[:, 5] = torch.max(pred_box_corneres_nms_select[:, :, 2], dim=1)[
                        0]

                    gt_box_corners_nms_select_points = torch.zeros((gt_box_corners_nms_select.shape[0], 6),
                                                                   device=self.device)
                    gt_box_corners_nms_select_points[:, 0] = torch.min(gt_box_corners_nms_select[:, :, 0], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 1] = torch.min(gt_box_corners_nms_select[:, :, 1], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 2] = torch.min(gt_box_corners_nms_select[:, :, 2], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 3] = torch.max(gt_box_corners_nms_select[:, :, 0], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 4] = torch.max(gt_box_corners_nms_select[:, :, 1], dim=1)[0]
                    gt_box_corners_nms_select_points[:, 5] = torch.max(gt_box_corners_nms_select[:, :, 2], dim=1)[0]

                    select_boxes_idx = []
                    assert nms_after.shape[0] == pred_box_corneres_nms_select.shape[0]
                    for idx_box_pred in range(pred_box_corneres_nms_select_points.shape[0]):
                        match_gt = 0
                        pred_box_select = pred_box_corneres_nms_select_points[idx_box_pred]
                        for idx_box_gt in range(gt_box_corners_nms_select_points.shape[0]):

                            gt_box_select = gt_box_corners_nms_select_points[idx_box_gt]
                            iou_pred_gt = self.cal_iou(pred_box_select, gt_box_select)
                            if iou_pred_gt > 0.25:
                                match_gt = 1
                                break
                        if match_gt == 0:
                            select_boxes_idx.append(idx_box_pred)

                    nms_after_iou_select = nms_after[select_boxes_idx]
                else:
                    nms_after_iou_select = nms_after
                    # print(1)
                    # box_pred_item = pred_box_corneres_nms_select[idx_box_pred]
                    # intersection_vol, iou_3d = box3d_overlap(pred_box_corneres_nms_select, gt_box_corners_nms_select, eps_coplanar=1e-4, eps_nonzero=1e-32) # give up the box of the seen objects by calculate IoU with GT
                    # print(gt_box_corners_nms_select)
                    # iou_3d_predict, _ = torch.max(iou_3d, dim=-1)
                    # nms_after_iou_select = nms_after[iou_3d_predict < 0.25] # select the iou < 0.25
                # else:
                #     nms_after_iou_select = nms_after
                # box_keep[idx_batch] = box_keep[idx_batch].index_fill(dim=0, index=nms_after_iou_select, value=1)
                # box_keep[idx_batch][
                #     scores < self.keep_objectness] = 0  # set the objectness to keep, only keep the box which have larger objectness
                # for idx_box in del_box_list:
                #     box_keep[idx_batch][idx_box] = 0
                # for idx_nms in range(nms_after.shape[0]):
                #     idx_box = nms_after[idx_nms]
                #     for gt_idx_box in range(gt_box_corners.shape[1]):
                #
                #         print(1)
                # novel_idx_list = nms_after_iou_select.cpu().numpy()
                # novel_idx_list_final = []
                # for novel_idx in novel_idx_list:
                #     if box_keep[idx_batch][novel_idx] == 0:
                #         continue
                #     novel_idx_list_final.append(novel_idx)

                box_save[idx_batch] = box_save[idx_batch].index_fill(dim=0, index=nms_after_iou_select, value=1)
                box_save[idx_batch][
                    scores < self.save_objectness] = 0  # set the objectness to keep, only keep the box which have larger objectness
                for idx_box in del_box_list:
                    box_save[idx_batch][idx_box] = 0
                # for idx_nms in range(nms_after.shape[0]):
                #     idx_box = nms_after[idx_nms]
                #     for gt_idx_box in range(gt_box_corners.shape[1]):
                #
                #         print(1)
                novel_idx_list = nms_after_iou_select.cpu().numpy()
                novel_idx_save_final = []
                for novel_idx in novel_idx_list:
                    if box_save[idx_batch][novel_idx] == 0:
                        continue
                    novel_idx_save_final.append(novel_idx)

                idx_maybe_novel_box_list = []
                proposal_maybe_novel_feature_list = []
                for idx_box in novel_idx_save_final:
                    img = inputs['input_image'][idx_batch]
                    box_size = outputs["size_unnormalized"][idx_batch][idx_box]
                    if torch.max(box_size) < 1e-16:  # when displaying gt, some of the box_size will be zeros
                        # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
                        # idx_box_list.append(0)
                        continue
                    corners_3d_image = center_2dpoints[idx_batch][idx_box]
                    xmin = int(torch.min(corners_3d_image[:, 0]))
                    ymin = int(torch.min(corners_3d_image[:, 1]))
                    xmax = int(torch.max(corners_3d_image[:, 0]))
                    ymax = int(torch.max(corners_3d_image[:, 1]))

                    if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                        img_crop = img[ymin:ymax, xmin:xmax]
                    #     img_crop_show = img_crop.cpu().numpy()
                    # # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
                    #     cv2.imwrite('debug_show_2/' + os.path.basename(img_batch_name[idx_batch])[:-4] + '_' + str(
                    #         idx_box) + '_2_predicted_embedding.png', img_crop_show)
                    # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
                    else:
                        continue

                    if torch.min(depth[idx_batch][idx_box])<0:
                        continue

                    idx_maybe_novel_box_list.append(idx_box)

                    w = ymax - ymin
                    h = xmax - xmin

                    if w > h:
                        max_edge = w
                    else:
                        max_edge = h
                    #
                    img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8,
                                                device=self.device) * 255
                    # part 2

                    y_begin = (max_edge - w) // 2
                    x_begin = (max_edge - h) // 2
                    # try:
                    # print(img_crop.shape)
                    # print(x_begin)
                    # print(y_begin)
                    # print(x_begin + h)
                    # print(y_begin + w)
                    # print('============')
                    img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
                    # except:
                    # print(1)
                    # part 3
                    img_crop = img_beckground.permute(2, 0, 1)
                    img_crop_resized = self.resize(img_crop)  # change version
                    # print(img_crop_resized.shape)
                    # img_crop_resized = self.test_resize(img_crop)
                    # cv2.imwrite(os.path.join('./%s_%s_3d_to_2d_box_gt.png' % (idx_batch, idx_box)), img_crop_resized.permute(1, 2, 0).cpu().numpy())
                    img_crop_resized_expand = img_crop_resized.unsqueeze(0)
                    proposal_maybe_novel_feature_list.append(img_crop_resized_expand)

                # if len(proposal_feature_list) < 1:
                #     continue
                if len(proposal_maybe_novel_feature_list) > 0:
                    img_proposal = torch.cat(proposal_maybe_novel_feature_list, 0)
                    # except:
                    #     print('No object on GT, too')
                    #     continue
                    # im_proposal_ref = np.load('img_proposal.npy')
                    # image_proposal_batch_ref = np.load('image_proposal_batch.npy')
                    # proposal_features_batch_ref = np.load('proposal_features_batch.npy')

                    # img_proposal = torch.cat(proposal_feature_list, 0)
                    image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]  # change version
                    # image_proposal_batch_copy = image_proposal_batch.detach().clone()
                    # proposal_features_batch = self.res_encoder(image_proposal_batch, if_pool=True, if_early_feat=False)
                    proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)  # change version
                    # proposal_features_batch_1 = self.clip_model_1.encode_image(image_proposal_batch_copy)  # change version
                    # assert torch.equal(proposal_features_batch, proposal_features_batch_0)
                    if isinstance(proposal_features_batch, tuple):
                        proposal_features_batch = proposal_features_batch[0]
                    proposal_features_batch = proposal_features_batch.to(torch.float32)

                    # tmp = outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list]
                    # outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list] = proposal_features_batch
                    # outputs['gt_text_correlation_embedding_mask'][idx_batch][idx_box_list] = 1
                    text_correlation_embedding = proposal_features_batch / (
                            proposal_features_batch.norm(dim=-1, keepdim=True) + 1e-32)
                    maybe_novel_correlation_map = torch.mm(text_correlation_embedding,
                                                           text_features_clip.permute(1, 0)) * temperature_param
                    maybe_novel_scores = torch.nn.functional.softmax(maybe_novel_correlation_map, dim=-1)[:, :]
                    maybe_max_score, maybe_max_idx = torch.max(maybe_novel_scores, dim=-1)
                    novel_idx_list_final_select = []
                    maybe_condition = torch.logical_and(maybe_max_score > self.clip_driven_keep_thres,
                                                        maybe_max_idx >= self.train_range_max)
                    if (curr_epoch % self.online_nms_update_save_epoch == 0):
                        for maybe_idx in range(maybe_condition.shape[0]):
                            maybe_item = maybe_condition[maybe_idx]
                            if maybe_item:
                                novel_idx_list_final_select.append(idx_maybe_novel_box_list[maybe_idx])
                                save_box_info[idx_batch][idx_maybe_novel_box_list[maybe_idx]][7] = maybe_max_idx[maybe_idx]
                                save_box_info[idx_batch][idx_maybe_novel_box_list[maybe_idx]][8] = maybe_max_score[maybe_idx]
                                save_box_info[idx_batch][idx_maybe_novel_box_list[maybe_idx]][9] = scores[idx_maybe_novel_box_list[maybe_idx]]
                    # if self.if_keep_box and (not if_test):  # and curr_epoch >= 540:
                    #     # begin_idx = int(torch.sum(inputs["gt_box_present"][idx_batch]).item())
                    #     begin_idx = int((inputs["gt_ori_box_num"][idx_batch]).item())
                    #     # novel_box_discovery_list = []
                    #     for novel_idx in novel_idx_list_final:
                    #         if begin_idx > 63:
                    #             print('Final gt boxs are more than 64')
                    #             break
                    #
                    #         inputs["gt_box_present"][idx_batch][begin_idx] = 1
                    #         this_novel_scores = outputs["angle_logits"][idx_batch][novel_idx].softmax(dim=-1)
                    #         max_this_novel_scores, max_idx_this_novel_scores = torch.max(this_novel_scores, dim=-1)
                    #         inputs["gt_angle_class_label"][idx_batch][begin_idx] = max_idx_this_novel_scores
                    #         inputs["gt_angle_residual_label"][idx_batch][begin_idx] = \
                    #             outputs["angle_residual"][idx_batch][novel_idx][max_idx_this_novel_scores]
                    #         inputs["gt_box_sizes_normalized"][idx_batch][begin_idx] = outputs["size_normalized"][idx_batch][
                    #             novel_idx]
                    #         inputs["gt_box_sizes"][idx_batch][begin_idx] = outputs["size_unnormalized"][idx_batch][
                    #             novel_idx]
                    #         inputs["gt_box_corners"][idx_batch][begin_idx] = outputs["box_corners"][idx_batch][novel_idx]
                    #         inputs["gt_box_corners_xyz"][idx_batch][begin_idx] = outputs["box_corners_xyz"][idx_batch][
                    #             novel_idx]
                    #         inputs["gt_box_angles"][idx_batch][begin_idx] = outputs["angle_continuous"][idx_batch][
                    #             novel_idx]
                    #         inputs["gt_box_centers_normalized"][idx_batch][begin_idx] = \
                    #             outputs["center_normalized"][idx_batch][
                    #                 novel_idx]
                    #         inputs["gt_box_centers"][idx_batch][begin_idx] = outputs["center_unnormalized"][idx_batch][
                    #             novel_idx]
                    #         begin_idx += 1

                    if (not if_test) and (
                            curr_epoch % self.online_nms_update_save_epoch == 0):
                        begin_idx = int((inputs["gt_ori_box_num"][idx_batch]).item())
                        novel_box_save_list = []
                        for novel_idx in novel_idx_list_final_select:
                            if begin_idx > 63:
                                print('Final gt boxs are more than 64')
                                break
                            if (curr_epoch % self.online_nms_update_save_epoch == 0):
                                novel_box_save_list.append(save_box_info[idx_batch][novel_idx])

                        if (curr_epoch % self.online_nms_update_save_epoch == 0) and len(
                                novel_box_save_list) > 0:
                            novel_box_path = inputs['pseudo_box_path'][idx_batch]
                            if not self.if_accumulate_former_pseudo_labels:
                                # novel_box_former = np.load(novel_box_path)
                                novel_box_save = np.array(novel_box_save_list)
                                # novel_box_final = np.concatenate(novel_box_save, axis=0)
                                np.save(novel_box_path, novel_box_save)
                            else:
                                # print('accumulate boxes')
                                novel_box_former = np.load(novel_box_path)
                                novel_box_later = np.array(novel_box_save_list)
                                if novel_box_former.shape[0] == 0: # no former boxes
                                    novel_box_final = novel_box_later
                                else: # have former boxes
                                    novel_box_final = np.concatenate((novel_box_former, novel_box_later), axis=0)
                                np.save(novel_box_path, novel_box_final)
                # print(1)
                # print('Have new boxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx:')
                # print(inputs["im_name"][idx_batch])
                # inputs["gt_box_present"][idx_batch][begin_idx] = 1
                # this_novel_scores = outputs["angle_logits"][idx_batch][novel_idx].softmax(dim=-1)
                # max_this_novel_scores, max_idx_this_novel_scores = torch.max(this_novel_scores, dim=-1)
                # inputs["gt_angle_class_label"][idx_batch][begin_idx] = max_idx_this_novel_scores
                # inputs["gt_angle_residual_label"][idx_batch][begin_idx] = \
                #     outputs["angle_residual"][idx_batch][novel_idx][max_idx_this_novel_scores]
                # inputs["gt_box_sizes_normalized"][idx_batch][begin_idx] = outputs["size_normalized"][idx_batch][
                #     novel_idx]
                # inputs["gt_box_sizes"][idx_batch][begin_idx] = outputs["size_unnormalized"][idx_batch][novel_idx]
                # inputs["gt_box_corners"][idx_batch][begin_idx] = outputs["box_corners"][idx_batch][novel_idx]
                # inputs["gt_box_corners_xyz"][idx_batch][begin_idx] = outputs["box_corners_xyz"][idx_batch][
                #     novel_idx]
                # inputs["gt_box_angles"][idx_batch][begin_idx] = outputs["angle_continuous"][idx_batch][novel_idx]
                # inputs["gt_box_centers_normalized"][idx_batch][begin_idx] = outputs["center_normalized"][idx_batch][
                #     novel_idx]
                # inputs["gt_box_centers"][idx_batch][begin_idx] = outputs["center_unnormalized"][idx_batch][
                #     novel_idx]
                # a = 1

            # outputs['gt_text_correlation_embedding'] = torch.zeros((center_2dpoints.shape[0], center_2dpoints.shape[1], outputs["text_correlation_embedding"].shape[2]), device=self.device)

        #     proposal_feature_list = []
        #     idx_box_list = []
        #     # cv2.imwrite(os.path.join('./%s_3d_to_2d_box_image.png' % idx_batch), inputs['input_image'][idx_batch].cpu().numpy())
        #     # if_keep_box = torch.rand(size=(center_2dpoints.shape[1], 1), device=self.device, requires_grad=False)
        #     if (not self.if_select_box_by_objectness):  # or curr_epoch < 540:
        #         select_list = np.random.choice(self.box_idx_list, self.distillation_box_num, replace=False)
        #     else:
        #         # locat_ob_list = torch.nonzero(objectness_prob_this_batch > 0.5)
        #         # locat_ob_list = locat_ob_list.cpu().numpy()
        #         ob_list = np.array(novel_idx_list_final, dtype=np.int8)
        #         if len(ob_list) >= self.distillation_box_num:
        #             # select_list = ob_list
        #             print('Novel boxes are more than 32')
        #             select_list = np.random.choice(ob_list, self.distillation_box_num, replace=False)
        #         else:
        #             other_list = []
        #             for box_idx in self.box_idx_list:
        #                 if box_idx not in novel_idx_list_final:
        #                     other_list.append(box_idx)
        #             other_array = np.array(other_list, dtype=np.int8)
        #             left_len = self.distillation_box_num - ob_list.shape[0]
        #             select_list = np.concatenate(
        #                 (ob_list, np.random.choice(other_array, left_len, replace=False)))
        #
        #     # select_list = np.random.choice(self.box_idx_list, 32, replace=False)
        #     # assert len(select_list) >= 30
        #     # print(select_list)
        #     # print(len(select_list))
        #     for idx_box in select_list:
        #         img = inputs['input_image'][idx_batch]
        #         box_size = outputs["size_unnormalized"][idx_batch][idx_box]
        #         if torch.max(box_size) < 1e-16:  # when displaying gt, some of the box_size will be zeros
        #             # outputs['gt_region_embedding'][idx_batch, idx_box] = outputs['region_embedding'][idx_batch, idx_box]
        #             # idx_box_list.append(0)
        #             continue
        #         corners_3d_image = center_2dpoints[idx_batch][idx_box]
        #         xmin = int(torch.min(corners_3d_image[:, 0]))
        #         ymin = int(torch.min(corners_3d_image[:, 1]))
        #         xmax = int(torch.max(corners_3d_image[:, 0]))
        #         ymax = int(torch.max(corners_3d_image[:, 1]))
        #
        #         if (xmax - xmin) > 0 and (ymax - ymin) > 0:
        #             img_crop = img[ymin:ymax, xmin:xmax]
        #         #     img_crop_show = img_crop.cpu().numpy()
        #         # # img_crop_ = img_.crop((ymin_, xmin_, ymax_, xmax_)) # the w, h of Iamge.open is diff from cv2.imread
        #         #     cv2.imwrite('debug_show_2/' + os.path.basename(img_batch_name[idx_batch])[:-4] + '_' + str(
        #         #         idx_box) + '_2_predicted_embedding.png', img_crop_show)
        #         # img_crop_.save(os.path.basename(img_batch_name[idx_batch])[:-4]+'_'+str(idx_box)+'.png')
        #         else:
        #             continue
        #         if torch.min(depth[idx_batch][idx_box]) < 0:
        #             continue
        #
        #         idx_box_list.append(idx_box)
        #
        #         w = ymax - ymin
        #         h = xmax - xmin
        #
        #         if w > h:
        #             max_edge = w
        #         else:
        #             max_edge = h
        #         #
        #         img_beckground = torch.ones(max_edge, max_edge, img.shape[2], dtype=torch.uint8,
        #                                     device=self.device) * 255
        #         # part 2
        #
        #         y_begin = (max_edge - w) // 2
        #         x_begin = (max_edge - h) // 2
        #         # try:
        #         # print(img_crop.shape)
        #         # print(x_begin)
        #         # print(y_begin)
        #         # print(x_begin + h)
        #         # print(y_begin + w)
        #         # print('============')
        #         img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
        #         # except:
        #         # print(1)
        #         # part 3
        #         img_crop = img_beckground.permute(2, 0, 1)
        #         img_crop_resized = self.resize(img_crop)  # change version
        #         # print(img_crop_resized.shape)
        #         # img_crop_resized = self.test_resize(img_crop)
        #         # cv2.imwrite(os.path.join('./%s_%s_3d_to_2d_box_gt.png' % (idx_batch, idx_box)), img_crop_resized.permute(1, 2, 0).cpu().numpy())
        #         img_crop_resized_expand = img_crop_resized.unsqueeze(0)
        #         proposal_feature_list.append(img_crop_resized_expand)
        #
        #     # if len(proposal_feature_list) < 1:
        #     #     continue
        #     try:
        #         img_proposal = torch.cat(proposal_feature_list, 0)
        #     except:
        #         print('No object on GT, too')
        #         continue
        #     # im_proposal_ref = np.load('img_proposal.npy')
        #     # image_proposal_batch_ref = np.load('image_proposal_batch.npy')
        #     # proposal_features_batch_ref = np.load('proposal_features_batch.npy')
        #
        #     # img_proposal = torch.cat(proposal_feature_list, 0)
        #     image_proposal_batch = self.preprocess_for_tensor(img_proposal).unsqueeze(0)[0]  # change version
        #     # image_proposal_batch_copy = image_proposal_batch.detach().clone()
        #     # proposal_features_batch = self.res_encoder(image_proposal_batch, if_pool=True, if_early_feat=False)
        #     proposal_features_batch = self.clip_model.encode_image(image_proposal_batch)  # change version
        #     # proposal_features_batch_1 = self.clip_model_1.encode_image(image_proposal_batch_copy)  # change version
        #     # assert torch.equal(proposal_features_batch, proposal_features_batch_0)
        #     if isinstance(proposal_features_batch, tuple):
        #         proposal_features_batch = proposal_features_batch[0]
        #     proposal_features_batch = proposal_features_batch.to(torch.float32)
        #     # tmp = outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list]
        #     outputs['gt_text_correlation_embedding'][idx_batch][idx_box_list] = proposal_features_batch
        #     outputs['gt_text_correlation_embedding_mask'][idx_batch][idx_box_list] = 1
        #
        #     # ob_idx_box_list = locat_ob_list
        #     # print(1)
        # if self.if_clip_weak_labels:
        #     text_features_clip = outputs["text_features_clip"].to(torch.float32)
        #     # print(text_features_clip.shape)
        #     temperature_param = outputs["logit_scale"]
        #     gt_image_embedding = outputs['gt_text_correlation_embedding']
        #     gt_image_embedding_mask = outputs['gt_text_correlation_embedding_mask']
        #     gt_image_embedding = gt_image_embedding / (
        #             gt_image_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        #     correlation_map = torch.bmm(gt_image_embedding,
        #                                 text_features_clip.permute(0, 2, 1)) * temperature_param
        #     box_scores = torch.nn.functional.softmax(correlation_map, dim=-1)
        #     max_score, max_id = torch.max(box_scores, dim=-1)
        #     # bg_id = correlation_map.shape[-1]
        #     weak_confidence_weight = max_score
        #     weak_box_cate_label = max_id
        #     # weak_box_cate_label[gt_image_embedding_mask[:,:,0] < 1] = bg_id # if cann't project to find a 2d box and also cann't assign in the loss, it is taken as a wrong box
        #     weak_confidence_weight[
        #         gt_image_embedding_mask[:, :,
        #         0] < 1] = 0.0  # confience_weight for bg object is set to 0, because it is no
        #     outputs["weak_box_cate_label"] = weak_box_cate_label
        #     outputs["weak_confidence_weight"] = weak_confidence_weight
        # just make sure the corners are right
        # for idx_batch in range(center_2dpoints.shape[0]):
        #     img_draw2d_box = inputs['input_image'][idx_batch].cpu().numpy()
        #     center_2dpoints_np = center_2dpoints.cpu().numpy()
        #     for idx_box in range(center_2dpoints.shape[1]):
        #
        #         # img_draw2d_box = sunrgbd_utils.draw_projected_box3d(img_draw2d_box, center_2dpoints_np[idx_batch][idx_box])
        #         img_draw2d_boximg_draw2d_box = img_draw2d_box
        #         cv2.imwrite(os.path.join('./debug_show/%s_%s_3d_to_2d_box_im_gt.png' % (idx_batch, idx_box)), img_draw2d_box)

        return outputs

    def cal_iou(self, pred_box_select, gt_box_select):
        x1 = pred_box_select[0]
        y1 = pred_box_select[1]
        z1 = pred_box_select[2]
        x2 = pred_box_select[3]
        y2 = pred_box_select[4]
        z2 = pred_box_select[5]
        a1 = gt_box_select[0]
        b1 = gt_box_select[1]
        c1 = gt_box_select[2]
        a2 = gt_box_select[3]
        b2 = gt_box_select[4]
        c2 = gt_box_select[5]

        xx1 = torch.maximum(x1, a1)
        yy1 = torch.maximum(y1, b1)
        zz1 = torch.maximum(z1, c1)
        xx2 = torch.minimum(x2, a2)
        yy2 = torch.minimum(y2, b2)
        zz2 = torch.minimum(z2, c2)
        zeros_tensor = torch.zeros_like(xx1, device=self.device)
        l = torch.maximum(zeros_tensor, xx2 - xx1)
        w = torch.maximum(zeros_tensor, yy2 - yy1)
        h = torch.maximum(zeros_tensor, zz2 - zz1)
        area1 = (x2 - x1) * (y2 - y1) * (z2 - z1)
        area2 = (a2 - a1) * (b2 - b1) * (c2 - c1)
        inter = l * w * h
        iou = inter / (area1 + area2 - inter)
        return iou

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features, point_clouds, inputs):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        # seen_cls_logits = self.mlp_heads["seen_sem_cls_head"](box_features).transpose(1, 2)
        center_offset = (
            self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
        )
        size_normalized = (
            self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
        )
        angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](
            box_features
        ).transpose(1, 2)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        # seen_cls_logits = seen_cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(
            num_layers, batch, num_queries, -1
        )
        angle_residual = angle_residual_normalized * (
            np.pi / angle_residual_normalized.shape[-1]
        )

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(
                center_offset[l], query_xyz, point_cloud_dims
            )
            angle_continuous = self.box_processor.compute_predicted_angle(
                angle_logits[l], angle_residual[l]
            )
            size_unnormalized = self.box_processor.compute_predicted_size(
                size_normalized[l], point_cloud_dims
            )
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )
            box_corners_xyz = self.box_processor.box_parametrization_to_corners_xyz(
                center_unnormalized, size_unnormalized, angle_continuous
            )
            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            # with torch.no_grad():
            #     (
            #         semcls_prob,
            #         objectness_prob,
            #     ) = self.box_processor.compute_objectness_and_cls_prob_sigmoid(cls_logits[l])

            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            # with torch.no_grad():
            #     (
            #         seen_semcls_prob,
            #         seen_objectness_prob,
            #     ) = self.box_processor.compute_objectness_and_cls_prob_sigmoid(seen_cls_logits[l])

            # #debug
            # semcls_prob = seen_semcls_prob
            # objectness_prob = seen_objectness_prob
            # print('======================')
            # print(semcls_prob)
            # print(semcls_prob.shape)
            # print(objectness_prob)
            # if l==7:
            #     print(objectness_prob.shape)
            # #debug

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                # "seen_sem_cls_logits": seen_cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                # "seen_objectness_prob": seen_objectness_prob,
                "sem_cls_prob": semcls_prob,
                # "seen_sem_cls_prob": seen_semcls_prob,
                "box_corners": box_corners,
                "box_corners_xyz": box_corners_xyz,
                "point_clouds": point_clouds
            }
            # print(objectness_prob.shape)
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]
        # if self.if_with_clip:
        #     outputs = self.clip_to_class_training(inputs, outputs, if_use_gt_box=self.if_use_gt_box, if_expand_box=self.if_expand_box)
        # if self.if_with_clip_embed:
        #     outputs =  self.clip_to_class_embedding(inputs, outputs, if_use_gt_box=self.if_use_gt_box, if_expand_box=self.if_expand_box)
        # if self.if_with_clip_train:
        #     outputs = self.clip_to_class_training(inputs, outputs, if_use_gt_box=self.if_use_gt_box, if_expand_box=self.if_expand_box)
        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False, if_test=False, curr_epoch=-1, if_real_test=False, if_cmp_class=False):
        point_clouds = inputs["point_clouds"]
        #print(point_clouds.shape)

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = self.encoder_to_decoder_projection(
            enc_features.permute(1, 2, 0)
        ).permute(2, 0, 1)
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, enc_features.transpose(0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = enc_pos.permute(2, 0, 1)
        query_embed = query_embed.permute(2, 0, 1)
        tgt = torch.zeros_like(query_embed)
        box_features = self.decoder(
            tgt, enc_features, query_pos=query_embed, pos=enc_pos
        )[0]

        box_predictions = self.get_box_predictions(
            query_xyz, point_cloud_dims, box_features, point_clouds, inputs
        )
        box_predictions["outputs"]["logit_scale"] = torch.clip(self.logit_scale.exp(), min=None, max=100)
        if (not if_real_test) and (not if_cmp_class) and (not if_test): # do not calculate when testing
            if self.if_clip_superset:
                box_predictions["outputs"]["text_features_clip"] = self.superset_text_features_fg_norm.unsqueeze(
                    0).repeat(point_clouds.shape[0], 1, 1)  # text_features_clip
            else:
                box_predictions["outputs"]["text_features_clip"] = self.text_features_fg_norm[:self.train_range_max, :].unsqueeze(0).repeat(point_clouds.shape[0], 1, 1)  # text_features_clip
            if self.online_nms_update_save_novel_label_clip_driven_with_cate_confidence:
                if self.if_clip_superset:
                    box_predictions["outputs"]["maybe_novel_text_features_clip"] = self.superset_text_features_fg_norm
                else:
                    box_predictions["outputs"]["maybe_novel_text_features_clip"] = self.text_features_fg_norm[
                                                                                   :self.test_range_max,
                                                                                   :]  # text_features_clip
                box_predictions["outputs"] = self.get_predicted_box_clip_embedding_nms_iou_save_keep_clip_driven_with_cate_confidence(
                    inputs,
                    box_predictions["outputs"],
                    curr_epoch=curr_epoch,
                    if_test=if_test)
        
        if if_real_test:
            box_predictions["outputs"] = self.clip_to_class_training(inputs, box_predictions["outputs"],
                                                                     if_use_gt_box=self.if_use_gt_box,
                                                                     if_expand_box=self.if_expand_box)
        if if_cmp_class:
            box_predictions["outputs"] = self.clip_to_class_training(inputs, box_predictions["outputs"],
                                                                     if_use_gt_box=self.if_use_gt_box,
                                                                     if_expand_box=self.if_expand_box, if_cmp_class=if_cmp_class)
        # print(box_predictions["outputs"]['sem_cls_prob'].shape)
        return box_predictions


def build_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder

def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )
    elif args.enc_type in ["masked"]:
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
        )
        interim_downsampling = PointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder

def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=True
    )
    return decoder

def build_3detr(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        if_with_clip=args.if_with_clip,
        if_with_clip_embed=args.if_with_clip_embed,
        if_use_gt_box=args.if_use_gt_box,
        if_expand_box=args.if_expand_box
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor

def build_3detr_predictedbox_distillation_head(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args) #build_image_text_encoder(args)
    decoder = build_decoder(args)
    # image_text_encoder = build_image_text_encoder(args)
    model = Model3DETRPredictedBoxDistillationHead(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        # image_text_encoder=image_text_encoder,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        if_with_clip=args.if_with_clip,
        if_with_clip_embed=args.if_with_clip_embed,
        if_use_gt_box=args.if_use_gt_box,
        if_expand_box=args.if_expand_box,
        if_with_fake_classes=args.if_with_fake_classes,
        pooling_methods=args.pooling_methods,
        if_clip_more_prompts=args.if_clip_more_prompts,
        if_keep_box=args.if_keep_box,
        if_select_box_by_objectness=args.if_select_box_by_objectness,
        keep_objectness=args.keep_objectness,
        online_nms_update_novel_label=args.online_nms_update_novel_label,
        online_nms_update_accumulate_novel_label=args.online_nms_update_accumulate_novel_label,
        online_nms_update_accumulate_epoch=args.online_nms_update_accumulate_epoch,
        distillation_box_num = args.distillation_box_num,
        args = args
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor

def build_3detr_multiclasshead(args, dataset_config):
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETRMultiClassHead(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        if_with_clip=args.if_with_clip,
        if_with_clip_embed=args.if_with_clip_embed,
        if_use_gt_box=args.if_use_gt_box,
        if_expand_box=args.if_expand_box,
        args=args
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor


