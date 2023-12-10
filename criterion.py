# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou, box3d_iou, is_clockwise
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss

class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        pred_cls_prob = outputs["sem_cls_prob"]
        gt_box_sem_cls_labels = (
            targets["gt_box_sem_cls_label"]
            .unsqueeze(1)
            .expand(batchsize, nqueries, ngt)
        )
        class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=pred_cls_prob.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=pred_cls_prob.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_cls_prob.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict, train_range_max=37, only_image_class=False, only_prompt_loss=False, args=None):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict
        self.only_image_class = only_image_class
        self.only_prompt_loss = only_prompt_loss
        self.zeros = torch.zeros(1).to('cuda')
        semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)
        print('==============how many classes for contrastive losses(including bg) in loss(during training)===========')
        assert train_range_max==37 or train_range_max==232 or train_range_max==10
        print(train_range_max + 1)
        seen_semcls_percls_weights = torch.ones(train_range_max + 1)
        seen_semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_contrast_weight"]
        self.register_buffer("seen_semcls_percls_weights", seen_semcls_percls_weights)
        self.if_skip_no_seen_scene_objectness = args.if_skip_no_seen_scene_objectness
        print('==================if_skip_no_seen_scene_objectness==================')
        print(self.if_skip_no_seen_scene_objectness)
        del loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_contrast_weight"]
        if loss_weight_dict["loss_sem_cls_softmax_skip_none_gt_sample_weight"] > 0 or loss_weight_dict["loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness_weight"] > 0 or loss_weight_dict['loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness_weight'] > 0:
            self.if_skip_no_seen_scene_objectness_sample = True
        else:
            self.if_skip_no_seen_scene_objectness_sample = False
        print('==================if_skip_no_seen_scene_objectness_sample==================')
        print(self.if_skip_no_seen_scene_objectness_sample)
        # self.logit_scale = nn.Parameter(torch.ones([]) * 100.0, requires_grad=False).to('cuda')
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to('cuda')
        # self.logit_scale = self.logit_scale.exp()
        # print('losssssssssssssssssssssssssssssssssssssssss')
        # print(self.logit_scale)
        # print('==========================')
        self.confidence_type = args.confidence_type
        assert self.confidence_type in ["non-confidence", "objectness", "clip+objectness", "clip-max-prob"]
        print('====================confidence_type=======================')
        print(self.confidence_type)
        self.if_only_seen_in_loss = args.if_only_seen_in_loss
        print('====================if_only_seen_in_loss==================')
        print(self.if_only_seen_in_loss)
        self.confidence_type_in_datalayer = args.confidence_type_in_datalayer
        if self.if_only_seen_in_loss:
            assert self.confidence_type_in_datalayer == 'zero-out'
        # self.l1_loss = torch.nn.L1Loss()
        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_sem_cls_softmax": self.loss_sem_cls_softmax,
            "loss_sem_cls_softmax_skip_none_gt_sample": self.loss_sem_cls_softmax_skip_none_gt_sample,
            "loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample": self.loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample,
            "loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness": self.loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness,
            "loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness": self.loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness,
            "loss_sem_cls_softmax_discovery_novel_objectness": self.loss_sem_cls_softmax_discovery_novel_objectness,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
            "loss_contrastive": self.loss_contrastive,
            "loss_sem_focal_cls": self.loss_sem_focal_cls,
            "loss_contrast_object_text": self.loss_contrast_object_text,
            "loss_region_embed": self.loss_region_embed,
            "loss_predicted_region_embed_l1": self.loss_predicted_region_embed_l1,
            "loss_predicted_region_embed_l1_only_last_layer": self.loss_predicted_region_embed_l1_only_last_layer,
            "loss_predicted_region_embed_cos": self.loss_predicted_region_embed_cos,
            "loss_image_seen_class": self.loss_image_seen_class,
            "loss_batchwise_contrastive": self.loss_batchwise_contrastive,
            "loss_feat_seen_sigmoid_loss": self.loss_feat_seen_sigmoid_loss,
            "loss_feat_seen_softmax_loss": self.loss_feat_seen_softmax_loss,
            "loss_feat_seen_softmax_weakly_loss": self.loss_feat_seen_softmax_weakly_loss,
            "loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi": self.loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi,
            "loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi": self.loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi,
            "loss_feat_seen_softmax_loss_with_novel_cate_confi": self.loss_feat_seen_softmax_loss_with_novel_cate_confi,
            "loss_feat_seen_sigmoid_with_full_image_loss": self.loss_feat_seen_sigmoid_with_full_image_loss,
            "loss_prompt_softmax": self.loss_prompt_softmax,
            "loss_prompt_sigmoid": self.loss_prompt_sigmoid
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["sem_cls_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}

    def loss_sem_cls_softmax(self, outputs, targets, assignments):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")
        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )
        if self.if_skip_no_seen_scene_objectness:
            if targets["num_boxes_replica"] == 0:
                loss = torch.sum(pred_logits) * 0
        return {"loss_sem_cls_softmax": loss}

    def loss_sem_cls_softmax_skip_none_gt_sample(self, outputs, targets, assignments):

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="none",
        )
        final_loss = 0
        cnt_has_object = 0
        for batch_id in range(loss.shape[0]):
            num_obj_thisbatch = torch.sum(targets['gt_box_present'][batch_id])
            if num_obj_thisbatch == 0:
                final_loss += (torch.sum(loss[batch_id]) * 0)
            else:
                final_loss += torch.sum(loss[batch_id])
                cnt_has_object += 1.0

        final_loss = final_loss/(cnt_has_object * loss.shape[1] + 1e-32)

        return {"loss_sem_cls_softmax_skip_none_gt_sample": final_loss}

    def loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample(self, outputs, targets, assignments):

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        novel_box_judge = targets['novel_box_judge']
        gt_box_label[novel_box_judge > 0] = 0
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="none",
        )
        final_loss = 0
        cnt_has_object = 0
        for batch_id in range(loss.shape[0]):
            num_obj_thisbatch = torch.sum(targets['gt_box_present'][batch_id])
            num_novel_box_judge = torch.sum(novel_box_judge[batch_id])
            if num_obj_thisbatch == 0 and num_novel_box_judge == 0:
                final_loss += (torch.sum(loss[batch_id]) * 0)
            else:
                final_loss += torch.sum(loss[batch_id])
                cnt_has_object += 1.0

        final_loss = final_loss/(cnt_has_object * loss.shape[1] + 1e-32)

        return {"loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample": final_loss}


    def loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness(self, outputs, targets, assignments):

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        gt_novel_discovery = targets['discovery_novel'].type_as(gt_box_label)
        gt_box_label[gt_novel_discovery > 0] = 0
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="none",
        )
        # weight_map = torch.ones_like(loss, device=loss.device)
        # weight_map[gt_novel_discovery > 0] = outputs['objectness_prob'][gt_novel_discovery>0]
        final_loss = 0
        cnt_has_object = 0
        for batch_id in range(loss.shape[0]):
            num_obj_thisbatch = torch.sum(targets['gt_box_present'][batch_id])
            num_discovery = torch.sum(gt_novel_discovery[batch_id])
            if num_obj_thisbatch == 0 and num_discovery == 0: # if there are no gt boxes and no discovery boxes
                final_loss += (torch.sum(loss[batch_id]) * 0)
            else:
                # final_loss += torch.sum(loss[batch_id]*weight_map[batch_id])
                final_loss += torch.sum(loss[batch_id])
                cnt_has_object += 1.0

        final_loss = final_loss/(cnt_has_object * loss.shape[1] + 1e-32)

        return {"loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness": final_loss}

    def loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness(self, outputs, targets, assignments):

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="none",
        )
        final_loss = 0
        cnt_has_object = 0
        gt_novel_discovery = targets['discovery_novel'].type_as(gt_box_label)
        loss_weights = torch.ones_like(loss)
        loss_weights[gt_novel_discovery > 0] = 0 # do not supervise novel boxes
        for batch_id in range(loss.shape[0]):
            num_obj_thisbatch = torch.sum(targets['gt_box_present'][batch_id])
            num_discovery = torch.sum(gt_novel_discovery[batch_id])
            loss_weights_this_batch = loss_weights[batch_id]
            if num_obj_thisbatch == 0:# and num_discovery == 0:
                final_loss += (torch.sum(loss[batch_id]) * 0)
            else:
                final_loss += torch.sum(loss[batch_id] * loss_weights_this_batch)
                cnt_has_object += torch.sum(loss_weights_this_batch)

        final_loss = final_loss/(cnt_has_object + 1e-32)

        return {"loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness": final_loss}

    def loss_sem_cls_softmax_discovery_novel_objectness(self, outputs, targets, assignments):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        gt_novel_discovery = targets['discovery_novel'].type_as(gt_box_label)
        gt_box_label[gt_novel_discovery > 0] = 0
        loss = F.cross_entropy(
            pred_logits.transpose(2, 1),
            gt_box_label,
            self.semcls_percls_weights,
            reduction="mean",
        )
        return {"loss_sem_cls_softmax_discovery_novel_objectness": loss}

    def loss_sem_cls(self, outputs, targets, assignments):

        pred_logits = outputs["sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        gt_box_label_focal = F.one_hot(gt_box_label)
        gt_box_label_focal = gt_box_label_focal.type_as(pred_logits)
        # print(gt_box_label_focal.shape)
        # for i in range(gt_box_label_focal.shape[0]):
        loss = sigmoid_focal_loss(pred_logits, gt_box_label_focal, reduction='mean')

        return {"loss_sem_cls": loss}

    def loss_image_seen_class(self, outputs, targets, assignments):

        pred_logits = outputs["seen_class_scores_per_image"]
        gt_image_label = targets["gt_image_class_label"]
        # print(gt_image_label)
        # print(gt_image_label)
        # gt_box_label = torch.gather(
        #     targets["gt_box_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        # )
        # gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
        #     pred_logits.shape[-1] - 1
        # )
        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        # gt_box_label_focal = F.one_hot(gt_box_label)
        # gt_box_label_focal = gt_box_label_focal.type_as(pred_logits)
        # print(gt_box_label_focal.shape)
        # for i in range(gt_box_label_focal.shape[0]):
        loss = sigmoid_focal_loss(pred_logits, gt_image_label.type_as(pred_logits), reduction='mean')

        return {"loss_image_seen_class": loss}

    def loss_contrast_object_text(self, outputs, targets, assignments):
        gt_box_label = torch.gather(
            targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )

        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        text_features_clip = targets["text_features_clip"]
        temperature_param = targets["logit_scale"]
        # temperature_param = torch.clip()
        # correlation_map = torch.bmm(text_correlation_embedding, text_features_clip.permute(0, 2, 1)) / 0.07 #* temperature_param
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1))  * temperature_param
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = correlation_map.shape[-1]-1
        loss = F.cross_entropy(
            correlation_map.permute(0, 2, 1), # 8, 38, 128
            gt_box_label, # 8, 128
            self.seen_semcls_percls_weights,
            reduction="mean",
            )
        return {"loss_contrast_object_text": loss}


    def loss_contrastive(self, outputs, targets, assignments):
        text_feats = outputs["pooled_updated_text_features"]
        image_feats = outputs["image_features_clip"]
        # logit_scale = outputs["logit_scale"]
        similarity = self.logit_scale * image_feats @ text_feats.t()
        labels = torch.arange(similarity.shape[0], device='cuda')
        loss_i = F.cross_entropy(
            similarity,
            labels,
            reduction="mean",
        )
        loss_t = F.cross_entropy(
            similarity.permute(1, 0),
            labels,
            reduction="mean",
        )

        return {"loss_contrastive": (loss_i+loss_t)/2.0}

    # def bias_init_with_prob(self, prior_prob: float) -> float:
    #     """initialize conv/fc bias value according to a given probability value."""
    #     bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    #     return bias_init
    #
    # nn.init.constant_(layer.bias, bias_init_with_prob(0.01))
    def loss_feat_seen_sigmoid_loss(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        # text_correlation_embedding = text_correlation_embedding / (
        #             text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        text_features_clip = targets["text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        # temperature_param = outputs["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1))
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        gt_box_label = torch.gather(targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"])


        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (correlation_map.shape[-1])
        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        gt_box_label_focal = F.one_hot(gt_box_label)[:,:,:-1]
        gt_box_label_focal = gt_box_label_focal.type_as(correlation_map)
        # for i in range(gt_box_label_focal.shape[0]):
        loss = sigmoid_focal_loss(correlation_map, gt_box_label_focal, reduction='none')
        # loss_weight = torch.ones_like(loss, device=loss.device)
        loss_pos = torch.ones_like(loss, device=loss.device)
        loss_neg = torch.ones_like(loss, device=loss.device)
        loss_neg[:, :, 10:] = 0.0
        loss_weight = torch.where(assignments["proposal_matched_mask"].unsqueeze(-1).repeat(1, 1, loss.shape[-1]).int() != 0, loss_pos, loss_neg) # do not apply novel signal for un-matched boxes

        all_num = torch.sum(assignments["proposal_matched_mask"].int() != 0) * loss.shape[-1] + torch.sum(assignments["proposal_matched_mask"].int() == 0) * 10
        final_loss = torch.sum(loss * loss_weight) / all_num
        return {"loss_feat_seen_sigmoid_loss": final_loss}

    def loss_feat_seen_softmax_loss(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (
                    text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)

        text_features_clip = targets["text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = targets["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1)) * temperature_param
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        gt_box_label = torch.gather(targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"])
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (correlation_map.shape[-1]-1)
        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        loss = F.cross_entropy(
            correlation_map.transpose(2, 1),
            gt_box_label,
            reduction="none",
        )
        # loss_weight = torch.ones_like(loss, device=loss.device)
        loss_pos = torch.ones_like(loss, device=loss.device)
        loss_neg = torch.zeros_like(loss, device=loss.device)
        loss_weight = torch.where(assignments["proposal_matched_mask"].int() != 0, loss_pos, loss_neg) # do not apply novel signal for un-matched boxes

        all_num = torch.sum(assignments["proposal_matched_mask"].int() != 0) * loss.shape[-1]
        final_loss = torch.sum(loss * loss_weight) / all_num
        return {"loss_feat_seen_softmax_loss": final_loss}


    def loss_feat_seen_softmax_weakly_loss(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (
                    text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)

        text_features_clip = targets["text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = targets["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1)) * temperature_param
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        seen_matched_box_label = torch.gather(targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"])
        # gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (correlation_map.shape[-1]-1)
        gt_box_label = torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_box_label, targets['weak_box_cate_label'])
        if self.confidence_type=="clip-max-prob":
            gt_box_confidence_map = targets['weak_confidence_weight'].detach().clone() #torch.where(assignments["proposal_matched_mask"].int()>0, seen_confidence, targets['weak_confidence_weight'])
            gt_box_confidence_map[torch.logical_and(assignments["proposal_matched_mask"].int() > 0, gt_box_label !=-1)] = 1.0
        elif self.confidence_type=="non-confidence":
            gt_box_confidence_map = torch.ones_like(gt_box_label, device=gt_box_label.device)
        elif self.confidence_type=="objectness":
            gt_box_confidence_map = outputs['objectness_prob'].detach().clone()
            gt_box_confidence_map[
                torch.logical_and(assignments["proposal_matched_mask"].int() > 0, gt_box_label != -1)] = 1.0
        elif self.confidence_type=="clip+objectness":
            gt_box_confidence_map = (outputs['objectness_prob'].detach().clone()+targets['weak_confidence_weight'].detach().clone())/2.0
            gt_box_confidence_map[torch.logical_and(assignments["proposal_matched_mask"].int() > 0, gt_box_label != -1)] = 1.0

        gt_box_confidence = gt_box_confidence_map

        gt_box_label = torch.where(gt_box_label == -1, targets['weak_box_cate_label'], gt_box_label) # when the seen is -1(means novel boxes), use weak label
        # seen_confidence = torch.ones_like(targets['weak_confidence_weight'], device=targets['weak_confidence_weight'].device)

        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        loss = F.cross_entropy(
            correlation_map.transpose(2, 1),
            gt_box_label,
            reduction="none",
        )
        # loss_weight = torch.ones_like(loss, device=loss.device)
        # loss_pos = torch.ones_like(loss, device=loss.device)
        # loss_neg = torch.zeros_like(loss, device=loss.device)
        # loss_weight = torch.where(assignments["proposal_matched_mask"].int() != 0, loss_pos, loss_neg) # do not apply novel signal for un-matched boxes
        #
        # all_num = torch.sum(assignments["proposal_matched_mask"].int() != 0) * loss.shape[-1]
        all_num = torch.sum(gt_box_confidence > 1e-32)
        final_loss = torch.sum(loss * gt_box_confidence) / all_num
        return {"loss_feat_seen_softmax_weakly_loss": final_loss}


    def loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (
                    text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)

        text_features_clip = targets["text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = targets["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1)) * temperature_param
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        seen_matched_box_label = torch.gather(targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"])
        seen_matched_confidence = torch.gather(targets["gt_box_seen_sem_cls_confi"], 1, assignments["per_prop_gt_inds"])
        # gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (correlation_map.shape[-1]-1)
        gt_box_label = torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_box_label, targets['weak_box_cate_label'])
        if self.confidence_type=="clip-max-prob":
            gt_box_confidence_map = torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_confidence, targets['weak_confidence_weight'])
        elif self.confidence_type=="non-confidence":
            gt_box_confidence_map = torch.where(assignments["proposal_matched_mask"].int() > 0, seen_matched_confidence,
                                                targets['weak_confidence_weight'])
            gt_box_confidence_map[gt_box_confidence_map>1e-16] = 1
        # gt_box_confidence_map = targets['weak_confidence_weight'].detach().clone() #torch.where(assignments["proposal_matched_mask"].int()>0, seen_confidence, targets['weak_confidence_weight'])
        # gt_box_confidence_map[torch.logical_and(assignments["proposal_matched_mask"].int() > 0, gt_box_label !=-1)] = 1.0
        gt_box_confidence = gt_box_confidence_map

        # gt_box_label = torch.where(gt_box_label == -1, targets['weak_box_cate_label'], gt_box_label) # when the seen is -1(means novel boxes), use weak label
        # seen_confidence = torch.ones_like(targets['weak_confidence_weight'], device=targets['weak_confidence_weight'].device)

        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        loss = F.cross_entropy(
            correlation_map.transpose(2, 1),
            gt_box_label,
            reduction="none",
        )
        # loss_weight = torch.ones_like(loss, device=loss.device)
        # loss_pos = torch.ones_like(loss, device=loss.device)
        # loss_neg = torch.zeros_like(loss, device=loss.device)
        # loss_weight = torch.where(assignments["proposal_matched_mask"].int() != 0, loss_pos, loss_neg) # do not apply novel signal for un-matched boxes
        #
        # all_num = torch.sum(assignments["proposal_matched_mask"].int() != 0) * loss.shape[-1]
        all_num = torch.sum(gt_box_confidence > 1e-32) + 1e-32
        final_loss = torch.sum(loss * gt_box_confidence) / all_num
        return {"loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi": final_loss}

    def loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (
                    text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)

        text_features_clip = targets["text_features_clip"].to(torch.float32)
        # print(text_features_clip.shape)
        temperature_param = targets["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1)) * temperature_param
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        # seen_matched_box_label = torch.gather(targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"])
        # seen_matched_confidence = torch.gather(targets["gt_box_seen_sem_cls_confi"], 1, assignments["per_prop_gt_inds"])
        # # gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (correlation_map.shape[-1]-1)
        # gt_box_label = torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_box_label, targets['weak_box_cate_label'])
        # gt_box_confidence_map = torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_confidence, targets['weak_confidence_weight'])
        # # gt_box_confidence_map = targets['weak_confidence_weight'].detach().clone() #torch.where(assignments["proposal_matched_mask"].int()>0, seen_confidence, targets['weak_confidence_weight'])
        # # gt_box_confidence_map[torch.logical_and(assignments["proposal_matched_mask"].int() > 0, gt_box_label !=-1)] = 1.0
        # gt_box_confidence = gt_box_confidence_map
        gt_box_label = targets['weak_box_cate_label']
        gt_box_confidence = targets['weak_confidence_weight']
        # gt_box_label = torch.where(gt_box_label == -1, targets['weak_box_cate_label'], gt_box_label) # when the seen is -1(means novel boxes), use weak label
        # seen_confidence = torch.ones_like(targets['weak_confidence_weight'], device=targets['weak_confidence_weight'].device)

        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        loss = F.cross_entropy(
            correlation_map.transpose(2, 1),
            gt_box_label,
            reduction="none",
        )
        # loss_weight = torch.ones_like(loss, device=loss.device)
        # loss_pos = torch.ones_like(loss, device=loss.device)
        # loss_neg = torch.zeros_like(loss, device=loss.device)
        # loss_weight = torch.where(assignments["proposal_matched_mask"].int() != 0, loss_pos, loss_neg) # do not apply novel signal for un-matched boxes
        #
        # all_num = torch.sum(assignments["proposal_matched_mask"].int() != 0) * loss.shape[-1]
        all_num = torch.sum(gt_box_confidence > 1e-32) + 1e-32
        final_loss = torch.sum(loss * gt_box_confidence) / all_num
        return {"loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi": final_loss}

    def loss_feat_seen_softmax_loss_with_novel_cate_confi(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        text_correlation_embedding = text_correlation_embedding / (
                    text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)

        text_features_clip = targets["text_features_clip"].to(torch.float32)
        if self.if_only_seen_in_loss:
            text_features_clip = text_features_clip[:,:10,:]
        # print(text_features_clip.shape)
        temperature_param = targets["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1)) * temperature_param
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        seen_matched_box_label = torch.gather(targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"])
        seen_matched_confidence = torch.gather(targets["gt_box_seen_sem_cls_confi"], 1, assignments["per_prop_gt_inds"])
        if self.if_only_seen_in_loss:
            seen_matched_box_label[seen_matched_confidence<1e-32] = 0 # just give up the label for confidenc=0, although here set it = 0, it will not works, because the confidence=0
        seen_matched_confidence[assignments["proposal_matched_mask"].int() == 0] = 0.0
        # gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (correlation_map.shape[-1]-1)
        #zeros_confidence_map = seen_matched_box_label #torch.zeros_like(seen_matched_box_label, device='cuda')
        gt_box_label = seen_matched_box_label #torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_box_label, targets['weak_box_cate_label'])
        gt_box_confidence_map = seen_matched_confidence #torch.where(assignments["proposal_matched_mask"].int()>0, seen_matched_confidence, targets['weak_confidence_weight'])
        # gt_box_confidence_map = targets['weak_confidence_weight'].detach().clone() #torch.where(assignments["proposal_matched_mask"].int()>0, seen_confidence, targets['weak_confidence_weight'])
        # gt_box_confidence_map[torch.logical_and(assignments["proposal_matched_mask"].int() > 0, gt_box_label !=-1)] = 1.0
        gt_box_confidence = gt_box_confidence_map

        # gt_box_label = torch.where(gt_box_label == -1, targets['weak_box_cate_label'], gt_box_label) # when the seen is -1(means novel boxes), use weak label
        # seen_confidence = torch.ones_like(targets['weak_confidence_weight'], device=targets['weak_confidence_weight'].device)

        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        loss = F.cross_entropy(
            correlation_map.transpose(2, 1),
            gt_box_label,
            reduction="none",
        )
        # loss_weight = torch.ones_like(loss, device=loss.device)
        # loss_pos = torch.ones_like(loss, device=loss.device)
        # loss_neg = torch.zeros_like(loss, device=loss.device)
        # loss_weight = torch.where(assignments["proposal_matched_mask"].int() != 0, loss_pos, loss_neg) # do not apply novel signal for un-matched boxes
        #
        # all_num = torch.sum(assignments["proposal_matched_mask"].int() != 0) * loss.shape[-1]
        all_num = torch.sum(gt_box_confidence > 1e-32)+1e-16
        final_loss = torch.sum(loss * gt_box_confidence) / all_num
        return {"loss_feat_seen_softmax_loss_with_novel_cate_confi": final_loss}

    def loss_prompt_sigmoid(self, outputs, targets, assignments):
        temperature_param = outputs['prompt_temperature_param']
        text_correlation_embedding = outputs['prompt_text_correlation_embedding']
        # text_correlation_embedding = text_correlation_embedding / (
        #             text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        text_features_clip =  outputs['prompt_text_features_clip'] #.to(torch.float16)
        # print(text_features_clip.shape)
        # temperature_param = outputs["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1))[:, 0, :]
        gt_seen = targets['seen_classes']
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]
        gt_box_label_focal = F.one_hot(gt_seen, num_classes=10)
        gt_box_label_focal = gt_box_label_focal.type_as(correlation_map)
        # print(gt_box_label_focal.shape)
        # for i in range(gt_box_label_focal.shape[0]):
        loss = sigmoid_focal_loss(correlation_map, gt_box_label_focal, reduction='mean') + 0 * temperature_param

        return loss

    def loss_prompt_softmax(self, outputs, targets, assignments):
        # temperature_param = outputs['prompt_temperature_param']
        gt_box_label = targets['seen_classes']

        text_correlation_embedding = outputs['prompt_text_correlation_embedding']
        text_correlation_embedding = text_correlation_embedding / (text_correlation_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        text_correlation_embedding = text_correlation_embedding
        text_features_clip = outputs['prompt_text_features_clip']
        temperature_param = outputs['prompt_temperature_param']
        # temperature_param = torch.clip()
        # correlation_map = torch.bmm(text_correlation_embedding, text_features_clip.permute(0, 2, 1)) / 0.07 #* temperature_param
        correlation_map = torch.bmm(text_correlation_embedding,
                                    text_features_clip.permute(0, 2, 1))  * temperature_param
        correlation_map = correlation_map[:, 0,:]
        loss = F.cross_entropy(
            correlation_map, # 8, 38, 128
            gt_box_label, # 8, 128
            reduction="mean",
            )
        return  loss

    def loss_feat_seen_sigmoid_with_full_image_loss(self, outputs, targets, assignments):

        text_correlation_embedding = outputs["text_correlation_embedding"]
        full_image_embedding = targets['full_image_embedding'].unsqueeze(1)
        # full_image_embedding = full_image_embedding / (
        #             full_image_embedding.norm(dim=-1, keepdim=True) + 1e-32)
        text_features_clip = targets["text_features_clip"].to(torch.float32)[:, :10, :]
        # print(text_features_clip.shape)
        # temperature_param = outputs["logit_scale"]
        correlation_map = torch.bmm(text_correlation_embedding,
                                    (text_features_clip * full_image_embedding).permute(0, 2, 1))
        # scores = torch.nn.functional.softmax(correlation_map, dim=-1)[:, :, :]

        gt_box_label = torch.gather(
            targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            correlation_map.shape[-1]
        )
        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        gt_box_label_focal = F.one_hot(gt_box_label)[:,:,:-1]
        gt_box_label_focal = gt_box_label_focal.type_as(correlation_map)
        # print(gt_box_label_focal.shape)
        # for i in range(gt_box_label_focal.shape[0]):
        loss = sigmoid_focal_loss(correlation_map, gt_box_label_focal, reduction='mean')

        return {"loss_feat_seen_sigmoid_with_full_image_loss": loss}


    def loss_batchwise_contrastive(self, outputs, targets, assignments):
        text_feats = outputs["text_queried_embedding"]
        image_feats = outputs["image_queried_embedding"]
        text_feats_pooled = torch.mean(text_feats, dim=1)
        image_feats_pooled = torch.mean(image_feats, dim=1)
        text_feats_pooled = text_feats_pooled / (text_feats_pooled.norm(dim=-1, keepdim=True) + 1e-32)
        image_feats_pooled = image_feats_pooled / (image_feats_pooled.norm(dim=-1, keepdim=True) + 1e-32)
        # logit_scale = outputs["logit_scale"]
        temperature_param = targets["logit_scale"]
        similarity = temperature_param * image_feats_pooled @ text_feats_pooled.t()
        labels = torch.arange(similarity.shape[0], device='cuda')
        loss_i = F.cross_entropy(
            similarity,
            labels,
            reduction="mean",
        )
        loss_t = F.cross_entropy(
            similarity.permute(1, 0),
            labels,
            reduction="mean",
        )
        return {"loss_batchwise_contrastive": (loss_i+loss_t)/2.0}

    def loss_angle(self, outputs, targets, assignments):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            # # Non vectorized version
            # assignments = assignments["assignments"]
            # p_angle_logits = []
            # p_angle_resid = []
            # t_angle_labels = []
            # t_angle_resid = []

            # for b in range(angle_logits.shape[0]):
            #     if len(assignments[b]) > 0:
            #         p_angle_logits.append(angle_logits[b, assignments[b][0]])
            #         p_angle_resid.append(angle_residual[b, assignments[b][0], gt_angle_label[b][assignments[b][1]]])
            #         t_angle_labels.append(gt_angle_label[b, assignments[b][1]])
            #         t_angle_resid.append(gt_angle_residual_normalized[b, assignments[b][1]])

            # p_angle_logits = torch.cat(p_angle_logits)
            # p_angle_resid = torch.cat(p_angle_resid)
            # t_angle_labels = torch.cat(t_angle_labels)
            # t_angle_resid = torch.cat(t_angle_resid)

            # angle_cls_loss = F.cross_entropy(p_angle_logits, t_angle_labels, reduction="sum")
            # angle_reg_loss = huber_loss(p_angle_resid.flatten() - t_angle_resid.flatten()).sum()

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"] # only calculate the matched boxes
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.sum(angle_logits)*0 # torch.zeros(1, device=angle_logits.device).squeeze() #
            angle_reg_loss = torch.sum(angle_residual)*0 # torch.zeros(1, device=angle_logits.device).squeeze() #
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_region_embed(self, outputs, targets, assignments):

        gt_text_correlation_embedding = torch.gather(
            targets['gt_text_correlation_embedding'], 1, assignments["per_prop_gt_inds"].unsqueeze(-1).repeat(1, 1, targets['gt_text_correlation_embedding'].shape[2])
        )
        # gt_text_correlation_embedding = gt_text_correlation_embedding / gt_text_correlation_embedding.norm(dim=-1, keepdim=True)


        text_correlation_embedding = outputs["text_correlation_embedding"]
        # text_correlation_embedding = text_correlation_embedding / text_correlation_embedding.norm(dim=-1, keepdim=True)

        # gt_text_correlation_embedding[assignments["proposal_matched_mask"].int() == 0] = correlation_map.shape[-1] - 1
        weight_maps = torch.ones_like(assignments["proposal_matched_mask"], device='cuda')
        weight_maps[assignments["proposal_matched_mask"].int() == 0] = 0.0
        weight_maps = weight_maps.unsqueeze(-1)
        ave_weight = text_correlation_embedding.shape[0]*text_correlation_embedding.shape[2]
        # pred_region_embed = outputs["region_embedding"]
        # target_region_embed = outputs["gt_region_embedding"]
        l1_loss = F.l1_loss(text_correlation_embedding*weight_maps/ave_weight, gt_text_correlation_embedding*weight_maps/ave_weight, reduction='sum')
        # print(l1_loss.shape)
        return {"loss_region_embed": l1_loss}

    def loss_predicted_region_embed_l1(self, outputs, targets, assignments):
        gt_text_correlation_embedding = targets['gt_text_correlation_embedding'] #torch.gather(
            #targets['gt_text_correlation_embedding'], 1, assignments["per_prop_gt_inds"].unsqueeze(-1).repeat(1, 1, targets['gt_text_correlation_embedding'].shape[2])
        #)
        # gt_text_correlation_embedding = gt_text_correlation_embedding / gt_text_correlation_embedding.norm(dim=-1, keepdim=True)


        text_correlation_embedding = outputs["text_correlation_embedding"]
        # text_correlation_embedding = text_correlation_embedding / text_correlation_embedding.norm(dim=-1, keepdim=True)

        # gt_text_correlation_embedding[assignments["proposal_matched_mask"].int() == 0] = correlation_map.shape[-1] - 1
        weight_maps = targets['gt_text_correlation_embedding_mask'] # torch.ones_like(assignments["proposal_matched_mask"], device='cuda')
        # weight_maps[assignments["proposal_matched_mask"].int() == 0] = 0.0
        # weight_maps = weight_maps.unsqueeze(-1)
        ave_weight = torch.sum(weight_maps)*text_correlation_embedding.shape[2]
        # pred_region_embed = outputs["region_embedding"]
        # target_region_embed = outputs["gt_region_embedding"]
        l1_loss = F.l1_loss(text_correlation_embedding*weight_maps, gt_text_correlation_embedding*weight_maps, reduction='sum') / ave_weight
        # print(l1_loss.shape)
        return {"loss_predicted_region_embed_l1": l1_loss}

    def loss_predicted_region_embed_l1_only_last_layer(self, outputs, targets, assignments):

        gt_text_correlation_embedding = targets['gt_text_correlation_embedding'] #torch.gather(
            #targets['gt_text_correlation_embedding'], 1, assignments["per_prop_gt_inds"].unsqueeze(-1).repeat(1, 1, targets['gt_text_correlation_embedding'].shape[2])
        #)
        # gt_text_correlation_embedding = gt_text_correlation_embedding / gt_text_correlation_embedding.norm(dim=-1, keepdim=True)


        text_correlation_embedding = outputs["text_correlation_embedding"]
        # text_correlation_embedding = text_correlation_embedding / text_correlation_embedding.norm(dim=-1, keepdim=True)

        # gt_text_correlation_embedding[assignments["proposal_matched_mask"].int() == 0] = correlation_map.shape[-1] - 1
        weight_maps = targets['gt_text_correlation_embedding_mask'] # torch.ones_like(assignments["proposal_matched_mask"], device='cuda')
        # weight_maps[assignments["proposal_matched_mask"].int() == 0] = 0.0
        # weight_maps = weight_maps.unsqueeze(-1)
        ave_weight = torch.sum(weight_maps)*text_correlation_embedding.shape[2]
        # pred_region_embed = outputs["region_embedding"]
        # target_region_embed = outputs["gt_region_embedding"]
        l1_loss = F.l1_loss(text_correlation_embedding*weight_maps, gt_text_correlation_embedding*weight_maps, reduction='sum') / ave_weight
        # print(l1_loss.shape)
        return {"loss_predicted_region_embed_l1_only_last_layer": l1_loss}

    def loss_predicted_region_embed_cos(self, outputs, targets, assignments):

        gt_text_correlation_embedding = targets['gt_text_correlation_embedding'] #torch.gather(
            #targets['gt_text_correlation_embedding'], 1, assignments["per_prop_gt_inds"].unsqueeze(-1).repeat(1, 1, targets['gt_text_correlation_embedding'].shape[2])
        #)
        # gt_text_correlation_embedding = gt_text_correlation_embedding / gt_text_correlation_embedding.norm(dim=-1, keepdim=True)


        text_correlation_embedding = outputs["text_correlation_embedding"]
        # text_correlation_embedding = text_correlation_embedding / text_correlation_embedding.norm(dim=-1, keepdim=True)

        # gt_text_correlation_embedding[assignments["proposal_matched_mask"].int() == 0] = correlation_map.shape[-1] - 1
        weight_maps = targets['gt_text_correlation_embedding_mask'] # torch.ones_like(assignments["proposal_matched_mask"], device='cuda')
        # weight_maps[assignments["proposal_matched_mask"].int() == 0] = 0.0
        # weight_maps = weight_maps.unsqueeze(-1)
        ave_weight = torch.sum(weight_maps)
        # pred_region_embed = outputs["region_embedding"]
        # target_region_embed = outputs["gt_region_embedding"]
        cos_loss = 1 - F.cosine_similarity(gt_text_correlation_embedding, text_correlation_embedding, dim=-1, eps=1e-16)
        final_loss = torch.sum(cos_loss * weight_maps[:,:,0]) / ave_weight
        # print(l1_loss.shape)
        return {"loss_predicted_region_embed_cos": final_loss}

    def loss_sem_focal_cls(self, outputs, targets, assignments):


        pred_logits = outputs["seen_sem_cls_logits"]
        gt_box_label = torch.gather(
            targets["gt_box_seen_sem_cls_label"], 1, assignments["per_prop_gt_inds"]
        )
        gt_box_label[assignments["proposal_matched_mask"].int() == 0] = (
            pred_logits.shape[-1] - 1
        )
        # gt_box_label_focal = torch.ones_like(pred_logits, device='cuda')
        # gt_box_label_focal[:,:,0]=1
        # print('===============================')
        # print(gt_box_label.shape)
        gt_box_label_focal = F.one_hot(gt_box_label)
        gt_box_label_focal = gt_box_label_focal.type_as(pred_logits)
        # print(gt_box_label_focal.shape)
        # for i in range(gt_box_label_focal.shape[0]):
        # print(pred_logits.shape)
        # print(gt_box_label_focal.shape)
        loss = sigmoid_focal_loss(pred_logits, gt_box_label_focal, reduction='mean')

        return {"loss_sem_focal_cls": loss}


    def loss_center(self, outputs, targets, assignments):
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"] # only calculate the matched loss
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.sum(center_dist) * 0 #torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments):
        gious_dist = 1 - outputs["gious"]

        # # Non vectorized version
        # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
        # assign = assignments["assignments"]

        # for b in range(gious_dist.shape[0]):
        #     if len(assign[b]) > 0:
        #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments):
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # p_sizes = []
            # t_sizes = []
            # assign = assignments["assignments"]
            # for b in range(pred_box_sizes.shape[0]):
            #     if len(assign[b]) > 0:
            #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
            #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
            # p_sizes = torch.cat(p_sizes)
            # t_sizes = torch.cat(t_sizes)
            # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"] # only calculate the matched boxes
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.sum(pred_box_sizes) * 0 #torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}

    def single_output_forward(self, outputs, targets, if_region_embed=False, if_aux=False, if_last_head=False):
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        assignments = self.matcher(outputs, targets)

        losses = {}

        for k in self.loss_functions:
            # if if_region_embed==False and k == "loss_region_embed":
            #     continue
            if if_aux and (k=="loss_contrastive" or k=='loss_image_seen_class' or k=='loss_batchwise_contrastive' or k=="loss_3d_2d_region_embed" or k=='loss_predicted_region_embed_l1_only_last_layer'):
                continue
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 1e-32
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                if loss_wt_key != "loss_contrast_3dto2d_text_weight" and loss_wt_key != "loss_3d_2d_region_embed_weight" and loss_wt_key!='loss_predicted_region_embed_l1_only_last_layer_weight':
                    curr_loss = self.loss_functions[k](outputs, targets, assignments)
                    losses.update(curr_loss)
                else:
                    if if_last_head:
                        curr_loss = self.loss_functions[k](outputs, targets, assignments)
                        losses.update(curr_loss)

        final_loss = 0
        for k in self.loss_weight_dict:
            # if if_region_embed==False and k == "loss_region_embed_weight":
            #     continue
            if if_aux and (k=="loss_contrastive_weight" or k=='loss_image_seen_class_weight' or k=='loss_batchwise_contrastive_weight' or k=='loss_3d_2d_region_embed'or k=='loss_predicted_region_embed_l1_only_last_layer'):
                continue

            if self.loss_weight_dict[k] > 1e-32:
                if k != "loss_contrast_3dto2d_text_weight" and k != "loss_3d_2d_region_embed_weight" and k!='loss_predicted_region_embed_l1_only_last_layer_weight':
                    losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                    final_loss += losses[k.replace("_weight", "")]
                else:
                    if if_last_head:
                        losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                        final_loss += losses[k.replace("_weight", "")]

        return final_loss, losses

    def forward(self, outputs, targets):
        if self.only_image_class:
            loss = self.loss_image_seen_class(outputs['outputs'], targets, None)["loss_image_seen_class"] * self.loss_weight_dict['loss_image_seen_class_weight']
            loss_dict = {}
            loss_dict['loss_image_seen_class'] = loss
            return loss, loss_dict

        if self.only_prompt_loss:
            if self.loss_weight_dict['loss_prompt_sigmoid_weight'] > 0:
                loss = self.loss_prompt_sigmoid(outputs, targets, None) * self.loss_weight_dict['loss_prompt_sigmoid_weight']
                loss_dict = {}
                loss_dict['loss_prompt_sigmoid'] = loss
            elif self.loss_weight_dict['loss_prompt_softmax_weight'] > 0:
                loss = self.loss_prompt_softmax(outputs, targets, None) * self.loss_weight_dict['loss_prompt_softmax_weight']
                loss_dict = {}
                loss_dict['loss_prompt_softmax'] = loss

            return loss, loss_dict
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training

        if "text_features_clip" in outputs["outputs"].keys():
            targets["text_features_clip"] = outputs["outputs"]["text_features_clip"]
        if "full_image_embedding" in outputs["outputs"].keys():
            targets['full_image_embedding'] = outputs["outputs"]["full_image_embedding"]
        if "logit_scale" in outputs["outputs"].keys():
            targets["logit_scale"] = outputs["outputs"]["logit_scale"]
        if 'gt_text_correlation_embedding' in outputs["outputs"].keys():
            targets["gt_text_correlation_embedding"] = outputs["outputs"]['gt_text_correlation_embedding']
        if 'gt_text_correlation_embedding_mask' in outputs["outputs"].keys():
            targets["gt_text_correlation_embedding_mask"] = outputs["outputs"]['gt_text_correlation_embedding_mask']
        if 'weak_box_cate_label' in outputs["outputs"].keys():
            targets["weak_box_cate_label"] = outputs["outputs"]['weak_box_cate_label']
        if 'weak_confidence_weight' in outputs["outputs"].keys():
            targets["weak_confidence_weight"] = outputs["outputs"]['weak_confidence_weight']
        if 'novel_box_judge' in outputs["outputs"].keys():
            targets["novel_box_judge"] = outputs["outputs"]['novel_box_judge']

        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets, if_region_embed=False, if_last_head=True)

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets, if_region_embed=False, if_aux=True, if_last_head=False
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


def build_criterion(args, dataset_config):
    if args.only_image_class:
        print('Attention: only image seen class loss!!!!!!')
        loss_weight_dict = {
            "loss_image_seen_class_weight": args.loss_image_seen_class_weight,
            "loss_no_object_weight": args.loss_no_object_weight,
            "loss_no_object_contrast_weight": args.loss_no_object_contrast_weight
        }
        criterion = SetCriterion(None, dataset_config, loss_weight_dict, train_range_max=args.train_range_max, only_image_class=args.only_image_class)
    elif args.only_prompt_loss:
        print('Attention: only image seen class loss!!!!!!')
        loss_weight_dict = {
            "loss_prompt_softmax_weight": args.loss_prompt_softmax_weight,
            "loss_prompt_sigmoid_weight": args.loss_prompt_sigmoid_weight,
            "loss_no_object_weight": args.loss_no_object_weight,
            "loss_no_object_contrast_weight": args.loss_no_object_contrast_weight
        }
        criterion = SetCriterion(None, dataset_config, loss_weight_dict, train_range_max=args.train_range_max, only_prompt_loss=args.only_prompt_loss)
    else:
        matcher = Matcher(
            cost_class=args.matcher_cls_cost,
            cost_giou=args.matcher_giou_cost,
            cost_center=args.matcher_center_cost,
            cost_objectness=args.matcher_objectness_cost,
        )

        loss_weight_dict = {
            "loss_giou_weight": args.loss_giou_weight,
            "loss_sem_cls_weight": args.loss_sem_cls_weight,
            "loss_sem_cls_softmax_weight": args.loss_sem_cls_softmax_weight,
            "loss_sem_cls_softmax_skip_none_gt_sample_weight": args.loss_sem_cls_softmax_skip_none_gt_sample_weight,
            "loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample_weight": args.loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample_weight,
            "loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness_weight": args.loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness_weight,
            "loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness_weight": args.loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness_weight,
            "loss_sem_cls_softmax_discovery_novel_objectness_weight": args.loss_sem_cls_softmax_discovery_novel_objectness_weight,
            "loss_no_object_weight": args.loss_no_object_weight,
            "loss_angle_cls_weight": args.loss_angle_cls_weight,
            "loss_angle_reg_weight": args.loss_angle_reg_weight,
            "loss_center_weight": args.loss_center_weight,
            "loss_size_weight": args.loss_size_weight,
            "loss_contrastive_weight": args.loss_contrastive_weight,
            "loss_sem_focal_cls_weight": args.loss_sem_focal_cls_weight,
            "loss_contrast_object_text_weight": args.loss_contrast_object_text,
            "loss_region_embed_weight": args.loss_region_embed_weight,
            "loss_predicted_region_embed_l1_weight": args.loss_predicted_region_embed_l1_weight,
            "loss_predicted_region_embed_l1_only_last_layer_weight": args.loss_predicted_region_embed_l1_only_last_layer_weight,
            "loss_predicted_region_embed_cos_weight": args.loss_predicted_region_embed_cos_weight,
            "loss_3d_2d_region_embed_weight": args.loss_3d_2d_region_embed_weight,
            "loss_no_object_contrast_weight": args.loss_no_object_contrast_weight,
            "loss_image_seen_class_weight": args.loss_image_seen_class_weight,
            "loss_batchwise_contrastive_weight": args.loss_batchwise_contrastive_weight,
            "loss_feat_seen_sigmoid_loss_weight": args.loss_feat_seen_sigmoid_loss_weight,
            "loss_feat_seen_softmax_loss_weight": args.loss_feat_seen_softmax_loss_weight,
            "loss_feat_seen_softmax_weakly_loss_weight": args.loss_feat_seen_softmax_weakly_loss_weight,
            "loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi_weight": args.loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi_weight,
            "loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi_weight": args.loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi_weight,
            "loss_feat_seen_softmax_loss_with_novel_cate_confi_weight": args.loss_feat_seen_softmax_loss_with_novel_cate_confi_weight,
            "loss_feat_seen_sigmoid_with_full_image_loss_weight": args.loss_feat_seen_sigmoid_with_full_image_loss_weight,
            "loss_prompt_softmax_weight": args.loss_prompt_softmax_weight,
            "loss_prompt_sigmoid_weight": args.loss_prompt_sigmoid_weight,
        }
        criterion = SetCriterion(matcher, dataset_config, loss_weight_dict, train_range_max=args.train_range_max, args=args)
    return criterion
