# copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import copy
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import scipy.special as scipy_special
import torch

from utils.box_util import (extract_pc_in_box3d, flip_axis_to_camera_np,
                            get_3d_box, get_3d_box_batch)
from utils.eval_det import eval_det_multiprocessing, get_iou_obb, eval_det
from utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from utils.box_util import box3d_iou

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2

# def flip_axis_to_depth_tensor(pc):
#     pc2 = pc.clone()
#     pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
#     pc2[..., 2] *= -1
#     return pc2

def softmax(x):
    """Numpy function for softmax"""
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs



def parse_predictions_obb(
   predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict, center_unnormalized, size_unnormalized, angle_continuous, reset_nms_iou=None
):
    """Parse predictions to OBB parameters and suppress overlapping boxes
    select boxes with (1) NMS (2) higher object score than the threshold

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    #print('-----------------------------')
    #pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    #pred_sem_cls = torch.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob
    #print(predicted_boxes.shape) # shape: [8, 128, 8, 3]
    if reset_nms_iou != None:
        config_dict['nms_iou'] = reset_nms_iou
    angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
    objectness_probs_ = torch.unsqueeze(objectness_probs, -1)
    #print(sem_cls_probs.shape)
    #print(objectness_probs.shape)
    #print(center_unnormalized.shape) # shape: [8, 128, 3]
    predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, sem_cls_probs,  objectness_probs_], dim=-1)
    #print(predicted_boxes_obb.shape)
    #print(size_unnormalized.shape) # shape: [8, 128, 3]
    #print(angle_continuous.shape) # shape: [8, 128]
    angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
    #print(predicted_boxes_obb.shape) #shape: [8, 128, 7]
    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()  # B,num_proposal,10
    obj_prob = objectness_probs.detach().cpu().numpy() # B, num_proposal, the prob of being object of every proposal
    #predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, pred_sem_cls, pred_sem_cls_prob, objectness_probs], dim=-1)
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    pred_sem_cls = np.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy() # B, num_proposal, 8, 3.

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K)) # use one to initialize the box mask

    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # bs, num_points, 3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                # print(torch.max(size_unnormalized[i, j]).item())
                if torch.max(size_unnormalized[i, j]).item()<1e-32: # the boxes with 0-size should be deleted
                    nonempty_box_mask[i, j] = 0
                else:
                    pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1 # if all the boxes have no points, set the one with max object prob to 1
        # -------------------------------------

    if "no_nms" in config_dict and config_dict["no_nms"]:
        # pred_mask = np.ones((bsize, K))
        pred_mask = nonempty_box_mask
    elif not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the min x refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the max x refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the min y refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the max y refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster( # conduct nms for the 2D-box data in every batch
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1 # decide the non-empty box by preded mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster( # conduct nms for 3d boxes in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls( # conduct nms for the box of the same class in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = ( # get some boxes with the higher object scores than the threshold
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict["per_class_proposal"]: # per_class_proposal=True,
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_semcls):
                # print(pred_corners_3d_upright_camera.shape[1])
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        elif config_dict["use_cls_confidence_only"]:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        #print(pred_corners_3d_upright_camera[i, j].shape)
        # print(config_dict["per_class_proposal"])  # True
        # print(config_dict["dataset_config"].num_semcls) # 1140 classes
        # print(len(batch_pred_map_cls)) # 1 batchize
        # print(len(batch_pred_map_cls[0]))  #  num_class * num_obj
        # print(batch_pred_map_cls[0][0][0])
        # print(batch_pred_map_cls[0][0][1].shape)
        # print(batch_pred_map_cls[0][0][2])
        # print(batch_pred_map_cls[0][0][3].shape)
        # print('==========================')

        # for tmp in batch_pred_map_cls:
        #     for tmp2 in tmp:
        #         print(np.min(tmp2[2]))
        #         print(config_dict["conf_thresh"])

    return batch_pred_map_cls


def parse_predictions_obb_nms_then_iou(
   predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict, center_unnormalized, size_unnormalized, angle_continuous, reset_nms_iou=None
):
    """Parse predictions to OBB parameters and suppress overlapping boxes
    select boxes with (1) NMS (2) higher object score than the threshold

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    #print('-----------------------------')
    #pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    #pred_sem_cls = torch.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob
    #print(predicted_boxes.shape) # shape: [8, 128, 8, 3]
    if reset_nms_iou != None:
        config_dict['nms_iou'] = reset_nms_iou
    angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
    objectness_probs_ = torch.unsqueeze(objectness_probs, -1)
    #print(sem_cls_probs.shape)
    #print(objectness_probs.shape)
    #print(center_unnormalized.shape) # shape: [8, 128, 3]
    predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, sem_cls_probs,  objectness_probs_], dim=-1)
    #print(predicted_boxes_obb.shape)
    #print(size_unnormalized.shape) # shape: [8, 128, 3]
    #print(angle_continuous.shape) # shape: [8, 128]
    angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
    #print(predicted_boxes_obb.shape) #shape: [8, 128, 7]
    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()  # B,num_proposal,10
    obj_prob = objectness_probs.detach().cpu().numpy() # B, num_proposal, the prob of being object of every proposal
    #predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, pred_sem_cls, pred_sem_cls_prob, objectness_probs], dim=-1)
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    pred_sem_cls = np.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy() # B, num_proposal, 8, 3.

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K)) # use one to initialize the box mask

    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # bs, num_points, 3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                # print(torch.max(size_unnormalized[i, j]).item())
                if torch.max(size_unnormalized[i, j]).item()<1e-32: # the boxes with 0-size should be deleted
                    nonempty_box_mask[i, j] = 0
                else:
                    pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1 # if all the boxes have no points, set the one with max object prob to 1
        # -------------------------------------

    if "no_nms" in config_dict and config_dict["no_nms"]:
        # pred_mask = np.ones((bsize, K))
        pred_mask = nonempty_box_mask
    elif not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the min x refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the max x refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the min y refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the max y refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster( # conduct nms for the 2D-box data in every batch
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1 # decide the non-empty box by preded mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster( # conduct nms for 3d boxes in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls( # conduct nms for the box of the same class in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = ( # get some boxes with the higher object scores than the threshold
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict["per_class_proposal"]: # per_class_proposal=True,
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_semcls):
                # print(pred_corners_3d_upright_camera.shape[1])
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        elif config_dict["use_cls_confidence_only"]:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        #print(pred_corners_3d_upright_camera[i, j].shape)
        # print(config_dict["per_class_proposal"])  # True
        # print(config_dict["dataset_config"].num_semcls) # 1140 classes
        # print(len(batch_pred_map_cls)) # 1 batchize
        # print(len(batch_pred_map_cls[0]))  #  num_class * num_obj
        # print(batch_pred_map_cls[0][0][0])
        # print(batch_pred_map_cls[0][0][1].shape)
        # print(batch_pred_map_cls[0][0][2])
        # print(batch_pred_map_cls[0][0][3].shape)
        # print('==========================')

        # for tmp in batch_pred_map_cls:
        #     for tmp2 in tmp:
        #         print(np.min(tmp2[2]))
        #         print(config_dict["conf_thresh"])

    return batch_pred_map_cls

def parse_predictions_obb_nms_then_iou_save_seen(
   predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict, center_unnormalized, size_unnormalized, angle_continuous, reset_nms_iou=None
, distillation_feat=None):
    """Parse predictions to OBB parameters and suppress overlapping boxes
    select boxes with (1) NMS (2) higher object score than the threshold

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    #print('-----------------------------')
    #pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    #pred_sem_cls = torch.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob
    #print(predicted_boxes.shape) # shape: [8, 128, 8, 3]
    if reset_nms_iou != None:
        config_dict['nms_iou'] = reset_nms_iou
    angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
    objectness_probs_ = torch.unsqueeze(objectness_probs, -1)
    #print(sem_cls_probs.shape)
    #print(objectness_probs.shape)
    #print(center_unnormalized.shape) # shape: [8, 128, 3]
    predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, sem_cls_probs,  objectness_probs_], dim=-1)
    #print(predicted_boxes_obb.shape)
    #print(size_unnormalized.shape) # shape: [8, 128, 3]
    #print(angle_continuous.shape) # shape: [8, 128]
    angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
    #print(predicted_boxes_obb.shape) #shape: [8, 128, 7]
    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()  # B,num_proposal,10
    obj_prob = objectness_probs.detach().cpu().numpy() # B, num_proposal, the prob of being object of every proposal
    #predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, pred_sem_cls, pred_sem_cls_prob, objectness_probs], dim=-1)
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    pred_sem_cls = np.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy() # B, num_proposal, 8, 3.

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K)) # use one to initialize the box mask

    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # bs, num_points, 3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                # print(torch.max(size_unnormalized[i, j]).item())
                if torch.max(size_unnormalized[i, j]).item()<1e-32: # the boxes with 0-size should be deleted
                    nonempty_box_mask[i, j] = 0
                else:
                    pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1 # if all the boxes have no points, set the one with max object prob to 1
        # -------------------------------------

    if "no_nms" in config_dict and config_dict["no_nms"]:
        # pred_mask = np.ones((bsize, K))
        pred_mask = nonempty_box_mask
    elif not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the min x refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the max x refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the min y refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the max y refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster( # conduct nms for the 2D-box data in every batch
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1 # decide the non-empty box by preded mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster( # conduct nms for 3d boxes in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls( # conduct nms for the box of the same class in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = ( # get some boxes with the higher object scores than the threshold
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict["per_class_proposal"]: # per_class_proposal=True,
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_semcls):
                # print(pred_corners_3d_upright_camera.shape[1])
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
        elif config_dict["use_cls_confidence_only"]:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
                        predicted_boxes_obb[i, j],
                        distillation_feat[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    # and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                        predicted_boxes_obb[i, j]
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        #print(pred_corners_3d_upright_camera[i, j].shape)
        # print(config_dict["per_class_proposal"])  # True
        # print(config_dict["dataset_config"].num_semcls) # 1140 classes
        # print(len(batch_pred_map_cls)) # 1 batchize
        # print(len(batch_pred_map_cls[0]))  #  num_class * num_obj
        # print(batch_pred_map_cls[0][0][0])
        # print(batch_pred_map_cls[0][0][1].shape)
        # print(batch_pred_map_cls[0][0][2])
        # print(batch_pred_map_cls[0][0][3].shape)
        # print('==========================')

        # for tmp in batch_pred_map_cls:
        #     for tmp2 in tmp:
        #         print(np.min(tmp2[2]))
        #         print(config_dict["conf_thresh"])

    return batch_pred_map_cls

# This is exactly the same as VoteNet so that we can compare evaluations.
def parse_predictions(
    predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict
):
    """Parse predictions to OBB parameters and suppress overlapping boxes
    select boxes with (1) NMS (2) higher object score than the threshold

    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """

    sem_cls_probs = sem_cls_probs.detach().cpu().numpy()  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    pred_sem_cls = np.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob
    obj_prob = objectness_probs.detach().cpu().numpy() # B, num_proposal, the prob of being object of every proposal

    pred_corners_3d_upright_camera = predicted_boxes.detach().cpu().numpy() # B, num_proposal, 8, 3.

    K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
    bsize = pred_corners_3d_upright_camera.shape[0]
    nonempty_box_mask = np.ones((bsize, K)) # use one to initialize the box mask
    if config_dict["remove_empty_box"]:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = point_cloud.cpu().numpy()[:, :, 0:3]  # bs, num_points, 3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                if np.max(box3d) < 1e-32 and np.min(box3d) > -1e-32: # all are zeros, then delete it
                    nonempty_box_mask[i, j] = 0
                else:
                    pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                    if len(pc_in_box) < 5:
                        nonempty_box_mask[i, j] = 0
                # if torch.max(size_unnormalized[i, j]).item()<1e-16: # the boxes with 0-size should be deleted
                #     nonempty_box_mask[i, j] = 0
                # else:
                #     pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                #     if len(pc_in_box) < 5:
                #         nonempty_box_mask[i, j] = 0
                # pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                # if len(pc_in_box) < 5:
                #     nonempty_box_mask[i, j] = 0
            if nonempty_box_mask[i].sum() == 0:
                nonempty_box_mask[i, obj_prob[i].argmax()] = 1 # if all the boxes have no points, set the one with max object prob to 1
        # -------------------------------------

    if "no_nms" in config_dict and config_dict["no_nms"]:
        # pred_mask = np.ones((bsize, K))
        pred_mask = nonempty_box_mask
    elif not config_dict["use_3d_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the min x refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 2] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0] # the max x refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the min y refer to the left-bottom corner point of 2D box
                )
                boxes_2d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2] # the max y refer to the right-top corner point of 2D box
                )
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_2d_faster( # conduct nms for the 2D-box data in every batch
                boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1 # decide the non-empty box by preded mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster( # conduct nms for 3d boxes in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 1] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 2] = np.min(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 3] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 0]
                )
                boxes_3d_with_prob[j, 4] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 1]
                )
                boxes_3d_with_prob[j, 5] = np.max(
                    pred_corners_3d_upright_camera[i, j, :, 2]
                )
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[
                    i, j
                ]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            assert len(nonempty_box_inds) > 0
            pick = nms_3d_faster_samecls( # conduct nms for the box of the same class in every batch
                boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                config_dict["nms_iou"],
                config_dict["use_old_type_nms"],
            )
            assert len(pick) > 0
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = ( # get some boxes with the higher object scores than the threshold
        []
    )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict["per_class_proposal"]:
            assert config_dict["use_cls_confidence_only"] is False
            cur_list = []
            for ii in range(config_dict["dataset_config"].num_semcls):
                cur_list += [
                    (
                        ii,
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, ii] * obj_prob[i, j],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            batch_pred_map_cls.append(cur_list)
    # for i in range(bsize):
    #     if config_dict["per_class_proposal"]:
    #         assert config_dict["use_cls_confidence_only"] is False
    #         cur_list = []
    #         for ii in range(config_dict["dataset_config"].num_semcls):
    #             all_boxes_probs_this_im_this_cls = sem_cls_probs[i, :, ii]
    #             sorted_all_boxes_probs_this_im_this_cls = np.sort(all_boxes_probs_this_im_this_cls)
    #             top_k_score_this_im_this_cls = sorted_all_boxes_probs_this_im_this_cls[-128]
    #
    #             cur_list += [
    #                 (
    #                     ii,
    #                     pred_corners_3d_upright_camera[i, j],
    #                     sem_cls_probs[i, j, ii] * obj_prob[i, j],
    #                 )
    #                 for j in range(pred_corners_3d_upright_camera.shape[1])
    #                 if pred_mask[i, j] == 1
    #                 and sem_cls_probs[i, j, ii] >= top_k_score_this_im_this_cls#obj_prob[i, j] > config_dict["conf_thresh"]
    #             ]
    #         batch_pred_map_cls.append(cur_list)

        elif config_dict["use_cls_confidence_only"]:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        else:
            batch_pred_map_cls.append(
                [
                    (
                        pred_sem_cls[i, j].item(),
                        pred_corners_3d_upright_camera[i, j],
                        obj_prob[i, j],
                    )
                    for j in range(pred_corners_3d_upright_camera.shape[1])
                    if pred_mask[i, j] == 1
                    and obj_prob[i, j] > config_dict["conf_thresh"]
                ]
            )
        #print(pred_corners_3d_upright_camera[i, j].shape)
  # print('--------------------------------')
  # print(config_dict["dataset_config"].num_semcls) #1140
  # print(len(batch_pred_map_cls)) # batch size # 28
  # print(len(batch_pred_map_cls[0])) # batch size # 1/2w
  # print(batch_pred_map_cls[0][0][0]) # batch size 0
  # print(batch_pred_map_cls[0][0][1].shape) # batch size 8,3
  # print(batch_pred_map_cls[0][0][2]) # batch size score
    #print(pred_corners_3d_upright_camera.shape) # number queries

    return batch_pred_map_cls


def get_ap_config_dict(
    remove_empty_box=True,
    use_3d_nms=True,
    nms_iou=0.25,
    use_old_type_nms=False,
    cls_nms=True,
    per_class_proposal=True,
    use_cls_confidence_only=False,
    # conf_thresh=0.05,
    conf_thresh=0.05,
    # conf_thresh=0.001, #0.05,
    no_nms=False,
    dataset_config=None,
):
    """
    Default mAP evaluation settings for VoteNet
    """

    config_dict = {
        "remove_empty_box": remove_empty_box,
        "use_3d_nms": use_3d_nms,
        "nms_iou": nms_iou,
        "use_old_type_nms": use_old_type_nms,
        "cls_nms": cls_nms,
        "per_class_proposal": per_class_proposal,
        "use_cls_confidence_only": use_cls_confidence_only,
        "conf_thresh": conf_thresh,
        "no_nms": no_nms,
        "dataset_config": dataset_config,
    }
    return config_dict


class APCalculator(object):
    """Calculating Average Precision"""

    def __init__(
        self,
        dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=None,
        exact_eval=True,
        args=None,
        ap_config_dict=None,
        reset_nms_iou = None
    ):
        """
        Args:
            ap_iou_thresh: List of float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        if ap_config_dict is None:
            ap_config_dict = get_ap_config_dict(
                dataset_config=dataset_config, remove_empty_box=exact_eval
            )
        self.ap_config_dict = ap_config_dict
        self.class2type_map = class2type_map
        self.args = args
        self.dataset_config = dataset_config
        self.reset()
        self.reset_nms_iou = reset_nms_iou

    def make_gt_list_show(self, gt_box_corners, gt_box_sem_cls_labels, gt_box_present, gt_box_centers, gt_box_sizes, gt_box_angles):
        batch_gt_map_cls = []
        bsize = gt_box_corners.shape[0]
        #print(gt_box_centers.shape)
        #print(gt_box_sizes.shape)
        #print(gt_box_angles.shape)
        gt_box_angles_ = gt_box_angles[:,:,np.newaxis]
        #print(gt_box_angles_.shape)
        gt_box_obb = np.concatenate((gt_box_centers, gt_box_sizes, gt_box_angles_), axis=-1)
        #print(gt_box_obb.shape)
        for i in range(bsize):
            batch_gt_map_cls.append(
                [
                    (gt_box_sem_cls_labels[i, j].item(), gt_box_corners[i, j], gt_box_obb[i, j])
                    for j in range(gt_box_corners.shape[1])
                    if gt_box_present[i, j] == 1
                ]
            )
        return batch_gt_map_cls

    def make_gt_list(self, gt_box_corners, gt_box_sem_cls_labels, gt_box_present):
        batch_gt_map_cls = []
        bsize = gt_box_corners.shape[0]
        #print(gt_box_centers.shape)
        #print(gt_box_sizes.shape)
        #print(gt_box_angles.shape)
        #print(gt_box_angles_.shape)
        #print(gt_box_obb.shape)
        for i in range(bsize):
            batch_gt_map_cls.append(
                [
                    (gt_box_sem_cls_labels[i, j].item(), gt_box_corners[i, j])
                    for j in range(gt_box_corners.shape[1])
                    if gt_box_present[i, j] == 1
                ]
            )
        return batch_gt_map_cls


    def step_meter_show(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        return self.step_show(
            predicted_box_corners=outputs["box_corners"],
            sem_cls_probs=outputs["sem_cls_prob"],
            objectness_probs=outputs["objectness_prob"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_sem_cls_labels=targets["gt_box_sem_cls_label"],
            gt_box_present=targets["gt_box_present"],
            center_unnormalized=outputs["center_unnormalized"],
            size_unnormalized=outputs["size_unnormalized"],
            angle_continuous=outputs["angle_continuous"],
            point_cloud_rgb=targets["point_clouds_rgb"],
            gt_box_centers=targets['gt_box_centers'],
            gt_box_sizes=targets['gt_box_sizes'],
            gt_box_angles=targets['gt_box_angles']
        )

    def step_meter_show_nms_then_iou(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        return self.step_show_nms_then_iou(
            predicted_box_corners=outputs["box_corners"],
            sem_cls_probs=outputs["sem_cls_prob"],
            objectness_probs=outputs["objectness_prob"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_sem_cls_labels=targets["gt_box_sem_cls_label"],
            gt_box_present=targets["gt_box_present"],
            center_unnormalized=outputs["center_unnormalized"],
            size_unnormalized=outputs["size_unnormalized"],
            angle_continuous=outputs["angle_continuous"],
            point_cloud_rgb=targets["point_clouds_rgb"],
            gt_box_centers=targets['gt_box_centers'],
            gt_box_sizes=targets['gt_box_sizes'],
            gt_box_angles=targets['gt_box_angles']
        )

    def step_meter_show_nms_then_iou_save_seen(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        return self.step_show_nms_then_iou_save_seen(
            predicted_box_corners=outputs["box_corners"],
            sem_cls_probs=outputs["sem_cls_prob"],
            objectness_probs=outputs["objectness_prob"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_sem_cls_labels=targets["gt_box_sem_cls_label"],
            gt_box_present=targets["gt_box_present"],
            center_unnormalized=outputs["center_unnormalized"],
            size_unnormalized=outputs["size_unnormalized"],
            angle_continuous=outputs["angle_continuous"],
            point_cloud_rgb=targets["point_clouds_rgb"],
            gt_box_centers=targets['gt_box_centers'],
            gt_box_sizes=targets['gt_box_sizes'],
            gt_box_angles=targets['gt_box_angles'],
            distillation_feat=outputs["text_correlation_embedding"]
        )

    def step_show(
        self,
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        point_cloud,
        gt_box_corners,
        gt_box_sem_cls_labels,
        gt_box_present,
        center_unnormalized,
        size_unnormalized,
        angle_continuous,
        point_cloud_rgb,
        gt_box_centers,
        gt_box_sizes,
        gt_box_angles
    ):
        """
        Perform NMS on predicted boxes and threshold them according to score.
        Convert GT boxes
        """
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        gt_box_centers = gt_box_centers.cpu().detach().numpy()
        gt_box_sizes = gt_box_sizes.cpu().detach().numpy()
        gt_box_angles = gt_box_angles.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list_show(
            gt_box_corners, gt_box_sem_cls_labels, gt_box_present, gt_box_centers, gt_box_sizes, gt_box_angles
        )

        #print('-----------------------------')
        #print(predicted_box_corners.shape) #shape: [bs, num_query, 8, 3]
        #print(sem_cls_probs.shape) #shape: [bs, num_query, 10]
        #print(objectness_probs.shape) #shape: [bs, num_query]
        #print(point_cloud.shape) #shape: [bs, num_points, 3]
        #print(gt_box_corners.shape) #shape: [bs, max_obj_num(64), 8, 3]
        #print(gt_box_sem_cls_labels.shape) #shape: [bs, max_obj_num]
        #print(gt_box_present.shape) #shape: [bs, max_obj_num]

        batch_pred_map_cls = parse_predictions_obb(
            predicted_box_corners,
            sem_cls_probs,
            objectness_probs,
            point_cloud,
            self.ap_config_dict,
            center_unnormalized,
            size_unnormalized,
            angle_continuous,
            reset_nms_iou=self.reset_nms_iou
        )

        #print('Here')
        #print(point_cloud_rgb.shape)
        self.accumulate_show(batch_pred_map_cls, batch_gt_map_cls, point_cloud_rgb)
        return batch_pred_map_cls, batch_gt_map_cls, point_cloud_rgb


    def step_show_nms_then_iou(
        self,
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        point_cloud,
        gt_box_corners,
        gt_box_sem_cls_labels,
        gt_box_present,
        center_unnormalized,
        size_unnormalized,
        angle_continuous,
        point_cloud_rgb,
        gt_box_centers,
        gt_box_sizes,
        gt_box_angles
    ):
        """
        Perform NMS on predicted boxes and threshold them according to score.
        Convert GT boxes
        """
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        gt_box_centers = gt_box_centers.cpu().detach().numpy()
        gt_box_sizes = gt_box_sizes.cpu().detach().numpy()
        gt_box_angles = gt_box_angles.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list_show(
            gt_box_corners, gt_box_sem_cls_labels, gt_box_present, gt_box_centers, gt_box_sizes, gt_box_angles
        )

        #print('-----------------------------')
        #print(predicted_box_corners.shape) #shape: [bs, num_query, 8, 3]
        #print(sem_cls_probs.shape) #shape: [bs, num_query, 10]
        #print(objectness_probs.shape) #shape: [bs, num_query]
        #print(point_cloud.shape) #shape: [bs, num_points, 3]
        #print(gt_box_corners.shape) #shape: [bs, max_obj_num(64), 8, 3]
        #print(gt_box_sem_cls_labels.shape) #shape: [bs, max_obj_num]
        #print(gt_box_present.shape) #shape: [bs, max_obj_num]
        tmp_ap_config_dict = copy.deepcopy(self.ap_config_dict)
        tmp_ap_config_dict["per_class_proposal"] = False
        tmp_ap_config_dict['use_cls_confidence_only'] = True
        tmp_ap_config_dict["cls_nms"] = False
        # tmp_ap_config_dict["conf_thresh"] = 0.0
        # tmp_ap_config_dict["nms_iou"] = 0.1
        batch_pred_map_cls = parse_predictions_obb_nms_then_iou(
            predicted_box_corners,
            sem_cls_probs,
            objectness_probs,
            point_cloud,
            tmp_ap_config_dict,
            center_unnormalized,
            size_unnormalized,
            angle_continuous,
            reset_nms_iou=self.reset_nms_iou
        )

        pred_this_batch_update = []
        for batch_idx in range(len(batch_pred_map_cls)):
            pred_this_batch = batch_pred_map_cls[batch_idx]
            gt_this_batch = batch_gt_map_cls[batch_idx]
            pred_this_image = []
            for idx_box in range(len(pred_this_batch)):
                box_tmp = pred_this_batch[idx_box]
                box_corners = box_tmp[1]
                pred_class_id = box_tmp[0]

                objectness_tmp = box_tmp[3][-1]
                assert box_tmp[3][7:-1].shape[0] == 37 or box_tmp[3][7:-1].shape[0] == 1 or box_tmp[3][7:-1].shape[0] == 27 or box_tmp[3][7:-1].shape[0] == 43 or box_tmp[3][7:-1].shape[0] == 232
                max_class_prob, max_class_id = torch.max(box_tmp[3][7:-1], dim=-1)
                max_iou = -1
                for idx_gt_box in range(len(gt_this_batch)):
                    gt_class_id = gt_this_batch[idx_gt_box][0]
                    if gt_class_id > 9: # ignore the novel box
                    # if gt_class_id > 36:
                        continue
                    gt_box_tmp = gt_this_batch[idx_gt_box][1]
                    assert gt_class_id < 10
                    # assert gt_class_id < 37
                    iou_3d, _ = box3d_iou(box_corners, gt_box_tmp)
                    if iou_3d > max_iou:
                        max_iou = iou_3d

                assert pred_class_id==max_class_id.item()
                if max_iou > 0.25:
                    continue
                else:
                    if objectness_tmp > 0.75: # and max_class_prob.item()>0.5:  #and pred_class_id > 9
                        pred_this_image.append(box_tmp) # only keep the iou which is less than 0.5
            pred_this_batch_update.append(pred_this_image)

        #print('Here')
        #print(point_cloud_rgb.shape)
        # self.accumulate_show(pred_this_batch_update, batch_gt_map_cls, point_cloud_rgb)
        return pred_this_batch_update, batch_gt_map_cls, point_cloud_rgb


    def step_show_nms_then_iou_save_seen(
        self,
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        point_cloud,
        gt_box_corners,
        gt_box_sem_cls_labels,
        gt_box_present,
        center_unnormalized,
        size_unnormalized,
        angle_continuous,
        point_cloud_rgb,
        gt_box_centers,
        gt_box_sizes,
        gt_box_angles,
        distillation_feat
    ):
        """
        Perform NMS on predicted boxes and threshold them according to score.
        Convert GT boxes
        """
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        gt_box_centers = gt_box_centers.cpu().detach().numpy()
        gt_box_sizes = gt_box_sizes.cpu().detach().numpy()
        gt_box_angles = gt_box_angles.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list_show(
            gt_box_corners, gt_box_sem_cls_labels, gt_box_present, gt_box_centers, gt_box_sizes, gt_box_angles
        )

        #print('-----------------------------')
        #print(predicted_box_corners.shape) #shape: [bs, num_query, 8, 3]
        #print(sem_cls_probs.shape) #shape: [bs, num_query, 10]
        #print(objectness_probs.shape) #shape: [bs, num_query]
        #print(point_cloud.shape) #shape: [bs, num_points, 3]
        #print(gt_box_corners.shape) #shape: [bs, max_obj_num(64), 8, 3]
        #print(gt_box_sem_cls_labels.shape) #shape: [bs, max_obj_num]
        #print(gt_box_present.shape) #shape: [bs, max_obj_num]
        tmp_ap_config_dict = copy.deepcopy(self.ap_config_dict)
        tmp_ap_config_dict["per_class_proposal"] = False
        tmp_ap_config_dict['use_cls_confidence_only'] = True
        tmp_ap_config_dict["cls_nms"] = False
        batch_pred_map_cls = parse_predictions_obb_nms_then_iou_save_seen(
            predicted_box_corners,
            sem_cls_probs,
            objectness_probs,
            point_cloud,
            tmp_ap_config_dict,
            center_unnormalized,
            size_unnormalized,
            angle_continuous,
            reset_nms_iou=self.reset_nms_iou,
            distillation_feat=distillation_feat
        )

        pred_this_batch_update = []
        for batch_idx in range(len(batch_pred_map_cls)):
            pred_this_batch = batch_pred_map_cls[batch_idx]
            gt_this_batch = batch_gt_map_cls[batch_idx]
            pred_this_image = []
            for idx_box in range(len(pred_this_batch)):
                box_tmp = pred_this_batch[idx_box]
                box_corners = box_tmp[1]
                pred_class_id = box_tmp[0]
                box_feat = box_tmp[4]

                objectness_tmp = box_tmp[3][-1]
                assert box_tmp[3][7:-1].shape[0] == 37 or box_tmp[3][7:-1].shape[0] == 1
                # max_class_prob, max_class_id = torch.max(box_tmp[3][7:-1], dim=-1)

                max_iou = -1

                for idx_gt_box in range(len(gt_this_batch)):
                    gt_class_id = gt_this_batch[idx_gt_box][0]
                    if gt_class_id > 9: # ignore the novel box
                        continue
                    gt_box_tmp = gt_this_batch[idx_gt_box][1]
                    assert gt_class_id < 10
                    iou_3d, _ = box3d_iou(box_corners, gt_box_tmp)
                    if iou_3d > max_iou:
                        max_iou = iou_3d
                        target_gt_class_id = gt_class_id

                # assert pred_class_id==max_class_id.item()
                if max_iou > 0.25:
                    pred_this_image.append([box_tmp, box_feat, target_gt_class_id])
                # else:
                #     if objectness_tmp > 0.5: # and max_class_prob.item()>0.5:  #and pred_class_id > 9
                         # only keep the iou which is less than 0.5
            pred_this_batch_update.append(pred_this_image)

        #print('Here')
        #print(point_cloud_rgb.shape)
        # self.accumulate_show(pred_this_batch_update, batch_gt_map_cls, point_cloud_rgb)
        return pred_this_batch_update, batch_gt_map_cls, point_cloud_rgb

    def step_meter(self, outputs, targets):
        if "outputs" in outputs:
            outputs = outputs["outputs"]
        self.step(
            predicted_box_corners=outputs["box_corners"],
            sem_cls_probs=outputs["sem_cls_prob"],
            objectness_probs=outputs["objectness_prob"],
            point_cloud=targets["point_clouds"],
            gt_box_corners=targets["gt_box_corners"],
            gt_box_sem_cls_labels=targets["gt_box_sem_cls_label"],
            gt_box_present=targets["gt_box_present"],
        )

    def step(
        self,
        predicted_box_corners,
        sem_cls_probs,
        objectness_probs,
        point_cloud,
        gt_box_corners,
        gt_box_sem_cls_labels,
        gt_box_present,
    ):
        """
        Perform NMS on predicted boxes and threshold them according to score.
        Convert GT boxes
        """
        gt_box_corners = gt_box_corners.cpu().detach().numpy()
        gt_box_sem_cls_labels = gt_box_sem_cls_labels.cpu().detach().numpy()
        gt_box_present = gt_box_present.cpu().detach().numpy()
        batch_gt_map_cls = self.make_gt_list(
            gt_box_corners, gt_box_sem_cls_labels, gt_box_present
        )

        #print('-----------------------------')
        #print(predicted_box_corners.shape) #shape: [bs, num_query, 8, 3]
        #print(sem_cls_probs.shape) #shape: [bs, num_query, 10]
        #print(objectness_probs.shape) #shape: [bs, num_query]
        #print(point_cloud.shape) #shape: [bs, num_points, 3]
        #print(gt_box_corners.shape) #shape: [bs, max_obj_num(64), 8, 3]
        #print(gt_box_sem_cls_labels.shape) #shape: [bs, max_obj_num]
        #print(gt_box_present.shape) #shape: [bs, max_obj_num]

        batch_pred_map_cls = parse_predictions(
            predicted_box_corners,
            sem_cls_probs,
            objectness_probs,
            point_cloud,
            self.ap_config_dict,
        )

        self.accumulate(batch_pred_map_cls, batch_gt_map_cls)

    def accumulate(self, batch_pred_map_cls, batch_gt_map_cls):
        """Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
        #    print(len(batch_gt_map_cls))
        #    print(batch_gt_map_cls[i][0][0])
        #    print(batch_gt_map_cls[i][0][1].shape)
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
        #    print(len(batch_pred_map_cls))
        #    print(batch_pred_map_cls[i][0][0])
        #    print(batch_pred_map_cls[i][0][1].shape)
        #    print(batch_pred_map_cls[i][0][2])
        #    print('--------------------------')
            self.scan_cnt += 1

    def accumulate_show(self, batch_pred_map_cls, batch_gt_map_cls, point_clouds):
        """Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        bsize = len(batch_pred_map_cls)
        assert bsize == len(batch_gt_map_cls)
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.point_clouds[self.scan_cnt] = point_clouds[i]
            self.scan_cnt += 1


    def compute_metrics(self):
        """Use accumulated predictions and groundtruths to compute Average Precision."""
        overall_ret = OrderedDict()
        for ap_iou_thresh in self.ap_iou_thresh:
            ret_dict = OrderedDict()
            # rec, prec, ap = eval_det_multiprocessing(
            #     self.pred_map_cls, self.gt_map_cls, ovthresh=ap_iou_thresh
            # )
            rec, prec, ap = eval_det(
               self.pred_map_cls, self.gt_map_cls, ovthresh=ap_iou_thresh
            )
            # print(self.class2type_map)
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                ret_dict["%s Average Precision" % (clsname)] = ap[key]
            ap_vals = np.array(list(ap.values()), dtype=np.float32)
            # print(np.isnan(ap_vals))
            ap_vals[np.isnan(ap_vals)] = 0
            if np.any(np.isnan(ap_vals)):
                print('There is Nan.')
            # print(ap_vals.shape)

            # ret_dict["mAP_fre"] = ap_vals[:4].mean()
            # ret_dict["mAP_common"] = ap_vals[4:37].mean()
            # ret_dict["mAP_base"] = ap_vals[:37].mean()
            # ret_dict["mAP_novel"] = ap_vals[37:].mean()

            # ################setting 2
            # ret_dict["mAP"] = ap_vals.mean()
            # ret_dict["mAP_fre"] = ap_vals[:37].mean()
            # ret_dict["mAP_common"] = ap_vals[37:96].mean()
            # ret_dict["mAP_base"] = ap_vals[:96].mean()
            # ret_dict["mAP_novel"] = ap_vals[96:].mean()
            # ################setting 2

            # # ################revised setting 2
            # if ap_vals.shape[0] > 2:
            #     ret_dict["mAP"] = ap_vals.mean()
            #     ret_dict["mAP_fre"] = ap_vals[:4].mean()
            #     ret_dict["mAP_common"] = ap_vals[4:37].mean()
            #     ret_dict["mAP_base"] = ap_vals[:37].mean()
            #     ret_dict["mAP_novel"] = ap_vals[37:].mean()
            # else:
            #     ret_dict["mAP"] = ap_vals.mean()
            # # ################revised setting 2

            # ################revised setting-2-10classes
            if ap_vals.shape[0] > 2:
                if self.args.dataset_name.find('scannet') == -1 or ap_vals.shape[0] < 21:
                    ret_dict["mAP"] = ap_vals.mean()
                    ret_dict["mAP_fre"] = ap_vals[:4].mean()
                    ret_dict["mAP_common"] = ap_vals[4:10].mean()
                    ret_dict["mAP_base"] = ap_vals[:10].mean()
                    ret_dict["mAP_novel"] = ap_vals[10:].mean()
                else:
                    ret_dict["mAP"] = ap_vals.mean()
                    ret_dict["mAP_fre"] = ap_vals[self.dataset_config.seen_idx_list].mean()
                    ret_dict["mAP_common"] = ap_vals[self.dataset_config.seen_idx_list].mean()
                    ret_dict["mAP_base"] = ap_vals[self.dataset_config.seen_idx_list].mean()
                    ret_dict["mAP_novel"] = ap_vals[self.dataset_config.novel_idx_list].mean()
            else:
                ret_dict["mAP"] = ap_vals.mean()
            # ################revised setting-2-10classes

            ################setting 4
            # ret_dict["mAP_fre"] = ap_vals[:4].mean()
            # ret_dict["mAP_common"] = ap_vals[4:37].mean()
            # ret_dict["mAP_base"] = ap_vals[:37].mean()
            # ret_dict["mAP_novel"] = ap_vals[37:137].mean()
            # ret_dict["mAP"] = ap_vals[:137].mean()
            ################setting 4

            prec_list = []
            # print(ap.keys())
            for key in sorted(prec.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                try:
                    ret_dict["%s Prec" % (clsname)] = prec[key][-1]
                    prec_list.append(prec[key][-1])
                except:
                    ret_dict["%s Prec" % (clsname)] = 0
                    prec_list.append(0)

            rec_list = []
            # print(ap.keys())
            for key in sorted(ap.keys()):
                clsname = self.class2type_map[key] if self.class2type_map else str(key)
                try:
                    ret_dict["%s Recall" % (clsname)] = rec[key][-1]
                    rec_list.append(rec[key][-1])
                except:
                    ret_dict["%s Recall" % (clsname)] = 0
                    rec_list.append(0)

            # ret_dict["AR_fre"] = np.mean(rec_list[:4])
            # ret_dict["AR_common"] = np.mean(rec_list[4:37])
            # ret_dict["AR_base"] = np.mean(rec_list[:37])
            # ret_dict["AR_novel"] = np.mean(rec_list[37:])
            # ################setting 2
            # ret_dict["AR_fre"] = np.mean(rec_list[:37])
            # ret_dict["AR_common"] = np.mean(rec_list[37:96])
            # ret_dict["AR_base"] = np.mean(rec_list[:96])
            # ret_dict["AR_novel"] = np.mean(rec_list[96:])
            # ret_dict["AR"] = np.mean(rec_list)

            # # ################setting 2
            #
            # if ap_vals.shape[0] > 2:
            #     ret_dict["Prec_fre"] = np.mean(prec_list[:4])
            #     ret_dict["Prec_common"] = np.mean(prec_list[4:37])
            #     ret_dict["Prec_base"] = np.mean(prec_list[:37])
            #     ret_dict["Prec_novel"] = np.mean(prec_list[37:])
            #     ret_dict["Prec"] = np.mean(prec_list)
            # else:
            #     ret_dict["Prec"] = np.mean(prec_list)
            #
            # # ################revised setting 2
            # if ap_vals.shape[0] > 2:
            #     ret_dict["AR_fre"] = np.mean(rec_list[:4])
            #     ret_dict["AR_common"] = np.mean(rec_list[4:37])
            #     ret_dict["AR_base"] = np.mean(rec_list[:37])
            #     ret_dict["AR_novel"] = np.mean(rec_list[37:])
            #     ret_dict["AR"] = np.mean(rec_list)
            # else:
            #     ret_dict["AR"] = np.mean(rec_list)
            # # ################setting 2

            # ################setting-2-10classes

            if ap_vals.shape[0] > 2:
                if self.args.dataset_name.find('scannet') == -1 or ap_vals.shape[0] < 21:
                    ret_dict["Prec_fre"] = np.mean(prec_list[:4])
                    ret_dict["Prec_common"] = np.mean(prec_list[4:10])
                    ret_dict["Prec_base"] = np.mean(prec_list[:10])
                    ret_dict["Prec_novel"] = np.mean(prec_list[10:])
                    ret_dict["Prec"] = np.mean(prec_list)
                else:
                    ret_dict["Prec_fre"] = np.mean(np.array(prec_list)[self.dataset_config.seen_idx_list])
                    ret_dict["Prec_common"] = np.mean(np.array(prec_list)[self.dataset_config.seen_idx_list])
                    ret_dict["Prec_base"] = np.mean(np.array(prec_list)[self.dataset_config.seen_idx_list])
                    ret_dict["Prec_novel"] = np.mean(np.array(prec_list)[self.dataset_config.novel_idx_list])
                    ret_dict["Prec"] = np.mean(prec_list)
            else:
                ret_dict["Prec"] = np.mean(prec_list)

            # ################revised setting-2-10classes
            if ap_vals.shape[0] > 2:
                if self.args.dataset_name.find('scannet') == -1 or ap_vals.shape[0] < 21:
                    ret_dict["AR_fre"] = np.mean(rec_list[:4])
                    ret_dict["AR_common"] = np.mean(rec_list[4:10])
                    ret_dict["AR_base"] = np.mean(rec_list[:10])
                    ret_dict["AR_novel"] = np.mean(rec_list[10:])
                    ret_dict["AR"] = np.mean(rec_list)
                else:
                    ret_dict["AR_fre"] = np.mean(np.array(rec_list)[self.dataset_config.seen_idx_list])
                    ret_dict["AR_common"] = np.mean(np.array(rec_list)[self.dataset_config.seen_idx_list])
                    ret_dict["AR_base"] = np.mean(np.array(rec_list)[self.dataset_config.seen_idx_list])
                    ret_dict["AR_novel"] = np.mean(np.array(rec_list)[self.dataset_config.novel_idx_list])
                    ret_dict["AR"] = np.mean(rec_list)
            else:
                ret_dict["AR"] = np.mean(rec_list)
            # ################setting-2-10classes


            ################setting 4
            # ret_dict["AR_fre"] = np.mean(rec_list[:4])
            # ret_dict["AR_common"] = np.mean(rec_list[4:37])
            # ret_dict["AR_base"] = np.mean(rec_list[:37])
            # ret_dict["AR_novel"] = np.mean(rec_list[37:137])
            # ret_dict["AR"] = np.mean(rec_list[:137])
            ################setting 4
            overall_ret[ap_iou_thresh] = ret_dict
        return overall_ret

    def __str__(self):
        overall_ret = self.compute_metrics()
        return self.metrics_to_str(overall_ret)

    def metrics_to_str(self, overall_ret, per_class=True):
        mAP_strs = []
        AR_strs = []
        prec_strs = []
        per_class_metrics = []
        for ap_iou_thresh in self.ap_iou_thresh:
            mAP = overall_ret[ap_iou_thresh]["mAP"] * 100
            # print(f"1mAP{ap_iou_thresh:.2f}: {mAP:.2f}\n")
            # print(f"2mAP{ap_iou_thresh}: {mAP:.2f}\n")
            # print(f"3mAP{ap_iou_thresh:.2f}:"+f" {mAP:.2f}\n")
            mAP_strs.append(f"mAP{ap_iou_thresh:.2f}: {mAP:.2f}\n")
            if "mAP_fre" in overall_ret[ap_iou_thresh].keys():
                mAP_fre = overall_ret[ap_iou_thresh]["mAP_fre"] * 100
                mAP_strs.append(f"mAP_fre{ap_iou_thresh:.2f}: {mAP_fre:.2f}\n")
                mAP_common = overall_ret[ap_iou_thresh]["mAP_common"] * 100
                mAP_strs.append(f"mAP_common{ap_iou_thresh:.2f}: {mAP_common:.2f}\n")
                mAP_base = overall_ret[ap_iou_thresh]["mAP_base"] * 100
                mAP_strs.append(f"mAP_base{ap_iou_thresh:.2f}: {mAP_base:.2f}\n")
                mAP_novel = overall_ret[ap_iou_thresh]["mAP_novel"] * 100
                mAP_strs.append(f"mAP_novel{ap_iou_thresh:.2f}: {mAP_novel:.2f}\n\n")
            # print(mAP_strs)
            ar = overall_ret[ap_iou_thresh]["AR"] * 100
            AR_strs.append(f"AR{ap_iou_thresh:.2f}: {ar:.2f}\n")
            if "AR_fre" in overall_ret[ap_iou_thresh].keys():
                AR_fre = overall_ret[ap_iou_thresh]["AR_fre"] * 100
                AR_strs.append(f"AR_fre{ap_iou_thresh:.2f}: {AR_fre:.2f}\n")
                AR_common = overall_ret[ap_iou_thresh]["AR_common"] * 100
                AR_strs.append(f"AR_common{ap_iou_thresh:.2f}: {AR_common:.2f}\n")
                AR_base = overall_ret[ap_iou_thresh]["AR_base"] * 100
                AR_strs.append(f"AR_base{ap_iou_thresh:.2f}: {AR_base:.2f}\n")
                AR_novel = overall_ret[ap_iou_thresh]["AR_novel"] * 100
                AR_strs.append(f"AR_novel{ap_iou_thresh:.2f}: {AR_novel:.2f}\n\n")

            prec = overall_ret[ap_iou_thresh]["Prec"] * 100
            prec_strs.append(f"Prec{ap_iou_thresh:.2f}: {prec:.2f}\n")
            if "Prec_fre" in overall_ret[ap_iou_thresh].keys():
                Prec_fre = overall_ret[ap_iou_thresh]["Prec_fre"] * 100
                prec_strs.append(f"Prec_fre{ap_iou_thresh:.2f}: {Prec_fre:.2f}\n")
                Prec_common = overall_ret[ap_iou_thresh]["Prec_common"] * 100
                prec_strs.append(f"Prec_common{ap_iou_thresh:.2f}: {Prec_common:.2f}\n")
                Prec_base = overall_ret[ap_iou_thresh]["Prec_base"] * 100
                prec_strs.append(f"Prec_base{ap_iou_thresh:.2f}: {Prec_base:.2f}\n")
                Prec_novel = overall_ret[ap_iou_thresh]["Prec_novel"] * 100
                prec_strs.append(f"Prec_novel{ap_iou_thresh:.2f}: {Prec_novel:.2f}\n\n")


            if per_class:
                # per-class metrics
                per_class_metrics.append("-" * 5)
                per_class_metrics.append(f"IOU Thresh={ap_iou_thresh}")
                # if self.args.dataset_name.find('scannet') == -1:
                for x in list(overall_ret[ap_iou_thresh].keys()):
                    if x == "mAP" or x == "AR" or x[-3:]=='fre' or x[-6:]=='common' or x[-4:]=='base' or x[-5:] == 'novel':
                        pass
                    else:
                        met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                        per_class_metrics.append(met_str)
                # else:
                #     if x == "mAP" or x == "AR" or x[-3:]=='fre' or x[-6:]=='common' or x[-4:]=='base' or x[-5:] == 'novel':
                #         pass
                #     else:
                #         met_str = f"{x}: {overall_ret[ap_iou_thresh][x]*100:.2f}"
                #         per_class_metrics.append(met_str)
            # break
        # ap_header = [f"mAP{x:.2f}" for x in self.ap_iou_thresh]
        # ap_str = ", ".join(ap_header)
        ap_str = ''
        ap_str += ''.join(mAP_strs)
        ap_str += "\n"

        # ar_header = [f"AR{x:.2f}" for x in self.ap_iou_thresh]
        # ap_str += ", ".join(ar_header)
        ap_str += ''.join(AR_strs)
        ap_str += "\n"

        ap_str += ''.join(prec_strs)
        ap_str += "\n"
        # ap_str += ": " + ", ".join(AR_strs)

        if per_class:
            per_class_metrics = "\n".join(per_class_metrics)
            ap_str += "\n"
            ap_str += per_class_metrics

        return ap_str

    def metrics_to_dict(self, overall_ret):
        metrics_dict = {}
        for ap_iou_thresh in self.ap_iou_thresh:
            metrics_dict[f"mAP_{ap_iou_thresh}"] = (
                overall_ret[ap_iou_thresh]["mAP"] * 100
            )
            metrics_dict[f"AR_{ap_iou_thresh}"] = overall_ret[ap_iou_thresh]["AR"] * 100
        return metrics_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.point_clouds = {}
        self.scan_cnt = 0



# def parse_predictions_obb_nms_then_iou_tensor(
#    predicted_boxes, sem_cls_probs, objectness_probs, point_cloud, config_dict, center_unnormalized, size_unnormalized, angle_continuous, reset_nms_iou=None
# ):
#     """Parse predictions to OBB parameters and suppress overlapping boxes
#     select boxes with (1) NMS (2) higher object score than the threshold
#
#     Args:
#         end_points: dict
#             {point_clouds, center, heading_scores, heading_residuals,
#             size_scores, size_residuals, sem_cls_scores}
#         config_dict: dict
#             {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
#             use_old_type_nms, conf_thresh, per_class_proposal}
#
#     Returns:
#         batch_pred_map_cls: a list of len == batch size (BS)
#             [pred_list_i], i = 0, 1, ..., BS-1
#             where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
#             where j = 0, ..., num of valid detections - 1 from sample input i
#     """
#
#     #print('-----------------------------')
#     #pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
#     #pred_sem_cls = torch.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob
#     #print(predicted_boxes.shape) # shape: [8, 128, 8, 3]
#     if reset_nms_iou != None:
#         config_dict['nms_iou'] = reset_nms_iou
#     angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
#     objectness_probs_ = torch.unsqueeze(objectness_probs, -1)
#     #print(sem_cls_probs.shape)
#     #print(objectness_probs.shape)
#     #print(center_unnormalized.shape) # shape: [8, 128, 3]
#     predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, sem_cls_probs,  objectness_probs_], dim=-1)
#     #print(predicted_boxes_obb.shape)
#     #print(size_unnormalized.shape) # shape: [8, 128, 3]
#     #print(angle_continuous.shape) # shape: [8, 128]
#     angle_continuous_ = torch.unsqueeze(angle_continuous, -1)
#     #print(predicted_boxes_obb.shape) #shape: [8, 128, 7]
#     sem_cls_probs = sem_cls_probs #.detach().cpu().numpy()  # B,num_proposal,10
#     obj_prob = objectness_probs #.detach().cpu().numpy() # B, num_proposal, the prob of being object of every proposal
#     #predicted_boxes_obb = torch.cat([center_unnormalized, size_unnormalized, angle_continuous_, pred_sem_cls, pred_sem_cls_prob, objectness_probs], dim=-1)
#     pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
#      # = np.argmax(sem_cls_probs, -1) # B, num_proposal. the object class(idx) of the max prob
#
#     pred_corners_3d_upright_camera = predicted_boxes #.detach().cpu().numpy() # B, num_proposal, 8, 3.
#
#     K = pred_corners_3d_upright_camera.shape[1]  # K==num_proposal
#     bsize = pred_corners_3d_upright_camera.shape[0]
#     # nonempty_box_mask = np.ones((bsize, K)) # use one to initialize the box mask
#     nonempty_box_mask = torch.ones((bsize, K), device="cuda")
#
#     if config_dict["remove_empty_box"]:
#         # -------------------------------------
#         # Remove predicted boxes without any point within them..
#         batch_pc = point_cloud[:, :, 0:3]  # bs, num_points, 3
#         for i in range(bsize):
#             pc = batch_pc[i, :, :]  # (N,3)
#             for j in range(K):
#                 box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
#                 box3d = flip_axis_to_depth_tensor(box3d)
#                 # print(torch.max(size_unnormalized[i, j]).item())
#                 if torch.max(size_unnormalized[i, j]).item()<1e-32: # the boxes with 0-size should be deleted
#                     nonempty_box_mask[i, j] = 0
#                 else:
#                     pc_in_box, inds = extract_pc_in_box3d(pc.cpu().numpy(), box3d.cpu().numpy())
#                     if len(pc_in_box) < 5:
#                         nonempty_box_mask[i, j] = 0
#             if nonempty_box_mask[i].sum() == 0:
#                 nonempty_box_mask[i, obj_prob[i].argmax()] = 1 # if all the boxes have no points, set the one with max object prob to 1
#         # -------------------------------------
#
#     # if "no_nms" in config_dict and config_dict["no_nms"]:
#     #     # pred_mask = np.ones((bsize, K))
#     #     pred_mask = nonempty_box_mask
#     # elif not config_dict["use_3d_nms"]:
#     #     # ---------- NMS input: pred_with_prob in (B,K,7) -----------
#     #     pred_mask = np.zeros((bsize, K))
#     #     for i in range(bsize):
#     #         boxes_2d_with_prob = np.zeros((K, 5))
#     #         for j in range(K):
#     #             boxes_2d_with_prob[j, 0] = np.min(
#     #                 pred_corners_3d_upright_camera[i, j, :, 0] # the min x refer to the left-bottom corner point of 2D box
#     #             )
#     #             boxes_2d_with_prob[j, 2] = np.max(
#     #                 pred_corners_3d_upright_camera[i, j, :, 0] # the max x refer to the right-top corner point of 2D box
#     #             )
#     #             boxes_2d_with_prob[j, 1] = np.min(
#     #                 pred_corners_3d_upright_camera[i, j, :, 2] # the min y refer to the left-bottom corner point of 2D box
#     #             )
#     #             boxes_2d_with_prob[j, 3] = np.max(
#     #                 pred_corners_3d_upright_camera[i, j, :, 2] # the max y refer to the right-top corner point of 2D box
#     #             )
#     #             boxes_2d_with_prob[j, 4] = obj_prob[i, j]
#     #         nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
#     #         assert len(nonempty_box_inds) > 0
#     #         pick = nms_2d_faster( # conduct nms for the 2D-box data in every batch
#     #             boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
#     #             config_dict["nms_iou"],
#     #             config_dict["use_old_type_nms"],
#     #         )
#     #         assert len(pick) > 0
#     #         pred_mask[i, nonempty_box_inds[pick]] = 1 # decide the non-empty box by preded mask
#         # ---------- NMS output: pred_mask in (B,K) -----------
#     # elif config_dict["use_3d_nms"] and (not config_dict["cls_nms"]):
#         # ---------- NMS input: pred_with_prob in (B,K,7) -----------
#     pred_mask = torch.zeros((bsize, K), device='cuda')
#     for i in range(bsize):
#         boxes_3d_with_prob = torch.zeros((K, 7), device='cuda')
#         for j in range(K):
#             boxes_3d_with_prob[j, 0] = torch.min(
#                 pred_corners_3d_upright_camera[i, j, :, 0]
#             )
#             boxes_3d_with_prob[j, 1] = torch.min(
#                 pred_corners_3d_upright_camera[i, j, :, 1]
#             )
#             boxes_3d_with_prob[j, 2] = torch.min(
#                 pred_corners_3d_upright_camera[i, j, :, 2]
#             )
#             boxes_3d_with_prob[j, 3] = torch.max(
#                 pred_corners_3d_upright_camera[i, j, :, 0]
#             )
#             boxes_3d_with_prob[j, 4] = torch.max(
#                 pred_corners_3d_upright_camera[i, j, :, 1]
#             )
#             boxes_3d_with_prob[j, 5] = torch.max(
#                 pred_corners_3d_upright_camera[i, j, :, 2]
#             )
#             boxes_3d_with_prob[j, 6] = obj_prob[i, j]
#         nonempty_box_inds = torch.nonzero(nonempty_box_mask[i, :] == 1) #[0]
#         assert len(nonempty_box_inds) > 0
#         pick = nms_3d_faster( # conduct nms for 3d boxes in every batch
#             boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
#             config_dict["nms_iou"],
#             config_dict["use_old_type_nms"],
#         )
#         assert len(pick) > 0
#         pred_mask[i, nonempty_box_inds[pick]] = 1
#         # ---------- NMS output: pred_mask in (B,K) -----------
#     # elif config_dict["use_3d_nms"] and config_dict["cls_nms"]:
#     #     # ---------- NMS input: pred_with_prob in (B,K,8) -----------
#     #     pred_mask = np.zeros((bsize, K))
#     #     for i in range(bsize):
#     #         boxes_3d_with_prob = np.zeros((K, 8))
#     #         for j in range(K):
#     #             boxes_3d_with_prob[j, 0] = np.min(
#     #                 pred_corners_3d_upright_camera[i, j, :, 0]
#     #             )
#     #             boxes_3d_with_prob[j, 1] = np.min(
#     #                 pred_corners_3d_upright_camera[i, j, :, 1]
#     #             )
#     #             boxes_3d_with_prob[j, 2] = np.min(
#     #                 pred_corners_3d_upright_camera[i, j, :, 2]
#     #             )
#     #             boxes_3d_with_prob[j, 3] = np.max(
#     #                 pred_corners_3d_upright_camera[i, j, :, 0]
#     #             )
#     #             boxes_3d_with_prob[j, 4] = np.max(
#     #                 pred_corners_3d_upright_camera[i, j, :, 1]
#     #             )
#     #             boxes_3d_with_prob[j, 5] = np.max(
#     #                 pred_corners_3d_upright_camera[i, j, :, 2]
#     #             )
#     #             boxes_3d_with_prob[j, 6] = obj_prob[i, j]
#     #             boxes_3d_with_prob[j, 7] = pred_sem_cls[
#     #                 i, j
#     #             ]  # only suppress if the two boxes are of the same class!!
#     #         nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
#     #         assert len(nonempty_box_inds) > 0
#     #         pick = nms_3d_faster_samecls( # conduct nms for the box of the same class in every batch
#     #             boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
#     #             config_dict["nms_iou"],
#     #             config_dict["use_old_type_nms"],
#     #         )
#     #         assert len(pick) > 0
#     #         pred_mask[i, nonempty_box_inds[pick]] = 1
#         # ---------- NMS output: pred_mask in (B,K) -----------
#
#     # batch_pred_map_cls = ( # get some boxes with the higher object scores than the threshold
#     #     []
#     # )  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
#     # for i in range(bsize):
#     #     if config_dict["per_class_proposal"]: # per_class_proposal=True,
#     #         assert config_dict["use_cls_confidence_only"] is False
#     #         cur_list = []
#     #         for ii in range(config_dict["dataset_config"].num_semcls):
#     #             # print(pred_corners_3d_upright_camera.shape[1])
#     #             cur_list += [
#     #                 (
#     #                     ii,
#     #                     pred_corners_3d_upright_camera[i, j],
#     #                     sem_cls_probs[i, j, ii] * obj_prob[i, j],
#     #                     predicted_boxes_obb[i, j]
#     #                 )
#     #                 for j in range(pred_corners_3d_upright_camera.shape[1])
#     #                 if pred_mask[i, j] == 1
#     #                 and obj_prob[i, j] > config_dict["conf_thresh"]
#     #             ]
#     #         batch_pred_map_cls.append(cur_list)
#     #     elif config_dict["use_cls_confidence_only"]:
#     #         batch_pred_map_cls.append(
#     #             [
#     #                 (
#     #                     pred_sem_cls[i, j].item(),
#     #                     pred_corners_3d_upright_camera[i, j],
#     #                     sem_cls_probs[i, j, pred_sem_cls[i, j].item()],
#     #                     predicted_boxes_obb[i, j]
#     #                 )
#     #                 for j in range(pred_corners_3d_upright_camera.shape[1])
#     #                 if pred_mask[i, j] == 1
#     #                 and obj_prob[i, j] > config_dict["conf_thresh"]
#     #             ]
#     #         )
#     #     else:
#     #         batch_pred_map_cls.append(
#     #             [
#     #                 (
#     #                     pred_sem_cls[i, j].item(),
#     #                     pred_corners_3d_upright_camera[i, j],
#     #                     obj_prob[i, j],
#     #                     predicted_boxes_obb[i, j]
#     #                 )
#     #                 for j in range(pred_corners_3d_upright_camera.shape[1])
#     #                 if pred_mask[i, j] == 1
#     #                 and obj_prob[i, j] > config_dict["conf_thresh"]
#     #             ]
#     #         )
#         #print(pred_corners_3d_upright_camera[i, j].shape)
#         # print(config_dict["per_class_proposal"])  # True
#         # print(config_dict["dataset_config"].num_semcls) # 1140 classes
#         # print(len(batch_pred_map_cls)) # 1 batchize
#         # print(len(batch_pred_map_cls[0]))  #  num_class * num_obj
#         # print(batch_pred_map_cls[0][0][0])
#         # print(batch_pred_map_cls[0][0][1].shape)
#         # print(batch_pred_map_cls[0][0][2])
#         # print(batch_pred_map_cls[0][0][3].shape)
#         # print('==========================')
#
#         # for tmp in batch_pred_map_cls:
#         #     for tmp2 in tmp:
#         #         print(np.min(tmp2[2]))
#         #         print(config_dict["conf_thresh"])
#
#     return batch_pred_map_cls