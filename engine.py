# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import random
import torch
import numpy as np
import datetime
import logging
import math
import time
import sys
import os
import json
from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)
from utils.box_util import extract_pc_in_box3d
import cv2
import datasets.sunrgbd_utils as sunrgbd_utils
import datasets.scannet_utils as scannet_utils
from datasets.sunrgbd_image import SunrgbdImageDatasetConfig as SunrgbdDatasetConfig
from utils.votenet_pc_util import write_oriented_bbox, write_ply, write_ply_rgb
from datasets.scannet_image import  ScannetImageDatasetConfig
# from vis_color_pc import read_pc_data
sunrgbd_v1_dataconfig = SunrgbdDatasetConfig(if_print=False, use_v1=True)
scannet_dataconfig = ScannetImageDatasetConfig(if_print=False)
def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):

    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )


    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    if hasattr(model, 'module'):
        if hasattr(model.module, 'clip_model'):
            # if not args.if_clip_trainable:
            model.module.clip_model.eval()
            # else:
            if not args.if_clip_trainable:
                if hasattr(model.module, 'res_encoder'):
                    model.module.res_encoder.eval()
            else:
                if hasattr(model.module, 'res_encoder'):
                    model.module.res_encoder.train()
                print('CLIP-image is trainable!!!!!!!!!!!!')
            #     model.module.clip_model.visual.train()
            #     print('CLIP-image is trainable!!!!!!!!!!!!')
        else:
            print('no clip here')
    else:
        if hasattr(model, 'clip_model'):
            # if not args.if_clip_trainable:
            model.clip_model.eval()
            if not args.if_clip_trainable:
                if hasattr(model, 'res_encoder'):
                    model.res_encoder.eval()
            else:
                if hasattr(model, 'res_encoder'):
                    model.res_encoder.train()
                print('CLIP-image is trainable!!!!!!!!!!!!')
            # else:
            #     model.clip_model.eval()
            #     model.clip_model.visual.train()
            #     print('CLIP-image is trainable!!!!!!!!!!!!')
        else:
            print('no clip here')


    for batch_idx, batch_data_label in enumerate(dataset_loader):
        # print(torch.version.cuda)
        # print(torch.backends.cudnn.version())
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]

        # Forward pass
        # print(model.bg_embed)
        # print(model.test_prob)
        # print(model.bg_embed.grad)
        # print(model.pre_encoder.mlp_module.layer0.conv.weight.grad)
        optimizer.zero_grad()

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        # }
        # batch_data_label_ori = copy.deepcopy(batch_data_label)
        outputs = model(batch_data_label, curr_epoch=curr_epoch)

        ####### here we can compare the version
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)

        #print([batch_data_label['num_boxes_replica']])

        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()

        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()

        if args.only_image_class:
            if is_primary():
                loss_image_seen_class = loss_dict_reduced["loss_image_seen_class"]
                time_delta.update(time.time() - curr_time)
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; loss_image_seen_class {loss_image_seen_class:0.6f} ; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
                logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")
            curr_iter += 1
            continue

        if args.only_prompt_loss:
            if is_primary():
                if "loss_prompt_softmax" in loss_dict_reduced.keys():
                    loss_prompt_softmax = loss_dict_reduced["loss_prompt_softmax"]
                else:
                    loss_prompt_softmax = -1.0
                if "loss_prompt_sigmoid" in loss_dict_reduced.keys():
                    loss_prompt_sigmoid = loss_dict_reduced["loss_prompt_sigmoid"]
                else:
                    loss_prompt_sigmoid = -1.0
                time_delta.update(time.time() - curr_time)
                mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                eta_seconds = (max_iters - curr_iter) * time_delta.avg
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; loss_prompt_softmax {loss_prompt_softmax:0.6f} ; loss_prompt_sigmoid {loss_prompt_sigmoid:0.6f} ; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
                )
                logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")
            curr_iter += 1
            continue

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            if "loss_region_embed" in loss_dict_reduced.keys():
                loss_reg_embed = loss_dict_reduced["loss_region_embed"]
            else:
                loss_reg_embed = -1.0
            if "loss_sem_cls" not in loss_dict_reduced.keys():
                loss_sem = -1.0
            else:
                loss_sem = loss_dict_reduced["loss_sem_cls"]
            if "loss_contrast_object_text" not in loss_dict_reduced.keys():
                loss_contrast_object_text = -1.0
            else:
                loss_contrast_object_text = loss_dict_reduced["loss_contrast_object_text"]
            if "loss_predicted_region_embed_cos" not in loss_dict_reduced.keys():
                loss_predicted_region_embed_cos = -1.0
            else:
                loss_predicted_region_embed_cos = loss_dict_reduced["loss_predicted_region_embed_cos"]
            if "loss_predicted_region_embed_l1" not in loss_dict_reduced.keys():
                loss_predicted_region_embed_l1 = -1.0
            else:
                loss_predicted_region_embed_l1 = loss_dict_reduced["loss_predicted_region_embed_l1"]
            if "loss_predicted_region_embed_l1_only_last_layer" not in loss_dict_reduced.keys():
                loss_predicted_region_embed_l1_only_last_layer = -1.0
            else:
                loss_predicted_region_embed_l1_only_last_layer = loss_dict_reduced["loss_predicted_region_embed_l1_only_last_layer"]
            if "loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness" not in loss_dict_reduced.keys():
                loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness = -1.0
            else:
                loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness = loss_dict_reduced['loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness']
            if "loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness" not in loss_dict_reduced.keys():
                loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness = -1.0
            else:
                loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness = loss_dict_reduced['loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness']



            loss_angle_cls = loss_dict_reduced["loss_angle_cls"]
            loss_angle_reg = loss_dict_reduced["loss_angle_reg"]
            loss_cardinality = loss_dict_reduced["loss_cardinality"]
            loss_center = loss_dict_reduced["loss_center"]
            loss_size = loss_dict_reduced["loss_size"]
            # loss_contrast_object_text = loss_dict_reduced["loss_contrast_object_text"]
            # if "loss_sem_focal_cls" in loss_dict_reduced.keys():
            #     loss_sem_focal = loss_dict_reduced["loss_sem_focal_cls"]
            # else:
            #     loss_sem_focal = -1
            if "loss_contrastive" in loss_dict_reduced.keys():
                loss_contrastive = loss_dict_reduced["loss_contrastive"]
            else:
                loss_contrastive = -1.0
            if "loss_sem_focal_cls" in loss_dict_reduced.keys():
                loss_sem_focal_cls = loss_dict_reduced["loss_sem_focal_cls"]
            else:
                loss_sem_focal_cls = -1.0

            if "loss_contrast_3dto2d_text" in loss_dict_reduced.keys():
                loss_contrast_3dto2d_text = loss_dict_reduced["loss_contrast_3dto2d_text"]
            else:
                loss_contrast_3dto2d_text = -1.0

            if "loss_batchwise_contrastive" in loss_dict_reduced.keys():
                loss_batchwise_contrastive = loss_dict_reduced["loss_batchwise_contrastive"]
            else:
                loss_batchwise_contrastive = -1.0

            if "loss_image_seen_class" in loss_dict_reduced.keys():
                loss_image_seen_class = loss_dict_reduced["loss_image_seen_class"]
            else:
                loss_image_seen_class = -1.0
            if "loss_3d_2d_region_embed" in loss_dict_reduced.keys():
                loss_3d_2d_region_embed = loss_dict_reduced["loss_3d_2d_region_embed"]
            else:
                loss_3d_2d_region_embed = -1.0

            if "loss_sem_cls_softmax" in loss_dict_reduced.keys():
                loss_sem_cls_softmax = loss_dict_reduced["loss_sem_cls_softmax"]
            else:
                loss_sem_cls_softmax = -1.0

            if "loss_sem_cls_softmax_skip_none_gt_sample" in loss_dict_reduced.keys():
                loss_sem_cls_softmax_skip_none_gt_sample = loss_dict_reduced["loss_sem_cls_softmax_skip_none_gt_sample"]
            else:
                loss_sem_cls_softmax_skip_none_gt_sample = -1.0

            if "loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample" in loss_dict_reduced.keys():
                loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample = loss_dict_reduced["loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample"]
            else:
                loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample = -1.0

            if "loss_sem_cls_softmax_discovery_novel_objectness" in loss_dict_reduced.keys():
                loss_sem_cls_softmax_discovery_novel_objectness = loss_dict_reduced["loss_sem_cls_softmax_discovery_novel_objectness"]
            else:
                loss_sem_cls_softmax_discovery_novel_objectness = -1.0

            if "loss_feat_seen_sigmoid_loss" not in loss_dict_reduced.keys():
                loss_feat_seen_sigmoid_loss = -1.0
            else:
                loss_feat_seen_sigmoid_loss = loss_dict_reduced["loss_feat_seen_sigmoid_loss"]

            if "loss_feat_seen_softmax_loss" not in loss_dict_reduced.keys():
                loss_feat_seen_softmax_loss = -1.0
            else:
                loss_feat_seen_softmax_loss = loss_dict_reduced["loss_feat_seen_softmax_loss"]

            if "loss_feat_seen_softmax_weakly_loss" not in loss_dict_reduced.keys():
                loss_feat_seen_softmax_weakly_loss = -1.0
            else:
                loss_feat_seen_softmax_weakly_loss = loss_dict_reduced["loss_feat_seen_softmax_weakly_loss"]

            if "loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi" not in loss_dict_reduced.keys():
                loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi = -1.0
            else:
                loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi = loss_dict_reduced["loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi"]

            if "loss_feat_seen_sigmoid_with_full_image_loss" not in loss_dict_reduced.keys():
                loss_feat_seen_sigmoid_with_full_image_loss = -1.0
            else:
                loss_feat_seen_sigmoid_with_full_image_loss = loss_dict_reduced["loss_feat_seen_sigmoid_with_full_image_loss"]
            if "loss_feat_seen_softmax_loss_with_novel_cate_confi" not in loss_dict_reduced.keys():
                loss_feat_seen_softmax_loss_with_novel_cate_confi = -1.0
            else:
                loss_feat_seen_softmax_loss_with_novel_cate_confi = loss_dict_reduced['loss_feat_seen_softmax_loss_with_novel_cate_confi']
            if "loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi" not in loss_dict_reduced.keys():
                loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi = -1.0
            else:
                loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi = loss_dict_reduced['loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi']


            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.6f}; loss_feat_seen_softmax_iou_match_weakly_loss {loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi:0.6f}; loss_feat_seen_softmax_loss_with_novel_cate_confi {loss_feat_seen_softmax_loss_with_novel_cate_confi:0.6f}; loss_sem_cls_softmax_2d_box_iou_supervised {loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample:0.6f}; loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi {loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi:0.6f}; loss_feat_seen_softmax_weakly_loss {loss_feat_seen_softmax_weakly_loss:0.6f} loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness {loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness:0.6f} loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness {loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness:0.6f} loss_feat_seen_sigmoid_with_full_image_loss {loss_feat_seen_sigmoid_with_full_image_loss:0.6f} loss_feat_seen_softmax_loss {loss_feat_seen_softmax_loss:0.6f} loss_feat_seen_sigmoid_loss {loss_feat_seen_sigmoid_loss:0.6f} loss_predicted_region_embed_l1_only_last_layer {loss_predicted_region_embed_l1_only_last_layer: 0.6f}; loss_predicted_region_embed_l1 {loss_predicted_region_embed_l1: 0.6f}; loss_predicted_region_embed_cos {loss_predicted_region_embed_cos: 0.6f}; loss_batchwise_contrastive {loss_batchwise_contrastive:0.6f}  loss_image_seen_class {loss_image_seen_class:0.6f} Loss_3d_2d_region_embed {loss_3d_2d_region_embed:0.6f} Loss_reg_embed {loss_reg_embed:0.6f} loss_contrast_3dto2d_text {loss_contrast_3dto2d_text:0.6f} loss_contrast_object_text {loss_contrast_object_text:0.6f} seen_classes_loss_sem_focal_cls {loss_sem_focal_cls:0.6f} anonymous_classes_loss_sem_focal_cls {loss_sem:0.6f} anonymous_classes_loss_sem_focal_cls_softmax {loss_sem_cls_softmax:0.6f} anonymous_loss_sem_cls_softmax_skip_none_gt_sample: {loss_sem_cls_softmax_skip_none_gt_sample:0.6f} anonymous_loss_sem_cls_softmax_discovery_novel_objectness: {loss_sem_cls_softmax_discovery_novel_objectness:0.6} loss_contrastive {loss_contrastive:0.6f} loss_angle_cls {loss_angle_cls:0.6f} loss_angle_reg {loss_angle_reg:0.6f} loss_cardinality {loss_cardinality:0.6f} loss_center {loss_center:0.6f} loss_size {loss_size:0.6f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            # print(loss_dict_reduced)
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1

    if args.if_online_keep_max_box_number:
        if args.ngpus > 1: # git the dic from all the gpus
            gathered_box_dict = [None for i in range(args.ngpus)]
            torch.distributed.barrier()
            torch.distributed.all_gather_object(gathered_box_dict, model.module.max_box_number_dict)
            for tmp_dic in gathered_box_dict:
                for tmp_keys in tmp_dic.keys():
                    if tmp_keys not in model.module.max_box_number_dict.keys():
                        model.module.max_box_number_dict[tmp_keys] = tmp_dic[tmp_keys]
                    else:
                        if tmp_dic[tmp_keys] > model.module.max_box_number_dict[tmp_keys]:
                            model.module.max_box_number_dict[tmp_keys] = tmp_dic[tmp_keys]


    # print(model.module.max_box_number_dict)
    if args.only_image_class:
        return None
    return ap_calculator

def if_near(box_list, x, y, thre):
    for box_item in box_list:
        if abs(x-box_item[0])<thre and abs(y-box_item[1])<thre:
            return True
    return False


def if_near_box(box_list, x1, y1, x2, y2, thre):
    for box_item in box_list:
        if abs(x1-box_item[0])<thre and abs(y1-box_item[1])<thre and abs(x2-box_item[2])<thre and abs(y2-box_item[3])<thre:
            return True
    return False


def camera_cord_to_image_plane(inputs, outputs, out_dir=None, if_padding_input=True, args=None):
    sem_cls_probs = outputs['sem_cls_prob']
    objectness_prob = outputs['objectness_prob']
    gt_classes = inputs['gt_box_sem_cls_label'].cpu().numpy()
    #pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    #pred_sem_cls = np.argmax(sem_cls_probs, -1)  # B,num_proposal. the max prob
    trans_mtx_batch = inputs['trans_mtx'].cpu().numpy()
    x_offset_batch = inputs['x_offset'].cpu().numpy()
    y_offset_batch = inputs['y_offset'].cpu().numpy()
    ori_width_batch = inputs['ori_width'].cpu().numpy()
    ori_height_batch = inputs['ori_height'].cpu().numpy()
    calib_name_list = inputs['calib_name']
    box_corners_gt = inputs['gt_box_corners'].cpu().numpy()
    img_batch = inputs['input_image'].cpu().numpy()
    im_name_list = inputs['im_name']
    # scan_name_list = inputs['scan_name']
    box_centers_batch = outputs['center_unnormalized']#.cpu().numpy()#inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
    # print(box_centers_batch[0].keys())
    # print(box_centers_batch[0].shape)
    box_angle_batch = outputs['angle_continuous']#.cpu().numpy() #inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
    #print(box_angle.shape)
    box_size_batch = outputs['size_unnormalized']#.cpu().numpy() #inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
    #print(box_size.shape)
    #print(len(calib_name_list))
    box_centers_batch_gt = inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
    #print(box_centers.shape)
    box_angle_batch_gt = inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
    #print(box_angle.shape)
    box_size_batch_gt = inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
    # trans_affine_batch = inputs['trans_affine'].cpu().numpy()
    if args.dataset_name.find('scannet') == -1:
        dataconfig = sunrgbd_v1_dataconfig
    else:
        dataconfig = scannet_dataconfig
    if args.dataset_name.find('scannet') != -1:
        data_utils = scannet_utils
    else:
        data_utils = sunrgbd_utils

    for idx_batch in range(len(calib_name_list)):
        x_offset = x_offset_batch[idx_batch]
        y_offset = y_offset_batch[idx_batch]
        ori_width = ori_width_batch[idx_batch]
        ori_height = ori_height_batch[idx_batch]

        trans_mtx = trans_mtx_batch[idx_batch]
        calib_name = calib_name_list[idx_batch]
        if args.dataset_name.find('scannet') != -1:
            squence_name = inputs['squence_name'][idx_batch]
            calib = data_utils.SCANNET_Calibration(calib_name, squence_name)
            # scene_name =
            im_name = os.path.basename(im_name_list[idx_batch])[:-4]
        else:
            calib = data_utils.SUNRGBD_Calibration(calib_name)
            im_name = os.path.basename(im_name_list[idx_batch])[:-4]

        img = img_batch[idx_batch]
        box_center_list = box_centers_batch[idx_batch]
        #print(box_angle.shape)
        box_angle_list = box_angle_batch[idx_batch]
        box_size_list = box_size_batch[idx_batch]

        box_center_gt_list = box_centers_batch_gt[idx_batch]
        #print(box_angle.shape)
        box_angle_gt_list = box_angle_batch_gt[idx_batch]
        box_size_gt_list = box_size_batch_gt[idx_batch]
        #print( box_corners_gt_list.shape)
        #print(calib_name)

        #labeled_class = {}
        #print(len(box_center_list))
        showed_label = {}

        img_draw2d_box = img.copy()#[:,:,::-1]
        # img_draw2d_box = img_draw2d_box[...,[2,1,0]]
        img_draw2d_box = cv2.cvtColor(img_draw2d_box, cv2.COLOR_RGB2BGR)
        # img_copy = img.copy()[:,:,::-1]
        if_has_novel = False
        for idx_box in range(len(box_center_gt_list)):
            # trans_mtx = trans_mtx_list[idx_box]
            box_center = box_center_gt_list[idx_box]
            box_angle = box_angle_gt_list[idx_box]
            box_size = box_size_gt_list[idx_box]
            # dataconfig.class2type[pred_sem_cls]
            #print(box_corners.shape)
            #print(np.min(box_corners))
            #print(np.max(box_corners))

            # if idx_box == 3:
            #     if if_padding_input:
            #         corners_3d_image, corners_3d, d = data_utils.compute_box_3d_offset(box_center/2, box_angle, box_size,
            #                                                                            calib, x_offset, y_offset)
            #         # print(corners_3d_image)
            #         xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
            #         ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
            #         xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
            #         ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
            #         corners_3d_image[:, 0] = np.clip(corners_3d_image[:, 0], 0, img.shape[1])
            #         corners_3d_image[:, 1] = np.clip(corners_3d_image[:, 1], 0, img.shape[0])
            #         if np.min(corners_3d_image[:, 0]) < 0 or np.min(corners_3d_image[:, 1]) < 0 or np.max(corners_3d_image[:, 0]) > (ori_width+y_offset) or np.max(corners_3d_image[:, 1]) > (ori_height+x_offset):
            #             continue
            #         print(d.shape)
            #         if np.min(d) < 0:
            #             continue
            #         print(im_name)
            #     else:
            #         corners_3d_image, corners_3d = data_utils.compute_box_3d_resize(box_center, box_angle, box_size,
            #                                                                         calib, trans_mtx)
            #         xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
            #         ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
            #         xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
            #         ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))
            #         # if if_padding_input:
            #         #     corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_offset(box_center, box_angle, box_size,
            #         #                                                                        calib, x_offset, y_offset)
            #         # else:
            #         #     corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size, calib, trans_mtx)
            #         # corners_3d_image,_ = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size, calib, trans_mtx)
            #         # print(len(box_center_gt_list))
            #         # print(corners_3d_image.shape)
            #     if np.max(box_size) == 0:
            #         continue
            #         # print(box_center)
            #         # print(box_angle)
            #         # xmin = np.min(corners_3d_image[:,0])
            #         # ymin = np.min(corners_3d_image[:,1])
            #         # print(xmin)
            #         # print(ymin)
            #         # print(box_class)
            #
            #         # box_class = dataconfig.class2type[int(gt_classes[idx_batch][idx_box])]
            #         # # print(gt_classes[idx_batch][idx_box])
            #         # if int(gt_classes[idx_batch][idx_box]) > 9:
            #         #     if_has_novel = True
            #         #     cv2.putText(img_draw2d_box, '%s' % (box_class), (max(int(xmin), 15),
            #         #                                                      max(int(ymin), 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #         #                 (255, 0, 0), 2)
            #         # else:
            #         #     cv2.putText(img_draw2d_box, '%s'%( box_class), (max(int(xmin),15),
            #         #     max(int(ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #         #     (0,255,0), 2)
            #         # print(np.max(corners_3d_image))
            #         # print(np.min(corners_3d_image))
            #         # print(img_draw2d_box.shape)
            #     # img_crop = img_copy[ymin:ymax, xmin:xmax, :]
            #     # cv2.imwrite(os.path.join(out_dir, im_name + '_3d_to_2d_box_gt_crop_%d.png'%idx_box), img_crop)
            #     img_draw2d_box = data_utils.draw_projected_box3d(img_draw2d_box, corners_3d_image,
            #                                                          color=(255, 0, 0))
            #     continue
            if if_padding_input:
                corners_3d_image, corners_3d, d = data_utils.compute_box_3d_offset(box_center, box_angle, box_size,
                                                                                   calib, x_offset, y_offset)
                # print(corners_3d_image)
                xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                # corners_3d_image[:, 0] = np.clip(corners_3d_image[:, 0], 0, img.shape[1])
                # corners_3d_image[:, 1] = np.clip(corners_3d_image[:, 1], 0, img.shape[0])
                corners_3d_image[:, 0] = np.clip(corners_3d_image[:, 0], y_offset, ori_width + y_offset)
                corners_3d_image[:, 1] = np.clip(corners_3d_image[:, 1], x_offset, ori_height + x_offset)
                if np.min(corners_3d_image[:, 0]) < 0 or np.min(corners_3d_image[:, 1]) < 0 or np.max(corners_3d_image[:, 0]) > (ori_width+y_offset) or np.max(corners_3d_image[:, 1]) > (ori_height+x_offset):
                    continue
                print(d.shape)
                if np.min(d) < 0:
                    continue
                print(im_name)
            else:
                corners_3d_image, corners_3d = data_utils.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                   calib, trans_mtx)
                xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))
            # if if_padding_input:
            #     corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_offset(box_center, box_angle, box_size,
            #                                                                        calib, x_offset, y_offset)
            # else:
            #     corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size, calib, trans_mtx)
            # corners_3d_image,_ = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size, calib, trans_mtx)
            #print(len(box_center_gt_list))
            #print(corners_3d_image.shape)
            if np.max(box_size) == 0:
                continue
            #print(box_center)
            #print(box_angle)
            # xmin = np.min(corners_3d_image[:,0])
            # ymin = np.min(corners_3d_image[:,1])
            #print(xmin)
            #print(ymin)
            #print(box_class)

            box_class = dataconfig.class2type[int(gt_classes[idx_batch][idx_box])]
            # print(gt_classes[idx_batch][idx_box])
            if int(gt_classes[idx_batch][idx_box]) > 9:
                if_has_novel = True
                cv2.putText(img_draw2d_box, '%s' % (box_class), (max(int(xmin), 15),
                                                                 max(int(ymin), 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 0, 0), 2)
                img_draw2d_box = data_utils.draw_projected_box3d(img_draw2d_box, corners_3d_image, color=(255, 0, 0))
            else:
                cv2.putText(img_draw2d_box, '%s'%( box_class), (max(int(xmin),15),
                max(int(ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,255,0), 2)
                img_draw2d_box = data_utils.draw_projected_box3d(img_draw2d_box, corners_3d_image, color=(0,255,0))
            #print(np.max(corners_3d_image))
            #print(np.min(corners_3d_image))
            #print(img_draw2d_box.shape)
            # img_crop = img_copy[ymin:ymax, xmin:xmax, :]
            # cv2.imwrite(os.path.join(out_dir, im_name + '_3d_to_2d_box_gt_crop_%d.png'%idx_box), img_crop)
            print(corners_3d_image)
            print(img_draw2d_box.shape)



            #break
        # if_has_novel = True
        # if not if_has_novel: # only write the image where there are novel classes
        #     continue
        cv2.imwrite(os.path.join(out_dir, im_name+'_3d_to_2d_box_gt.png'), img_draw2d_box)

        # trans_affine = trans_affine_batch[idx_batch]
        img_draw2d_box = img.copy()
        img_draw2d_box = cv2.cvtColor(img_draw2d_box, cv2.COLOR_RGB2BGR)
        for idx_box in range(len(box_center_list)):
            # trans_mtx = trans_mtx_list[idx_box]
            box_center = box_center_list[idx_box].cpu().numpy() 
            box_angle = box_angle_list[idx_box].cpu().numpy() 
            box_size = box_size_list[idx_box].cpu().numpy() 
            #print(box_corners.shape)
            pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs[idx_batch][idx_box], -1)  # B,num_proposal. the max prob
            pred_objectness_prob =  objectness_prob[idx_batch][idx_box]
            # if pred_objectness_prob < 0.75:
            #     continue
            pred_sem_cls_prob = pred_sem_cls_prob.cpu().numpy()
            pred_sem_cls = int(pred_sem_cls.cpu().numpy())
            pred_objectness_prob = pred_objectness_prob.cpu().numpy()
            # if pred_objectness_prob > 0.05:
            #     continue
            # print(pred_objectness_prob)
            #print('=================')
            #print(pred_sem_cls_prob)
            #print(pred_sem_cls)
            #print(dataconfig.class2type)
            if args.if_clip_superset:
                box_class = dataconfig.superclass2type[pred_sem_cls]
            else:
                box_class = dataconfig.class2type[pred_sem_cls]

            #print(np.max(box_corners))
            if if_padding_input:
                corners_3d_image, corners_3d, d = data_utils.compute_box_3d_offset(box_center, box_angle, box_size,
                                                                                   calib, x_offset, y_offset)
                xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                corners_3d_image[:, 0] = np.clip(corners_3d_image[:, 0], y_offset, ori_width + y_offset)
                corners_3d_image[:, 1] = np.clip(corners_3d_image[:, 1], x_offset, ori_height + x_offset)
                # if np.min(corners_3d_image[:, 0]) < 0 or np.min(corners_3d_image[:, 1]) < 0 or np.max(corners_3d_image[:, 0]) > (ori_width+y_offset) or np.max(corners_3d_image[:, 1]) > (ori_height+x_offset):
                #     continue
                # print(d.shape)
                # if np.min(d) < 0:
                #     continue
                # print(im_name)
            else:
                corners_3d_image, corners_3d = data_utils.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                   calib, trans_mtx)
                xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))
            # corners_3d_image = _affine_transform(corners_3d_image_, trans_affine)
            # print(corners_3d_image)
            # xmin = np.min(corners_3d_image[:,0])
            # ymin = np.min(corners_3d_image[:,1])
            if np.min(corners_3d_image[:, 0]) < 0 or np.min(corners_3d_image[:, 1]) < 0 or np.max(corners_3d_image[:, 0]) > (ori_width+y_offset) or np.max(corners_3d_image[:, 1]) > (ori_height+x_offset):
                continue
            print(d.shape)
            if np.min(d) < 0:
                continue

            if (box_class not in showed_label.keys()) or (not if_near(showed_label[box_class], xmin, ymin, 1e-16)):  # prevent too many texts for the same box to show, here is a bug, we can not show the biggest score and objectness actually.
                #print(box_class)
                if pred_sem_cls > 9:
                    # if pred_sem_cls_prob < 0.5:
                    #     continue
                    img_draw2d_box = data_utils.draw_projected_box3d(img_draw2d_box, corners_3d_image,
                                                                     color=(255, 0, 0))
                    cv2.putText(img_draw2d_box, '%s'%( box_class), (max(int(xmin),15),
                    max(int(ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,0,0), 2)
                    cv2.putText(img_draw2d_box, '%.02f'%( pred_sem_cls_prob), (max(int(xmin),15),
                    max(int(ymin),15)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,0,0), 2)
                    cv2.putText(img_draw2d_box, '%.02f'% pred_objectness_prob, (max(int(xmin),15),
                    max(int(ymin),15)+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,0,0), 2)
                else:
                    # if pred_sem_cls_prob < 0.5:
                    #     continue
                    img_draw2d_box = data_utils.draw_projected_box3d(img_draw2d_box, corners_3d_image,
                                                                     color=(0, 255, 0))
                    cv2.putText(img_draw2d_box, '%s'%( box_class), (max(int(xmin),15),
                    max(int(ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)
                    cv2.putText(img_draw2d_box, '%.02f'%( pred_sem_cls_prob), (max(int(xmin),15),
                    max(int(ymin),15)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)
                    cv2.putText(img_draw2d_box, '%.02f'% pred_objectness_prob, (max(int(xmin),15),
                    max(int(ymin),15)+50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)
            # try:
            #     if info_list != None:
            #         tmp_gt_class = dataconfig.class2type[int(gt_classes[idx_batch][idx_box])]
            #         if box_class != tmp_gt_class:
            #             if (box_class,tmp_gt_class) in info_list.keys():
            #                 info_list[(box_class,tmp_gt_class)] += 1
            #             else:
            #                 info_list[(box_class, tmp_gt_class)] = 0
            # except:

            # if info_list != None:
            #     # if idx_box < 64:
            #     tmp_gt_class = dataconfig.class2type[int(gt_classes[idx_batch][idx_box])]
            #     if box_class != tmp_gt_class:
            #         if (box_class,tmp_gt_class) in info_list.keys():
            #             info_list[(box_class,tmp_gt_class)] += 1
            #         else:
            #             info_list[(box_class, tmp_gt_class)] = 0
            # # except:
            #
            #     print(len(box_center_gt_list))
            #
            #     print('Here is an error:')
            #     print(idx_batch)
            #     print(idx_box)
            #     print('====================')
            if box_class not in showed_label.keys():
                showed_label[box_class] = []
            showed_label[box_class].append([xmin, ymin])
          #  print(np.max(corners_3d_image))
          #  print(np.min(corners_3d_image))
          #  print(img_draw2d_box.shape)
          #  print(corners_3d_image.shape)
          #   if pred_objectness_prob > 0.5:

            # print(pred_sem_cls)
            # print('=================================================')

          #  print(corners_3d_image.shape)
          #  print(np.min(corners_3d_image[:,0]))
          #  print(np.min(corners_3d_image[:,1]))
          #  print(np.max(corners_3d_image[:,0]))
          #  print(np.max(corners_3d_image[:,1]))
            #break
        # print('===================================')
        # print(os.path.join(out_dir, im_name+'_3d_to_2d_box_pred.png'))
        cv2.imwrite(os.path.join(out_dir, im_name+'_3d_to_2d_box_pred.png'), img_draw2d_box)



    # return info_list



def crop_camera_cord_to_image_plane(inputs, outputs, out_dir=None, if_padding_input=True):
    trans_mtx_batch = inputs['trans_mtx'].cpu().numpy()
    x_offset_batch = inputs['x_offset'].cpu().numpy()
    y_offset_batch = inputs['y_offset'].cpu().numpy()
    ori_width_batch = inputs['ori_width'].cpu().numpy()
    ori_height_batch = inputs['ori_height'].cpu().numpy()
    sem_cls_probs = outputs['sem_cls_prob']
    objectness_prob = outputs['objectness_prob']
    gt_classes = inputs['gt_box_sem_cls_label'].cpu().numpy()
    #pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal. the max prob
    
    calib_name_list = inputs['calib_name']
    box_corners_gt = inputs['gt_box_corners'].cpu().numpy()
    img_batch = inputs['input_image'].cpu().numpy()
    im_name_list = inputs['im_name']
    box_centers_batch = outputs['center_unnormalized']#.cpu().numpy()#inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
    #print(box_centers.shape)
    box_angle_batch = outputs['angle_continuous']#.cpu().numpy() #inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
    #print(box_angle.shape)
    box_size_batch = outputs['size_unnormalized']#.cpu().numpy() #inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()
    #print(len(calib_name_list))
    box_centers_batch_gt = inputs['gt_box_centers'].cpu().numpy()#outputs['center_unnormalized'].cpu().numpy()
    #print(box_centers.shape)
    box_angle_batch_gt = inputs['gt_box_angles'].cpu().numpy() #outputs['angle_continuous'].cpu().numpy()
    #print(box_angle.shape)
    box_size_batch_gt = inputs["gt_box_sizes"].cpu().numpy() #outputs['size_unnormalized'].cpu().numpy()

    for idx_batch in range(len(calib_name_list)):
        x_offset = x_offset_batch[idx_batch]
        y_offset = y_offset_batch[idx_batch]
        ori_width = ori_width_batch[idx_batch]
        ori_height = ori_height_batch[idx_batch]
        trans_mtx = trans_mtx_batch[idx_batch]
        calib_name = calib_name_list[idx_batch]
        im_name = os.path.basename(im_name_list[idx_batch])[:-4]
        img = img_batch[idx_batch]
        box_center_list = box_centers_batch[idx_batch]
        #print(box_angle.shape)
        box_angle_list = box_angle_batch[idx_batch]
        box_size_list = box_size_batch[idx_batch]

        box_center_gt_list = box_centers_batch_gt[idx_batch]
        #print(box_angle.shape)
        box_angle_gt_list = box_angle_batch_gt[idx_batch]
        box_size_gt_list = box_size_batch_gt[idx_batch]
        #print( box_corners_gt_list.shape)
        #print(calib_name)
        calib = sunrgbd_utils.SUNRGBD_Calibration(calib_name)
        img_draw2d_box = img.copy()
        #print(img.shape)
        #labeled_class = {}
        #print(len(box_center_list))
        showed_label = []
        os.makedirs(out_dir+'/projected_2d_box_region/', exist_ok=True)
        for idx_box in range(len(box_center_list)):
            box_center = box_center_list[idx_box].cpu().numpy() 
            box_angle = box_angle_list[idx_box].cpu().numpy() 
            box_size = box_size_list[idx_box].cpu().numpy() 
            #print(box_corners.shape)
            pred_sem_cls_prob, pred_sem_cls = torch.max(sem_cls_probs[idx_batch][idx_box], -1)  # B,num_proposal. the max prob
            pred_objectness_prob =  objectness_prob[idx_batch][idx_box]
            pred_sem_cls_prob = pred_sem_cls_prob.cpu().numpy()
            pred_sem_cls = int(pred_sem_cls.cpu().numpy())
            pred_objectness_prob = pred_objectness_prob.cpu().numpy()
            #print('=================')
            #print(pred_sem_cls_prob)
            #print(pred_sem_cls)
            #print(dataconfig.class2type)
            box_class = dataconfig.class2type[pred_sem_cls]


            #print(np.max(box_corners))
            # corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d(box_center, box_angle, box_size, calib)
            # corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size, calib,
            #                                                                    trans_mtx)
            if np.max(box_size) == 0:
                continue


            if if_padding_input:
                corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_offset(box_center, box_angle, box_size,
                                                                                   calib, x_offset, y_offset)
                xmin = int(min(max(np.min(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                ymin = int(min(max(np.min(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
                xmax = int(min(max(np.max(corners_3d_image[:, 0]), y_offset), ori_width + y_offset))
                ymax = int(min(max(np.max(corners_3d_image[:, 1]), x_offset), ori_height + x_offset))
            else:
                corners_3d_image, corners_3d = sunrgbd_utils.compute_box_3d_resize(box_center, box_angle, box_size,
                                                                                   calib, trans_mtx)
                xmin = int(min(max(np.min(corners_3d_image[:, 0]), 0), img.shape[1]))
                ymin = int(min(max(np.min(corners_3d_image[:, 1]), 0), img.shape[0]))
                xmax = int(min(max(np.max(corners_3d_image[:, 0]), 0), img.shape[1]))
                ymax = int(min(max(np.max(corners_3d_image[:, 1]), 0), img.shape[0]))

            # xmin = int(min(max(np.min(corners_3d_image[:,0]), 0), img.shape[1]))
            # ymin = int(min(max(np.min(corners_3d_image[:,1]), 0), img.shape[0]))
            # xmax = int(min(max(np.max(corners_3d_image[:,0]), 0), img.shape[1]))
            # ymax = int(min(max(np.max(corners_3d_image[:,1]), 0), img.shape[0]))


            ############## hypothetical exp
            # w = ymax-ymin
            # h = xmax-xmin
            # if w > h:
            #     xmin = xmin - (w-h)//2
            #     xmax = xmax + (w-h)//2
            # else:
            #     ymin = ymin - (h-w)//2
            #     ymax = ymax + (h-w)//2
            # xmin = int(min(max(xmin, 0), img.shape[1]))
            # ymin = int(min(max(ymin, 0), img.shape[0]))
            # xmax = int(min(max(xmax, 0), img.shape[1]))
            # ymax = int(min(max(ymax, 0), img.shape[0]))
            ###############
            if  (not if_near_box(showed_label, xmin, ymin, xmax, ymax, 5)):  # prevent too many texts for the same box to show, here is a bug, we can not show the biggest score and objectness actually.
            #print(np.min(corners_3d_image[:,0]))
            #print(np.min(corners_3d_image[:,1]))
            #print(np.max(corners_3d_image[:,0]))
            #print(np.max(corners_3d_image[:,1]))
                showed_label.append([xmin, ymin, xmax, ymax])
            #print(xmin)
            #print(ymin)
            #print(xmax)
            #print(ymax)

                if (xmax-xmin)>0 and (ymax-ymin)>0:
                    img_crop = img[ymin:ymax, xmin:xmax, :]
            #print(img_crop.shape)
                #print(box_class)
         #       cv2.putText(img_draw2d_box, '%s'%( box_class), (max(int(xmin),15),
            #    print(os.path.join(out_dir, 'projected_2d_box_region/'+im_name+'_%05d.png' % idx_box))
            #    print(img_crop.shape)

                    #####exp1: blank background#####
                    # img_crop_ = img_crop.clone()
                    w = ymax - ymin
                    h = xmax - xmin
                    if w > h:
                        max_edge = w
                    else:
                        max_edge = h
                    # max_edge = max_edge + 20
                    # print(img_crop.shape)
                    # print(torch.max(img_crop))
                    # print(torch.min(img_crop))
                    img_beckground = np.ones((max_edge, max_edge, img.shape[2]), dtype=np.uint8) * 255

                    # print(torch.max(img_beckground))
                    y_begin = (max_edge - w) // 2
                    x_begin = (max_edge - h) // 2
                    img_beckground[y_begin:y_begin + w, x_begin:x_begin + h, :] = img_crop
                    ###################################
                    ###############################################
                    # select_class = ['plants'] #['chairs', 'endtable', 'tissuebox', 'books', 'walldecor', 'frige', 'bulletinboard', 'plants', 'shoes', 'stepstool']
                    # # [, , 'frige', 'bulletinboard', 'plants', 'shoes', 'stepstool']
                    # if box_class in select_class:
                    #     print(box_class)
                    #     cv2.imwrite(os.path.join(out_dir, 'projected_2d_box_region/' + im_name + '_%05d_%s.png' % (idx_box, box_class)),
                    #                 img_beckground)
                    ###############################################
                    cv2.imwrite(os.path.join(out_dir, 'projected_2d_box_region/' + im_name + '_%05d_%s.png' % (idx_box, box_class)),
                                img_beckground)
                    # cv2.imwrite(os.path.join(out_dir, 'projected_2d_box_region/'+im_name+'_%05d.png' % idx_box), img_crop)


#       img_draw2d_box = img.copy()
#       for idx_box in range(len(box_center_gt_list)):
#           box_center = box_center_gt_list[idx_box]
#           box_angle = box_angle_gt_list[idx_box]
#           box_size = box_size_gt_list[idx_box]
#           #print(box_corners.shape)
#           #print(np.min(box_corners))
#           #print(np.max(box_corners))
#           corners_3d_image,_ = sunrgbd_utils.compute_box_3d(box_center, box_angle, box_size, calib)
#           #print(len(box_center_gt_list))
#           #print(corners_3d_image.shape)
#           if np.max(box_size) == 0:
#               continue
#           #print(box_center)
#           #print(box_angle)
#           #print(box_size)
#           #print(corners_3d_image)
#           #print(np.min(corners_3d_image[:,0]))
#           xmin = np.min(corners_3d_image[:,0])
#           ymin = np.min(corners_3d_image[:,1])
#           #print(xmin)
#           #print(ymin)
#           #print(box_class)
#           box_class = dataconfig.class2type[int(gt_classes[idx_batch][idx_box])]
#           cv2.putText(img_draw2d_box, '%s'%( box_class), (max(int(xmin),15),
#               max(int(ymin),15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#               (0,255,0), 2)
#           #print(np.max(corners_3d_image))
#           #print(np.min(corners_3d_image))
#           #print(img_draw2d_box.shape)
#           #print(corners_3d_image.shape)
#           img_draw2d_box = sunrgbd_utils.draw_projected_box3d(img_draw2d_box, corners_3d_image)
#           #break
#       cv2.imwrite(os.path.join(out_dir, im_name+'_3d_to_2d_box_gt.png'), img_draw2d_box)



@torch.no_grad()
def crop_image(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
    out_dir,
    if_after_nms=False
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            #print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]
                

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        # }
        outputs = model(batch_data_label)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        #print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        im_name_list = batch_data_label['im_name']
        pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)

        for sample_idx in range(len(point_cloud_rgb_this_batch)):
            im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
            os.makedirs(out_dir+'/pc/', exist_ok=True)
            write_ply_rgb(tmp_point_clouds_np[:, 0:6], out_dir+'/pc/'+'%s_pc.obj' % im_name_tmp)



            boxes_this_image = []
            for box_idx in range(len(pred_obb_this_batch[sample_idx])):
            #print('==================')
            #print(len(tmp_pred_map_cls))
            #print(len(tmp_pred_map_cls[batch_idx]))
                boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy())
            #break
            boxes_this_image = np.array(boxes_this_image)
        #print(boxes_this_image.shape)
            if boxes_this_image.shape[0]>0:
                os.makedirs(out_dir+'/pred_bboxes_obb/', exist_ok=True)
                write_oriented_bbox(boxes_this_image, out_dir+'/pred_bboxes_obb/'+'%s_pred.ply' % im_name_tmp)

            boxes_this_image_gt = []
            for box_idx in range(len(gt_this_batch[sample_idx])):
            #print('==================')
            #print(len(tmp_pred_map_cls))
            #print(len(tmp_pred_map_cls[batch_idx]))
                boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
            #break
            boxes_this_image_gt = np.array(boxes_this_image_gt)
            if boxes_this_image_gt.shape[0] > 0:
                os.makedirs(out_dir+'/gt_bboxes_obb/', exist_ok=True)
                write_oriented_bbox(boxes_this_image_gt, out_dir+'/gt_bboxes_obb/'+'%s_gt.ply' % im_name_tmp)
        #print(len(ap_calculator.pred_map_cls))
        #pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        #print('===========================')
        #print(len(pred_obb_this_batch))
        output_nms = {}
        output_nms['center_unnormalized'] = {}
        output_nms['size_unnormalized'] = {}
        output_nms['angle_continuous'] = {}
        output_nms['class_max'] = {}
        output_nms['sem_cls_prob'] = {}
        output_nms['objectness_prob'] = {}
        for idx_batch in range(len(pred_obb_this_batch)):
            output_nms['center_unnormalized'][idx_batch] = {}
            output_nms['size_unnormalized'][idx_batch] = {}
            output_nms['angle_continuous'][idx_batch] = {}
            output_nms['sem_cls_prob'][idx_batch] = {}
            output_nms['objectness_prob'][idx_batch] = {}
            for idx_box in range(len(pred_obb_this_batch[idx_batch])):
                output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
                output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
                output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
                output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:17]
                output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][17]
            
        

        if if_after_nms:
            crop_camera_cord_to_image_plane(batch_data_label, output_nms, out_dir)
        else:
            crop_camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir)


        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        #break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator


@torch.no_grad()
def calculate_wrong_class(
        args,
        curr_epoch,
        model,
        criterion,
        dataset_config,
        dataset_loader,
        logger,
        curr_train_iter,
        out_dir,
        if_after_nms=True
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )



    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    info_list = {}
    right_list = {}
    wrong_list = {}
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            # print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #
        # }
        outputs = model(batch_data_label)
        # Compute loss
        # loss_str = ""
        # if criterion is not None:
        #     loss, loss_dict = criterion(outputs, batch_data_label)
        #
        #     loss_reduced = all_reduce_average(loss)
        #     loss_dict_reduced = reduce_dict(loss_dict)
        #
        #     loss_avg.update(loss_reduced.item())
        #     loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        # print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        # im_name_list = batch_data_label['im_name']
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs,
        #                                                                                                batch_data_label)
        # print('44444444444444444444444')
        # for sample_idx in range(len(point_cloud_rgb_this_batch)):
        #     im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
        #
        #     # just comments for a while
        #     # tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
        #     # write_ply_rgb(tmp_point_clouds_np[:, 0:3], (tmp_point_clouds_np[:, 3:]*256).astype(np.int8), out_dir+'/'+'%s_pc.obj' % im_name_tmp)
        #
        #     boxes_this_image = []
        #     for box_idx in range(len(pred_obb_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy())
        #     # break
        #     boxes_this_image = np.array(boxes_this_image)
        #     # print(boxes_this_image.shape)
        #
        #     # just comments for a while
        #     # if boxes_this_image.shape[0]>0:
        #     #     write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_pred.ply' % im_name_tmp)
        #
        #     boxes_this_image_gt = []
        #     for box_idx in range(len(gt_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
        #     # break
        #     boxes_this_image_gt = np.array(boxes_this_image_gt)

            # just comments for a while
            # if boxes_this_image_gt.shape[0] > 0:
            #     write_oriented_bbox(boxes_this_image_gt, out_dir+'/'+'%s_gt.ply' % im_name_tmp)

        # print('333333333333333333333333333')
        # print(len(ap_calculator.pred_map_cls))
        # pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        # print('===========================')
        # print(len(pred_obb_this_batch))
        # output_nms = {}
        # output_nms['center_unnormalized'] = {}
        # output_nms['size_unnormalized'] = {}
        # output_nms['angle_continuous'] = {}
        # output_nms['class_max'] = {}
        # output_nms['sem_cls_prob'] = {}
        # output_nms['objectness_prob'] = {}
        # for idx_batch in range(len(pred_obb_this_batch)):
        #     output_nms['center_unnormalized'][idx_batch] = {}
        #     output_nms['size_unnormalized'][idx_batch] = {}
        #     output_nms['angle_continuous'][idx_batch] = {}
        #     output_nms['sem_cls_prob'][idx_batch] = {}
        #     output_nms['objectness_prob'][idx_batch] = {}
        #     for idx_box in range(len(pred_obb_this_batch[idx_batch])):
        #         output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
        #         output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
        #         output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
        #         output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:-1]
        #         output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][-1]

        # print('2222222222222222222222222')

        ### get the info
        pred_prob_batch = outputs['outputs']['sem_cls_prob']
        gt_classes_batch = batch_data_label['gt_box_sem_cls_label']
        for idx_batch in range(len(pred_prob_batch)):
            # print(len(batch_data_label))
            # print(len(pred_prob_batch))
            for idx_box in range(len(pred_prob_batch[idx_batch])):
                pred_sem_cls_prob, pred_sem_cls = torch.max(pred_prob_batch[idx_batch][idx_box], -1)
                pred_sem_cls = int(pred_sem_cls.cpu().numpy())
                box_class = dataconfig.class2type[pred_sem_cls]
                tmp_gt_class = dataconfig.class2type[int(gt_classes_batch[idx_batch][idx_box])]

                if box_class != tmp_gt_class:
                    if box_class in wrong_list.keys():
                        wrong_list[box_class] += 1
                    else:
                        wrong_list[box_class] = 1
                    if (box_class, tmp_gt_class) in info_list.keys():
                        info_list[(box_class, tmp_gt_class)] += 1
                    else:
                        info_list[(box_class, tmp_gt_class)] = 1
                else:
                    if box_class in right_list.keys():
                        right_list[box_class] += 1
                    else:
                        right_list[box_class] = 1
            # print(pred_prob_batch.shape)
            # print(gt_classes_batch.shape)
            # fsd
        # if if_after_nms:
        #     # print('1111111111111111111111111111111')
        #     camera_cord_to_image_plane(batch_data_label, output_nms, out_dir)
        # else:
        #     camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir)

        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        # break
    print('============================Info list============================')
    sorted_info_list = sorted(info_list.items(), key=lambda d:d[1], reverse = False )
    print(sorted_info_list)
    print('============================Wrong list============================')
    print(wrong_list)
    print('============================Right list============================')
    print(right_list)

    Accu = {}
    for key in wrong_list.keys():
        if key in right_list.keys():
            sum_key = wrong_list[key] + right_list[key]
            Accu[key] = right_list[key]*1.0 / sum_key
        else:
            Accu[key] = 0.0
    sorted_Accu = sorted(Accu.items(), key=lambda d: d[1])
    print('============================Acc list============================')
    print(sorted_Accu)

    with open('cal_classes_log.txt', 'a') as file:
        file.write('============================Class map list============================\n')
        file.write(json.dumps(sorted_info_list))
        file.write('============================Wrong list============================\n')
        file.write(json.dumps(wrong_list))
        file.write('============================Right list============================\n')
        file.write(json.dumps(right_list))
        file.write('============================Acc list============================\n')
        file.write(json.dumps(sorted_Accu))

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator

@torch.no_grad()
def show_boxes(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
    out_dir,
    if_after_nms=True
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    if args.dataset_name.find('scannet')==-1:
        dataconfig = sunrgbd_v1_dataconfig
    else:
        dataconfig = scannet_dataconfig

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""


    # info_list = {}
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            #print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]
                

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #
        # }
        outputs = model(batch_data_label, if_real_test=True)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        #print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        im_name_list = batch_data_label['im_name']


        pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show_nms_then_iou(outputs, batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)
        # print('44444444444444444444444')
        for sample_idx in range(len(point_cloud_rgb_this_batch)):
            if args.dataset_name.find('scannet') != -1:
                scan_name = batch_data_label['data_name'][sample_idx]
                im_name_tmp = scan_name + '_' + os.path.basename(im_name_list[sample_idx])[:-4]
            else:
                im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            # if im_name_tmp != '000080':
            #     continue
            # just comments for a while
            color_pc_path = 'Data/sunrgb_d/sunrgbd_v1_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_0415_val/%s' % im_name_tmp
            # read_pc_data(color_pc_path, out_path=out_dir)
            tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
            write_ply_rgb(tmp_point_clouds_np[:, 0:6], out_dir+'/'+'%s_pc.ply' % im_name_tmp)



            boxes_this_image = []
            boxes_cls_max = []
            for box_idx in range(len(pred_obb_this_batch[sample_idx])):
            #print('=================='
            #print(len(tmp_pred_map_cls))
            #print(len(tmp_pred_map_cls[batch_idx]))
                max_pro_box = torch.max(pred_obb_this_batch[sample_idx][box_idx][3][7:-1]).cpu().numpy()
                #print('=====================================')
                #print(pred_obb_this_batch[sample_idx][box_idx][3][7:-1].shape)
                # if max_pro_box < 0.5:  # only show the boxes with max prob > 0.5
                #     continue
                boxes_cls_max.append(max_pro_box)
                boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy())

            #break
            boxes_this_image = np.array(boxes_this_image)
        #print(boxes_this_image.shape)

            # # just comments for a while
            boxes_cls_max = np.array(boxes_cls_max) # do not select larger than 0.5
            if boxes_this_image.shape[0]>0:
                print('----------------------------')
                # print(boxes_this_image.shape)
                print('==============================')
                # for idx_show_box in range(boxes_this_image.shape[0]):
                #     sem_max_prob = boxes_cls_max[idx_show_box]
                    # if sem_max_prob > 0.5:
                    #     tmp_box = boxes_this_image[idx_show_box,np.newaxis,:]
                    #     tmp_box[:, -1] = tmp_box[:, -1] * -1
                    #     write_oriented_bbox(boxes_this_image[idx_show_box,np.newaxis,:], out_dir + '/' + '%s_pred_%d.ply' % (im_name_tmp, idx_show_box))
                tmp_box = boxes_this_image[:, :7].copy()
                if args.dataset_name.find('sunrgbd') != -1:
                    tmp_box[:, -1] = tmp_box[:, -1]  * -1
                tmp_box = tmp_box #[boxes_cls_max>0.5,:]
                # print(tmp_box[:, -1])
                # print('********************')
                # print(tmp_box.shape)
                write_oriented_bbox(tmp_box, out_dir+'/'+'%s_pred.ply' % im_name_tmp)

            boxes_this_image_gt = []
            for box_idx in range(len(gt_this_batch[sample_idx])):
            #print('==================')
            #print(len(tmp_pred_map_cls))
            #print(len(tmp_pred_map_cls[batch_idx]))
                boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
            #break
            boxes_this_image_gt = np.array(boxes_this_image_gt)

            # # just comments for a while
            if boxes_this_image_gt.shape[0] > 0:
                # print(boxes_this_image_gt.shape)
                boxes_this_image_gt_tmp = boxes_this_image_gt.copy()
                if args.dataset_name.find('sunrgbd') != -1:
                    boxes_this_image_gt_tmp[:, -1] *= -1
                write_oriented_bbox(boxes_this_image_gt_tmp, out_dir+'/'+'%s_gt.ply' % im_name_tmp)
                # for idx_show_box in range(boxes_this_image_gt.shape[0]):
                #     write_oriented_bbox(boxes_this_image_gt[idx_show_box,np.newaxis,:], out_dir + '/' + '%s_gt_%d.ply' % (im_name_tmp, idx_show_box))
                #     tmp_box = boxes_this_image_gt[idx_show_box, np.newaxis, :]
                #     tmp_box[:, :3] = tmp_box[:, :3]/2
                #     write_oriented_bbox(tmp_box, out_dir + '/' + '%s_noise_%d.ply' % (im_name_tmp, idx_show_box))
                # print(out_dir+'/'+'%s_gt.ply' % im_name_tmp)

        # continue
        # print('333333333333333333333333333')
        #print(len(ap_calculator.pred_map_cls))
        #pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        #print('===========================')
        #print(len(pred_obb_this_batch))
        output_nms = {}
        output_nms['center_unnormalized'] = {}
        output_nms['size_unnormalized'] = {}
        output_nms['angle_continuous'] = {}
        output_nms['class_max'] = {}
        output_nms['sem_cls_prob'] = {}
        output_nms['objectness_prob'] = {}
        for idx_batch in range(len(pred_obb_this_batch)):
            output_nms['center_unnormalized'][idx_batch] = {}
            output_nms['size_unnormalized'][idx_batch] = {}
            output_nms['angle_continuous'][idx_batch] = {}
            output_nms['sem_cls_prob'][idx_batch] = {}
            output_nms['objectness_prob'][idx_batch] = {}
            for idx_box in range(len(pred_obb_this_batch[idx_batch])):
                output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
                output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
                output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
                output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:-1]
                output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][-1]

        # print('2222222222222222222222222')
        # print(output_nms['angle_continuous'][idx_batch][idx_box])
        ### get the info
        pred_prob_batch = outputs['outputs']['sem_cls_prob']
        gt_classes_batch = batch_data_label['gt_box_sem_cls_label']
        for idx_batch in range(len(pred_prob_batch)):
            # print(len(batch_data_label))
            # print(len(pred_prob_batch))
            for idx_box in range(len(pred_prob_batch[idx_batch])):
                pred_sem_cls_prob, pred_sem_cls = torch.max(pred_prob_batch[idx_batch][idx_box], -1)
                pred_sem_cls = int(pred_sem_cls.cpu().numpy())
                if args.if_clip_superset:
                    box_class = dataconfig.superclass2type[pred_sem_cls]
                else:
                    box_class = dataconfig.class2type[pred_sem_cls]

                # tmp_gt_class = dataconfig.class2type[int(gt_classes_batch[idx_batch][idx_box])]
                # if box_class != tmp_gt_class:
                #     if (box_class,tmp_gt_class) in info_list.keys():
                #         info_list[(box_class,tmp_gt_class)] += 1
                #     else:
                #         info_list[(box_class, tmp_gt_class)] = 0
            # print(pred_prob_batch.shape)
            # print(gt_classes_batch.shape)
            # fsd
        if if_after_nms:
            # print('1111111111111111111111111111111')
            camera_cord_to_image_plane(batch_data_label, output_nms, out_dir, args=args)
        else:
            camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir, args=args)

        # print(info_list)
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        #break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator


@torch.no_grad()
def save_box_points(
        args,
        curr_epoch,
        model,
        criterion,
        dataset_config,
        dataset_loader,
        logger,
        curr_train_iter,
        out_dir,
        if_after_nms=True
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    if args.dataset_name.find('scannet') == -1:
        dataconfig = sunrgbd_v1_dataconfig
    else:
        dataconfig = scannet_dataconfig

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    # info_list = {}
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            # print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #
        # }
        outputs = model(batch_data_label, if_real_test=True)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        # print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        im_name_list = batch_data_label['im_name']

        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs,
        #                                                                                                batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show_nms_then_iou(outputs, batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)
        # print('44444444444444444444444')
        output_corners = outputs["outputs"]['box_corners'].cpu().numpy()
        pred_objectness = outputs["outputs"]['objectness_prob'].cpu().numpy()
        pred_center = outputs["outputs"]['center_unnormalized'].cpu().numpy()
        pred_size = outputs["outputs"]['size_unnormalized'].cpu().numpy()
        pred_angle = outputs["outputs"]['angle_continuous'].cpu().numpy()
        pc_input = batch_data_label['point_clouds'].cpu().numpy()

        for sample_idx in range(pred_angle.shape[0]):
            if args.dataset_name.find('scannet') != -1:
                scan_name = batch_data_label['data_name'][sample_idx]
                im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            else:
                im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            # if im_name_tmp != '000080':
            #     continue
            # just comments for a while
            # color_pc_path = 'Data/sunrgb_d/sunrgbd_v1_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_0415_val/%s' % im_name_tmp
            # read_pc_data(color_pc_path, out_path=out_dir)
            # tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy() # np.min: -1.74, max: 8.3

            pred_corners_this_batch = output_corners[sample_idx]
            pred_objectness_this_batch = pred_objectness[sample_idx]
            boxes_this_image_corners = []
            pred_center_this_batch = pred_center[sample_idx]
            pred_size_this_batch = pred_size[sample_idx]
            pred_angle_this_batch = pred_angle[sample_idx]
            pc_this_batch = pc_input[sample_idx]
            # write_ply(pc_this_batch, out_dir + '/' + '%s_pc.ply' % im_name_tmp)  # value range:
            for box_idx in range(pred_corners_this_batch.shape[0]):
                # print('==================')
                # print(len(tmp_pred_map_cls))
                # print(len(tmp_pred_map_cls[batch_idx]))
                boxes_this_image_corners.append(pred_corners_this_batch[box_idx]) # 1 for corners
            # break
            boxes_this_image_corners = np.array(boxes_this_image_corners)


            if boxes_this_image_corners.shape[0] > 0:
                # print(boxes_this_image_corners.shape)
                boxes_this_image_corners = boxes_this_image_corners.copy()
                ############ save box points
                for idx_box in range(boxes_this_image_corners.shape[0]):
                    # box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                    if pred_objectness_this_batch[idx_box] < 0.05:
                        continue
                    center_tmp = pred_center_this_batch[idx_box]
                    size_tmp = pred_size_this_batch[idx_box]
                    angle_tmp = np.array(pred_angle_this_batch[idx_box])[np.newaxis]
                    obj_tmp = np.array(pred_objectness_this_batch[idx_box])[np.newaxis]
                    # if args.dataset_name.find('scannet')!=-1:
                    #     box_tmp = np.concatenate((center_tmp, size_tmp, -1*angle_tmp, obj_tmp))[np.newaxis, :]
                    # else:
                    box_tmp = np.concatenate((center_tmp, size_tmp, angle_tmp, obj_tmp))[np.newaxis, :]
                    box3d = flip_axis_to_depth(boxes_this_image_corners[idx_box])
                    pc_tmp, _ = extract_pc_in_box3d(pc_this_batch, box3d)
                    if len(pc_tmp) < 5: #
                        continue
                    np.save(out_dir + '/' + '%s_%04d_pred_box.npy' % (im_name_tmp, idx_box), box_tmp) # even the shown boxes(*.ply) is not right, the boxes are right.
                    np.save(out_dir + '/' + '%s_%04d_pred_pc.npy' % (im_name_tmp, idx_box), pc_tmp)
                    # write_oriented_bbox(box_tmp, out_dir + '/' + '%s_%04d_pred_box.ply' % (im_name_tmp, idx_box))
                    # write_ply(pc_tmp, out_dir + '/' + '%s_%04d_pred_pc.ply' % (im_name_tmp, idx_box))


            # boxes_this_image = []
            # for box_idx in range(len(pred_obb_this_batch[sample_idx])):
            #     # print('==================')
            #     # print(len(tmp_pred_map_cls))
            #     # print(len(tmp_pred_map_cls[batch_idx]))
            #     boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy()) # 1 for corners
            # # break
            # boxes_this_image = np.array(boxes_this_image)
            # print(boxes_this_image.shape)

            # boxes_this_image_corners = []
            # for box_idx in range(len(pred_obb_this_batch[sample_idx])):
            #     # print('==================')
            #     # print(len(tmp_pred_map_cls))
            #     # print(len(tmp_pred_map_cls[batch_idx]))
            #     boxes_this_image_corners.append(pred_obb_this_batch[sample_idx][box_idx][1]) # 1 for corners
            # # break
            # boxes_this_image_corners = np.array(boxes_this_image_corners)
            #
            #
            # if boxes_this_image_corners.shape[0] > 0:
            #     print(boxes_this_image_corners.shape)
            #     boxes_this_image_corners = boxes_this_image_corners.copy()
            #     ############ save box points
            #     for idx_box in range(boxes_this_image_corners.shape[0]):
            #         # box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
            #         box3d = flip_axis_to_depth(boxes_this_image_corners[idx_box])
            #         pc_tmp, _ = extract_pc_in_box3d(tmp_point_clouds_np, box3d)
            #         write_ply_rgb(pc_tmp, out_dir + '/' + '%s_%d_pred_pc.ply' % (im_name_tmp, idx_box))
                ############ save box points


            # # # just comments for a while
            # if boxes_this_image.shape[0] > 0:
            #     # for idx_show_box in range(boxes_this_image.shape[0]):
            #     #     write_oriented_bbox(boxes_this_image[idx_show_box,np.newaxis,:], out_dir + '/' + '%s_pred_%d.ply' % (im_name_tmp, idx_show_box))
            #     write_oriented_bbox(boxes_this_image, out_dir + '/' + '%s_pred.ply' % im_name_tmp)
            #
            # boxes_this_image_gt = []
            # for box_idx in range(len(gt_this_batch[sample_idx])):
            #     # print('==================')
            #     # print(len(tmp_pred_map_cls))
            #     # print(len(tmp_pred_map_cls[batch_idx]))
            #     boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
            # # break
            # boxes_this_image_gt = np.array(boxes_this_image_gt)

            # boxes_this_image_gt_corners = []
            # for box_idx in range(len(gt_this_batch[sample_idx])):
            #     # print('==================')
            #     # print(len(tmp_pred_map_cls))
            #     # print(len(tmp_pred_map_cls[batch_idx]))
            #     boxes_this_image_gt_corners.append(gt_this_batch[sample_idx][box_idx][1])
            # break
            # boxes_this_image_gt_corners = np.array(boxes_this_image_gt_corners)
            #
            # if boxes_this_image_gt_corners.shape[0] > 0:
            #     print(boxes_this_image_gt.shape)
            #     boxes_this_image_gt_corners = boxes_this_image_gt_corners.copy()
            #     ############ save box points
            #     for idx_box in range(boxes_this_image_gt_corners.shape[0]):
            #         # box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
            #         box3d = flip_axis_to_depth(boxes_this_image_gt_corners[idx_box])
            #         pc_tmp, _ = extract_pc_in_box3d(tmp_point_clouds_np, box3d)
            #         write_ply_rgb(pc_tmp, out_dir + '/' + '%s_%d_pc.ply' % (im_name_tmp, idx_box))
            #     ############ save box points

            # # just comments for a while
            # if boxes_this_image_gt.shape[0] > 0:
            #     print(boxes_this_image_gt.shape)
            #     boxes_this_image_gt_tmp = boxes_this_image_gt.copy()
            #     boxes_this_image_gt_tmp[:, -1] *= -1
                ############ save box points
                # for idx_box in range(boxes_this_image_gt_tmp.shape[0]):
                #     x_min = boxes_this_image_gt_tmp[idx_box, 0] - boxes_this_image_gt_tmp[idx_box, 3] / 2.0
                #     x_max = boxes_this_image_gt_tmp[idx_box, 0] + boxes_this_image_gt_tmp[idx_box, 3] / 2.0
                #     y_min = boxes_this_image_gt_tmp[idx_box, 1] - boxes_this_image_gt_tmp[idx_box, 4] / 2.0
                #     y_max = boxes_this_image_gt_tmp[idx_box, 1] + boxes_this_image_gt_tmp[idx_box, 4] / 2.0
                #     z_min = boxes_this_image_gt_tmp[idx_box, 2] - boxes_this_image_gt_tmp[idx_box, 5] / 2.0
                #     z_max = boxes_this_image_gt_tmp[idx_box, 2] + boxes_this_image_gt_tmp[idx_box, 5] / 2.0
                #     x_logic = np.logical_and(tmp_point_clouds_np[:, 0] >= x_min, tmp_point_clouds_np[:, 0] <= x_max)
                #     y_logic = np.logical_and(tmp_point_clouds_np[:, 1] >= y_min, tmp_point_clouds_np[:, 1] <= y_max)
                #     z_logic = np.logical_and(tmp_point_clouds_np[:, 2] >= z_min, tmp_point_clouds_np[:, 2] <= z_max)
                #     full_logic = np.logical_and(np.logical_and(x_logic, y_logic), z_logic)
                #     tmp_box_point = tmp_point_clouds_np[full_logic]
                #     write_ply_rgb(tmp_box_point, out_dir + '/' + '%s_%d_pc.ply' % (im_name_tmp, idx_box))
                ############
                # write_oriented_bbox(boxes_this_image_gt_tmp, out_dir + '/' + '%s_gt.ply' % im_name_tmp)
                # # for idx_show_box in range(boxes_this_image_gt.shape[0]):
                # #     write_oriented_bbox(boxes_this_image_gt[idx_show_box,np.newaxis,:], out_dir + '/' + '%s_gt_%d.ply' % (im_name_tmp, idx_show_box))
                # #     tmp_box = boxes_this_image_gt[idx_show_box, np.newaxis, :]
                # #     tmp_box[:, :3] = tmp_box[:, :3]/2
                # #     write_oriented_bbox(tmp_box, out_dir + '/' + '%s_noise_%d.ply' % (im_name_tmp, idx_show_box))
                # print(out_dir + '/' + '%s_gt.ply' % im_name_tmp)

        # continue
        # print('333333333333333333333333333')
        # print(len(ap_calculator.pred_map_cls))
        # pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        # print('===========================')
        # print(len(pred_obb_this_batch))
        # output_nms = {}
        # output_nms['center_unnormalized'] = {}
        # output_nms['size_unnormalized'] = {}
        # output_nms['angle_continuous'] = {}
        # output_nms['class_max'] = {}
        # output_nms['sem_cls_prob'] = {}
        # output_nms['objectness_prob'] = {}
        # for idx_batch in range(len(pred_obb_this_batch)):
        #     output_nms['center_unnormalized'][idx_batch] = {}
        #     output_nms['size_unnormalized'][idx_batch] = {}
        #     output_nms['angle_continuous'][idx_batch] = {}
        #     output_nms['sem_cls_prob'][idx_batch] = {}
        #     output_nms['objectness_prob'][idx_batch] = {}
        #     for idx_box in range(len(pred_obb_this_batch[idx_batch])):
        #         output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
        #         output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
        #         output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
        #         output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:-1]
        #         output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][-1]
        #
        # # print('2222222222222222222222222')
        #
        # ### get the info
        # pred_prob_batch = outputs['outputs']['sem_cls_prob']
        # gt_classes_batch = batch_data_label['gt_box_sem_cls_label']
        # for idx_batch in range(len(pred_prob_batch)):
        #     # print(len(batch_data_label))
        #     # print(len(pred_prob_batch))
        #     for idx_box in range(len(pred_prob_batch[idx_batch])):
        #         pred_sem_cls_prob, pred_sem_cls = torch.max(pred_prob_batch[idx_batch][idx_box], -1)
        #         pred_sem_cls = int(pred_sem_cls.cpu().numpy())
        #         if args.if_clip_superset:
        #             box_class = dataconfig.superclass2type[pred_sem_cls]
        #         else:
        #             box_class = dataconfig.class2type[pred_sem_cls]
        #
        #         # tmp_gt_class = dataconfig.class2type[int(gt_classes_batch[idx_batch][idx_box])]
        #         # if box_class != tmp_gt_class:
        #         #     if (box_class,tmp_gt_class) in info_list.keys():
        #         #         info_list[(box_class,tmp_gt_class)] += 1
        #         #     else:
        #         #         info_list[(box_class, tmp_gt_class)] = 0
        #     # print(pred_prob_batch.shape)
        #     # print(gt_classes_batch.shape)
        #     # fsd
        # if if_after_nms:
        #     # print('1111111111111111111111111111111')
        #     camera_cord_to_image_plane(batch_data_label, output_nms, out_dir, args=args)
        # else:
        #     camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir, args=args)

        # print(info_list)
        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        # break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2

@torch.no_grad()
def save_seen(
        args,
        curr_epoch,
        model,
        criterion,
        dataset_config,
        dataset_loader,
        logger,
        curr_train_iter,
        out_dir,
        if_after_nms=True
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    dataconfig = SunrgbdDatasetConfig()

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    # info_list = {}
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            # print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #
        # }
        outputs = model(batch_data_label)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        # print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        im_name_list = batch_data_label['im_name']
        pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show_nms_then_iou_save_seen(
            outputs, batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)
        # print('44444444444444444444444')
        for sample_idx in range(len(point_cloud_rgb_this_batch)):
            im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            # if 'Data/sunrgb_d/sunrgbd_trainval/image/007426.jpg' == im_name_list[sample_idx]:
            #     print(1)


            pred_obb_this_image = pred_obb_this_batch[sample_idx]
            new_boxes = np.zeros((len(pred_obb_this_image), 8))
            new_box_name = im_name_tmp + '_novel_bbox.npy'
            for idx_box in range(len(pred_obb_this_image)):
                box_name = im_name_tmp + '_%04d_seen_bbox_feat_info.npy' % idx_box
                new_boxes[idx_box][:7] = pred_obb_this_image[idx_box][0][3][:7].cpu().numpy()
                new_boxes[idx_box][-1] = -1
                box_feat = pred_obb_this_image[idx_box][1]
                box_cls = pred_obb_this_image[idx_box][2]
                # box_info = np.concatenate()
                np.save(out_dir + '/' + box_name, np.array([box_feat.cpu().numpy(), box_cls], dtype=object))
            np.save(out_dir + '/' + new_box_name, new_boxes)
            # just comments for a while
        #     tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
        #     write_ply_rgb(tmp_point_clouds_np[:, 0:3], (tmp_point_clouds_np[:, 3:]*256).astype(np.int8), out_dir+'/'+'%s_pc.obj' % im_name_tmp)
        #
        #     boxes_this_image = []
        #     for box_idx in range(len(pred_obb_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy())
        #     # break
        #     boxes_this_image = np.array(boxes_this_image)
        #     # print(boxes_this_image.shape)
        #
        #     # just comments for a while
        #     if boxes_this_image.shape[0]>0:
        #         write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_pred.ply' % im_name_tmp)
        #
        #     boxes_this_image_gt = []
        #     for box_idx in range(len(gt_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
        #     # break
        #     boxes_this_image_gt = np.array(boxes_this_image_gt)
        #
        #     # just comments for a while
        #     if boxes_this_image_gt.shape[0] > 0:
        #         write_oriented_bbox(boxes_this_image_gt, out_dir+'/'+'%s_gt.ply' % im_name_tmp)
        #
        # # print('333333333333333333333333333')
        # # print(len(ap_calculator.pred_map_cls))
        # # pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        # # print('===========================')
        # # print(len(pred_obb_this_batch))
        # output_nms = {}
        # output_nms['center_unnormalized'] = {}
        # output_nms['size_unnormalized'] = {}
        # output_nms['angle_continuous'] = {}
        # output_nms['class_max'] = {}
        # output_nms['sem_cls_prob'] = {}
        # output_nms['objectness_prob'] = {}
        # for idx_batch in range(len(pred_obb_this_batch)):
        #     output_nms['center_unnormalized'][idx_batch] = {}
        #     output_nms['size_unnormalized'][idx_batch] = {}
        #     output_nms['angle_continuous'][idx_batch] = {}
        #     output_nms['sem_cls_prob'][idx_batch] = {}
        #     output_nms['objectness_prob'][idx_batch] = {}
        #     for idx_box in range(len(pred_obb_this_batch[idx_batch])):
        #         output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
        #         output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
        #         output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
        #         output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:-1]
        #         output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][-1]
        #
        # # print('2222222222222222222222222')
        #
        # ### get the info
        # pred_prob_batch = outputs['outputs']['sem_cls_prob']
        # gt_classes_batch = batch_data_label['gt_box_sem_cls_label']
        # for idx_batch in range(len(pred_prob_batch)):
        #     # print(len(batch_data_label))
        #     # print(len(pred_prob_batch))
        #     for idx_box in range(len(pred_prob_batch[idx_batch])):
        #         pred_sem_cls_prob, pred_sem_cls = torch.max(pred_prob_batch[idx_batch][idx_box], -1)
        #         pred_sem_cls = int(pred_sem_cls.cpu().numpy())
        #         box_class = dataconfig.class2type[pred_sem_cls]
        #         # tmp_gt_class = dataconfig.class2type[int(gt_classes_batch[idx_batch][idx_box])]
        #         # if box_class != tmp_gt_class:
        #         #     if (box_class,tmp_gt_class) in info_list.keys():
        #         #         info_list[(box_class,tmp_gt_class)] += 1
        #         #     else:
        #         #         info_list[(box_class, tmp_gt_class)] = 0
        #     # print(pred_prob_batch.shape)
        #     # print(gt_classes_batch.shape)
        #     # fsd
        # if if_after_nms:
        #     # print('1111111111111111111111111111111')
        #     camera_cord_to_image_plane(batch_data_label, output_nms, out_dir)
        # else:
        #     camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir)
    #
    #     # print(info_list)
        time_delta.update(time.time() - curr_time)
    #
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        # break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator

@torch.no_grad()
def save_novel(
        args,
        curr_epoch,
        model,
        criterion,
        dataset_config,
        dataset_loader,
        logger,
        curr_train_iter,
        out_dir,
        if_after_nms=True
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    dataconfig = SunrgbdDatasetConfig()

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    # info_list = {}
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            # print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #
        # }
        outputs = model(batch_data_label)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        # print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        im_name_list = batch_data_label['im_name']
        pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show_nms_then_iou(
            outputs, batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)
        # print('44444444444444444444444')
        for sample_idx in range(len(point_cloud_rgb_this_batch)):
            im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            # if 'Data/sunrgb_d/sunrgbd_trainval/image/007426.jpg' == im_name_list[sample_idx]:
            #     print(1)
            box_name = im_name_tmp + '_novel_bbox.npy'

            pred_obb_this_image = pred_obb_this_batch[sample_idx]
            new_boxes = np.zeros((len(pred_obb_this_image), 8))
            for idx_box in range(len(pred_obb_this_image)):
                new_boxes[idx_box][:7] = pred_obb_this_image[idx_box][3][:7].cpu().numpy()
                new_boxes[idx_box][-1] = -1
            np.save(out_dir + '/' + box_name, new_boxes)
            # just comments for a while
        #     tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
        #     write_ply_rgb(tmp_point_clouds_np[:, 0:3], (tmp_point_clouds_np[:, 3:]*256).astype(np.int8), out_dir+'/'+'%s_pc.obj' % im_name_tmp)
        #
        #     boxes_this_image = []
        #     for box_idx in range(len(pred_obb_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy())
        #     # break
        #     boxes_this_image = np.array(boxes_this_image)
        #     # print(boxes_this_image.shape)
        #
        #     # just comments for a while
        #     if boxes_this_image.shape[0]>0:
        #         write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_pred.ply' % im_name_tmp)
        #
        #     boxes_this_image_gt = []
        #     for box_idx in range(len(gt_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
        #     # break
        #     boxes_this_image_gt = np.array(boxes_this_image_gt)
        #
        #     # just comments for a while
        #     if boxes_this_image_gt.shape[0] > 0:
        #         write_oriented_bbox(boxes_this_image_gt, out_dir+'/'+'%s_gt.ply' % im_name_tmp)
        #
        # # print('333333333333333333333333333')
        # # print(len(ap_calculator.pred_map_cls))
        # # pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        # # print('===========================')
        # # print(len(pred_obb_this_batch))
        # output_nms = {}
        # output_nms['center_unnormalized'] = {}
        # output_nms['size_unnormalized'] = {}
        # output_nms['angle_continuous'] = {}
        # output_nms['class_max'] = {}
        # output_nms['sem_cls_prob'] = {}
        # output_nms['objectness_prob'] = {}
        # for idx_batch in range(len(pred_obb_this_batch)):
        #     output_nms['center_unnormalized'][idx_batch] = {}
        #     output_nms['size_unnormalized'][idx_batch] = {}
        #     output_nms['angle_continuous'][idx_batch] = {}
        #     output_nms['sem_cls_prob'][idx_batch] = {}
        #     output_nms['objectness_prob'][idx_batch] = {}
        #     for idx_box in range(len(pred_obb_this_batch[idx_batch])):
        #         output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
        #         output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
        #         output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
        #         output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:-1]
        #         output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][-1]
        #
        # # print('2222222222222222222222222')
        #
        # ### get the info
        # pred_prob_batch = outputs['outputs']['sem_cls_prob']
        # gt_classes_batch = batch_data_label['gt_box_sem_cls_label']
        # for idx_batch in range(len(pred_prob_batch)):
        #     # print(len(batch_data_label))
        #     # print(len(pred_prob_batch))
        #     for idx_box in range(len(pred_prob_batch[idx_batch])):
        #         pred_sem_cls_prob, pred_sem_cls = torch.max(pred_prob_batch[idx_batch][idx_box], -1)
        #         pred_sem_cls = int(pred_sem_cls.cpu().numpy())
        #         box_class = dataconfig.class2type[pred_sem_cls]
        #         # tmp_gt_class = dataconfig.class2type[int(gt_classes_batch[idx_batch][idx_box])]
        #         # if box_class != tmp_gt_class:
        #         #     if (box_class,tmp_gt_class) in info_list.keys():
        #         #         info_list[(box_class,tmp_gt_class)] += 1
        #         #     else:
        #         #         info_list[(box_class, tmp_gt_class)] = 0
        #     # print(pred_prob_batch.shape)
        #     # print(gt_classes_batch.shape)
        #     # fsd
        # if if_after_nms:
        #     # print('1111111111111111111111111111111')
        #     camera_cord_to_image_plane(batch_data_label, output_nms, out_dir)
        # else:
        #     camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir)
    #
    #     # print(info_list)
        time_delta.update(time.time() - curr_time)
    #
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        # break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator

@torch.no_grad()
def save_novel_with_class(
        args,
        curr_epoch,
        model,
        criterion,
        dataset_config,
        dataset_loader,
        logger,
        curr_train_iter,
        out_dir,
        if_after_nms=True
):
    print('Output dir:')
    print(out_dir)
    print('If after nms:')
    print(if_after_nms)

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    dataconfig = SunrgbdDatasetConfig()

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    # info_list = {}
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            # print(key)
            if not isinstance(batch_data_label[key], list):
                batch_data_label[key] = batch_data_label[key].to(net_device)
            else:
                batch_data_label[key] = batch_data_label[key]

        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_clouds_rgb": batch_data_label["point_clouds_rgb"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        #
        # }
        outputs = model(batch_data_label)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        # print(batch_data_label.keys())

        batch_data_label = all_gather_dict(batch_data_label)

        im_name_list = batch_data_label['im_name']
        pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show_nms_then_iou(
            outputs, batch_data_label)
        # pred_obb_this_batch, gt_this_batch, point_cloud_rgb_this_batch = ap_calculator.step_meter_show(outputs, batch_data_label)
        # print('44444444444444444444444')
        for sample_idx in range(len(point_cloud_rgb_this_batch)):
            im_name_tmp = os.path.basename(im_name_list[sample_idx])[:-4]
            # if 'Data/sunrgb_d/sunrgbd_trainval/image/007426.jpg' == im_name_list[sample_idx]:
            #     print(1)
            box_name = im_name_tmp + '_novel_bbox.npy'

            pred_obb_this_image = pred_obb_this_batch[sample_idx]
            new_boxes = []
            for idx_box in range(len(pred_obb_this_image)):
                max_prob_this_box, max_class_this_box = torch.max(pred_obb_this_image[idx_box][3][7:-1], -1)
                if max_prob_this_box < 0.5: # ignore some boxes which clip is not good at.
                    continue
                tmp_new_box = np.zeros((8))
                tmp_new_box[:7] = pred_obb_this_image[idx_box][3][:7].cpu().numpy()

                tmp_new_box[-1] = -1 * (max_class_this_box+10)
                new_boxes.append(tmp_new_box)
            if len(new_boxes) > 0:
                new_boxes = np.array(new_boxes)
            else:
                new_boxes = np.zeros((0, 8))

            np.save(out_dir + '/' + box_name, new_boxes)
            # just comments for a while
        #     tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
        #     write_ply_rgb(tmp_point_clouds_np[:, 0:3], (tmp_point_clouds_np[:, 3:]*256).astype(np.int8), out_dir+'/'+'%s_pc.obj' % im_name_tmp)
        #
        #     boxes_this_image = []
        #     for box_idx in range(len(pred_obb_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image.append(pred_obb_this_batch[sample_idx][box_idx][3].cpu().numpy())
        #     # break
        #     boxes_this_image = np.array(boxes_this_image)
        #     # print(boxes_this_image.shape)
        #
        #     # just comments for a while
        #     if boxes_this_image.shape[0]>0:
        #         write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_pred.ply' % im_name_tmp)
        #
        #     boxes_this_image_gt = []
        #     for box_idx in range(len(gt_this_batch[sample_idx])):
        #         # print('==================')
        #         # print(len(tmp_pred_map_cls))
        #         # print(len(tmp_pred_map_cls[batch_idx]))
        #         boxes_this_image_gt.append(gt_this_batch[sample_idx][box_idx][2])
        #     # break
        #     boxes_this_image_gt = np.array(boxes_this_image_gt)
        #
        #     # just comments for a while
        #     if boxes_this_image_gt.shape[0] > 0:
        #         write_oriented_bbox(boxes_this_image_gt, out_dir+'/'+'%s_gt.ply' % im_name_tmp)
        #
        # # print('333333333333333333333333333')
        # # print(len(ap_calculator.pred_map_cls))
        # # pred_obb_this_batch = pred_obb_this_batch[batch_idx*8:(batch_idx+1)*8]
        # # print('===========================')
        # # print(len(pred_obb_this_batch))
        # output_nms = {}
        # output_nms['center_unnormalized'] = {}
        # output_nms['size_unnormalized'] = {}
        # output_nms['angle_continuous'] = {}
        # output_nms['class_max'] = {}
        # output_nms['sem_cls_prob'] = {}
        # output_nms['objectness_prob'] = {}
        # for idx_batch in range(len(pred_obb_this_batch)):
        #     output_nms['center_unnormalized'][idx_batch] = {}
        #     output_nms['size_unnormalized'][idx_batch] = {}
        #     output_nms['angle_continuous'][idx_batch] = {}
        #     output_nms['sem_cls_prob'][idx_batch] = {}
        #     output_nms['objectness_prob'][idx_batch] = {}
        #     for idx_box in range(len(pred_obb_this_batch[idx_batch])):
        #         output_nms['center_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][:3]
        #         output_nms['size_unnormalized'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][3:6]
        #         output_nms['angle_continuous'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][6]
        #         output_nms['sem_cls_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][7:-1]
        #         output_nms['objectness_prob'][idx_batch][idx_box] = pred_obb_this_batch[idx_batch][idx_box][3][-1]
        #
        # # print('2222222222222222222222222')
        #
        # ### get the info
        # pred_prob_batch = outputs['outputs']['sem_cls_prob']
        # gt_classes_batch = batch_data_label['gt_box_sem_cls_label']
        # for idx_batch in range(len(pred_prob_batch)):
        #     # print(len(batch_data_label))
        #     # print(len(pred_prob_batch))
        #     for idx_box in range(len(pred_prob_batch[idx_batch])):
        #         pred_sem_cls_prob, pred_sem_cls = torch.max(pred_prob_batch[idx_batch][idx_box], -1)
        #         pred_sem_cls = int(pred_sem_cls.cpu().numpy())
        #         box_class = dataconfig.class2type[pred_sem_cls]
        #         # tmp_gt_class = dataconfig.class2type[int(gt_classes_batch[idx_batch][idx_box])]
        #         # if box_class != tmp_gt_class:
        #         #     if (box_class,tmp_gt_class) in info_list.keys():
        #         #         info_list[(box_class,tmp_gt_class)] += 1
        #         #     else:
        #         #         info_list[(box_class, tmp_gt_class)] = 0
        #     # print(pred_prob_batch.shape)
        #     # print(gt_classes_batch.shape)
        #     # fsd
        # if if_after_nms:
        #     # print('1111111111111111111111111111111')
        #     camera_cord_to_image_plane(batch_data_label, output_nms, out_dir)
        # else:
        #     camera_cord_to_image_plane(batch_data_label, outputs['outputs'], out_dir)
    #
    #     # print(info_list)
        time_delta.update(time.time() - curr_time)
    #
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        # break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
    if_real_test=False,
    if_cmp_class=False,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
        args=args
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()

    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    all_time = 0
    cnt_time = 0
    cnt_clip_time = 0
    for batch_idx, batch_data_label in enumerate(dataset_loader):

        curr_time = time.time()
        for key in batch_data_label:
            if isinstance(batch_data_label[key], torch.Tensor):
        #        print(batch_data_label[key].type)
                batch_data_label[key] = batch_data_label[key].to(net_device)
        # inputs = {
        #     "point_clouds": batch_data_label["point_clouds"],
        #     "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
        #     "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        # }

        outputs = model(batch_data_label, if_real_test=if_real_test, if_cmp_class=if_cmp_class)

        # ############test time############
        # # tmp_cpu = batch_data_label["point_clouds"].cpu()
        # torch.cuda.synchronize()
        # test_time_begin = time.time()
        # outputs, clip_time = model(batch_data_label, if_test=True)
        # # tmp_cpu = outputs["outputs"]["sem_cls_prob"].cpu()
        # torch.cuda.synchronize()
        # all_time += time.time() - test_time_begin
        # cnt_clip_time += clip_time
        # cnt_time += 1
        # print(all_time)
        # if cnt_time > 0 and (cnt_time % 10)==0:
        #     print('all time:')
        #     print(all_time/10)
        #     print('clip_time:')
        #     print(cnt_clip_time/10)
        # ###########test time############

        time_delta.update(time.time() - curr_time)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)

            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])

        batch_data_label = all_gather_dict(batch_data_label)

        ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)

        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg

        curr_iter += 1
        # if curr_iter > 1:
        #     break

    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    return ap_calculator
