# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import argparse
import os
import sys
import pickle

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_CPP_LOG_LEVEL"]= "INFO"
import torch

from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import evaluate, train_one_epoch, show_boxes, save_box_points, crop_image, calculate_wrong_class, save_novel, save_seen, save_novel_with_class
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger
from utils.votenet_pc_util import write_oriented_bbox, write_ply, write_ply_rgb
from pandas import DataFrame

here_path = os.getcwd() 
if here_path.find('user-job-dir')!=-1:
    os.environ['TORCH_HOME'] = '/cache/data/pretrained_models/imagenet/'




def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--former_prompt_len", default=3, type=int)
    parser.add_argument("--later_prompt_len", default=3, type=int)
    parser.add_argument("--distillation_box_num", default=32, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr", "3detrclip", "3detr3d2dcorrelationdecoder", "3detr3d2dcorrelationstepbystep", "3detr3d2dcorrelationencoder", "3detr3d2dcorrelationclipencoder", "3detr3d2dcorrelationclipencodereverylayer", "3detr3d2dcorrelationfreezeclipencodereverylayer", "3detr3d2dcorrelationfreezeclipencodereverylayermulticlasshead", "3detrmulticlasshead", "3detr3d2dcorrelationvitencodereverylayermulticlasshead", "3detr3d2dcorrelationinvitencodermulticlasshead", "3detr3d2dtwomodalinvitencodermulticlasshead", "3detr3d2dtextresnetencodermulticlasshead", "3detr3d2dtextresnetfollowtheencodermulticlasshead", "3detr3d2dtextresnetencoder_assignconcat_multiclasshead", "3detr3d2dtextresnetencoder_assignconcat_beforeencoder_multiclasshead", "3detr3d2dtextresnet_fpn_encoder_assignconcat_beforeencoder_multiclasshead", "3detr3d2dtext_shallow_clip_resnet_encoder_assignconcat_beforeencoder_multiclasshead", "3detr3d2dtext_shallow_clip_resnet_encoder_assignconcat_afterencoder_multiclasshead", "3detr3d2dtextresnetencoder_assignconcat_decoder_correlation", "3detr3d2dtextresnetencoder_assignconcat_decoder_3Dto2Dtext_correlation", "3detr3d2d_superclasstext_resnetencoder_assignconcat_decoder_3Dto2Dtext_correlation", "3detr3d2d_superclasstext_resnetencoder_assignconcat_decoder_3Dto2Dtext_correlation_batchwise_contrast", "3detr3d2d_superclasstext_resnetencoder_assignconcat_decoder_3Dto2Dtext_correlation_multiclass_attention", "image_multiclass_attention", "image_multiclass_tokens", "image_multiclass_tokens_clip_trainable", "3detr_assignconcat_predictedbox_distillation", "3detr_predictedbox_distillation", "3detr_assignconcat_predictedbox_distillation_only_last_layer", "3detr_3d_predicted_box_distill_2d", "3detr_3d_predicted_box_distill_2d_with_shared_attention", "3detr_3d_predicted_box_distill_2d_with_guided_shared_attention", "3detr_predictedbox_distillation_resnet", "3detrmulticlasshead_likem1n2o1", "3detr_predictedbox_distillation_no_objectness", "clip_prompt", "model_pass"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--if_add_norm", default=False, action="store_true")
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--nms_iou_keep", default=0.25, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--if_clip_more_prompts", default=False, action="store_true")
    parser.add_argument("--if_clip_superset", default=False, action="store_true")
    parser.add_argument("--if_clip_weak_labels", default=False, action="store_true")
    parser.add_argument("--if_accumulate_former_pseudo_labels", default=False, action="store_true")
    parser.add_argument("--if_only_novel_prompt", default=False, action="store_true")
    parser.add_argument("--if_only_seen_in_loss", default=False, action="store_true")
    parser.add_argument("--only_image_class", default=False, action="store_true")
    parser.add_argument("--if_clip_trainable", default=False, action="store_true")
    parser.add_argument("--if_adopt_2d_box_iou_supervision", default=False, action="store_true")
    parser.add_argument("--box2d_iou_thres", default=1, type=float)
    parser.add_argument("--box2d_gt_score_thres", default=0, type=float)
    parser.add_argument("--iou_match_thres", default=0.25, type=float)
    parser.add_argument("--if_skip_no_seen_scene_objectness", default=False, action="store_true")
    parser.add_argument("--online_nms_update_novel_label", default=False, action="store_true")
    parser.add_argument("--online_nms_update_novel_label_for_objectness", default=False, action="store_true")
    parser.add_argument("--online_nms_update_novel_label_for_clip_driven_objectness", default=False, action="store_true")
    parser.add_argument("--online_nms_update_novel_label_for_objectness_with_max_number", default=False, action="store_true")
    parser.add_argument("--online_nms_update_accumulate_novel_label", default=False, action="store_true")
    parser.add_argument("--online_nms_update_accumulate_epoch", default=10, type=int)
    parser.add_argument("--online_nms_update_save_novel_label", default=False, action="store_true")
    parser.add_argument("--online_nms_update_save_novel_label_clip_driven", default=False, action="store_true")
    parser.add_argument("--online_nms_update_save_novel_label_clip_driven_with_cate_confidence", default=False, action="store_true")
    parser.add_argument("--online_nms_update_save_novel_label_clip_driven_with_cate_confidence_2d_box", default=False, action="store_true")
    parser.add_argument("--online_nms_update_save_novel_label_clip_driven_with_cate_confidence_iou_match_weakly", default=False, action="store_true")
    parser.add_argument("--if_distill_also_match", default=False, action="store_true")
    parser.add_argument("--online_nms_update_save_novel_label_with_prob", default=False, action="store_true")
    parser.add_argument("--begin_keep_epoch", default=100000000000000, type=int)
    parser.add_argument("--clip_driven_keep_thres", default=1000000.0, type=float)

    parser.add_argument("--online_nms_update_save_epoch", default=10, type=int)
    parser.add_argument("--online_nms_update_max_num_epoch", default=10, type=int)
    parser.add_argument("--conclusion_thres", default=1000, type=int)

    parser.add_argument("--cross_enc_nlayers", default=6, type=int)
    parser.add_argument("--cross_enc_dim", default=512, type=int)
    parser.add_argument("--cross_enc_ffn_dim", default=128, type=int)
    parser.add_argument("--cross_enc_dropout", default=0.1, type=float)
    parser.add_argument("--cross_enc_nhead", default=4, type=int)
    parser.add_argument("--repeat_time", default=2, type=int)


    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)
    parser.add_argument("--cross_enc_activation", default="relu", type=str)
    parser.add_argument("--cross_num_layers", default=1, type=int)
    parser.add_argument("--cross_heads", default=8, type=int)
    parser.add_argument("--every_number", default=4, type=int)
    parser.add_argument("--trans_layer_numbers", default=3, type=int)
    parser.add_argument("--trans_head_numbers", default=4, type=int)
    parser.add_argument("--eval_layer_id", default=-1, type=int)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--if_use_v1", default=False, action="store_true")
    parser.add_argument("--on_cloud", default=True, action="store_false")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_contrast_object_text", default=0, type=float)
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_region_embed_weight", default=0, type=float)
    parser.add_argument("--loss_3d_2d_region_embed_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_softmax_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_softmax_discovery_novel_objectness_weight", default=0, type=float)
    parser.add_argument("--loss_sem_focal_cls_weight", default=0, type=float)
    parser.add_argument("--loss_contrastive_weight", default=0, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)
    parser.add_argument("--loss_contrast_3dto2d_text_weight", default=0, type=float)
    parser.add_argument("--loss_image_seen_class_weight", default=0, type=float)
    parser.add_argument("--loss_batchwise_contrastive_weight", default=0, type=float)
    parser.add_argument("--loss_predicted_region_embed_l1_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_softmax_2d_box_iou_supervised_skip_none_gt_sample_weight", default=0, type=float)
    parser.add_argument("--loss_predicted_region_embed_l1_only_last_layer_weight", default=0, type=float)
    parser.add_argument("--loss_predicted_region_embed_cos_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_sigmoid_loss_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_softmax_loss_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_softmax_weakly_loss_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_softmax_weakly_loss_with_novel_cate_confi_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_softmax_iou_match_weakly_loss_with_novel_cate_confi_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_softmax_loss_with_novel_cate_confi_weight", default=0, type=float)
    parser.add_argument("--loss_feat_seen_sigmoid_with_full_image_loss_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_softmax_skip_none_gt_sample_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_softmax_skip_none_gt_sample_en_discovery_objectness_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_softmax_skip_none_gt_sample_keep_discovery_objectness_weight", default=0, type=float)
    parser.add_argument("--loss_prompt_softmax_weight", default=0, type=float)
    parser.add_argument("--loss_prompt_sigmoid_weight", default=0, type=float)
    parser.add_argument("--keep_objectness", default=0.5, type=float)
    parser.add_argument("--clip_with_objectness", default=-1, type=float)
    parser.add_argument("--save_objectness", default=0.75, type=float)
    parser.add_argument(
        "--confidence_type", default="clip-max-prob", choices=["clip-max-prob", "non-confidence", "objectness", "clip+objectness"]
    )
    parser.add_argument(
        "--confidence_type_in_datalayer", default="clip-max-prob", choices=["clip-max-prob", "zero-out", "objectness", "clip+objectness", "weight_one"]
    )
    parser.add_argument(
        "--loss_no_object_contrast_weight", default=0.05, type=float
    )  # "no object" or "background" class for detection


    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "scannet_anonymous", "scannet_image", "scannet_anonymous_aligned_image", "sunrgbd", "sunrgbd_image", "sunrgbd_anonymous", "sunrgbd_anonymous_image", "sunrgbd_anonymous_aligned_image", "sunrgbd_anonymous_aligned_image_with_novel", "sunrgbd_anonymous_aligned_image_with_novel_three_classes", "sunrgbd_anonymous_aligned_image_with_novel_repeated_novel_scene", "sunrgbd_anonymous_aligned_image_with_novel_with_class", "sunrgbd_anonymous_aligned_image_with_novel_with_prob", "sunrgbd_seen_features", "sunrgbd_anonymous_aligned_image_2d_boxes", "sunrgbd_anonymous_aligned_image_with_novel_cate_confi", "scannet_anonymous_aligned_image_with_novel_cate_confi", "model_pass_data", "scannet50_image", "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_with_2d_boxes", 'model_pass_scannet', "sunrgbd_anonymous_aligned_image_object_aug"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--dataset_num_workers_test", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)
    parser.add_argument("--batchsize_per_gpu_test", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--real_eval_every_epoch", default=10, type=int)
    parser.add_argument("--real_cmp_eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_semcls", default=238, type=int)
    parser.add_argument("--test_num_semcls", default=37, type=int)
    parser.add_argument("--train_range_min", default=0, type=int)
    parser.add_argument("--train_range_max", default=96, type=int)
    parser.add_argument("--test_range_min", default=0, type=int)
    parser.add_argument("--test_range_max", default=238, type=int)
    parser.add_argument("--reset_scannet_num", default=50, type=int)
    parser.add_argument("--train_range_list", default=-1, nargs='+', type=int)
    parser.add_argument("--test_range_list", default=-1, nargs='+', type=int)
    parser.add_argument("--if_image_augment", default=False, type=bool)
    parser.add_argument("--pseudo_setting", default='None', type=str)
    parser.add_argument("--pooling_methods", required=False, type=str, choices=["average", "max"])
    parser.add_argument("--if_with_fake_classes", default=False,  action="store_true")
        ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--minitest_only", default=False, action="store_true")
    parser.add_argument("--show_only", default=False, action="store_true")
    parser.add_argument("--show_box_points", default=False, action="store_true")
    parser.add_argument("--cal_class_only", default=False, action="store_true")
    parser.add_argument("--save_novel_only", default=False, action="store_true")
    parser.add_argument("--save_novel_with_class_only", default=False, action="store_true")
    parser.add_argument("--save_seen_feat_only", default=False, action="store_true")
    parser.add_argument("--only_prompt_loss", default=False, action="store_true")
    parser.add_argument("--save_novel_dir", default=None, type=str)
    parser.add_argument("--save_seen_dir", default=None, type=str)
    parser.add_argument("--prompt_embedding_dir", default=None, type=str)
    parser.add_argument("--if_reset_epoch_periodically", default=False, action="store_true")
    parser.add_argument("--reset_epoch_periodically", default=60, type=int)


    parser.add_argument("--if_input_image", default=False, action="store_true")
    parser.add_argument("--if_with_clip", default=False, action="store_true")
    parser.add_argument("--if_adopt_region_embed", default=False, action="store_true")
    parser.add_argument("--if_with_clip_embed", default=False, action="store_true")
    parser.add_argument("--if_use_gt_box", default=False, action="store_true")
    parser.add_argument("--if_after_nms", default=False, action="store_true")
    parser.add_argument("--if_expand_box", default=False, action="store_true")
    parser.add_argument("--if_keep_box", default=False, action="store_true")
    parser.add_argument("--if_online_keep_max_box_number", default=False, action="store_true")
    parser.add_argument("--if_with_larger_embedding", default=False, action="store_true")
    parser.add_argument("--if_select_box_by_objectness", default=False, action="store_true")
    parser.add_argument("--if_concat_transformer", default=False, action="store_true")
    parser.add_argument("--if_clip_text_only_seen", default=False, action="store_true")
    parser.add_argument("--show_dir", default=None, type=str)
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--crop_only", default=False, action="store_true")
    parser.add_argument("--if_detach_the_guidence_attention", default=False, action="store_true")
    parser.add_argument("--crop_dir", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--checkpoint_file", default=None, type=str)
    parser.add_argument("--log_file", default='./log.lst', type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--set_epoch", default=-1, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)
    # parser.add_argument("--image_size", default=[730, 531], type=list)
    parser.add_argument("--image_size_width", default=730, type=int)
    parser.add_argument("--image_size_height", default=531, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser


def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion,
    dataset_config,
    dataloaders,
    best_val_metrics,
    real_test_dataset_config,
    real_cmp_test_dataset_config
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    logger = Logger(args.checkpoint_dir+'/log/')

    os.makedirs(args.checkpoint_dir, exist_ok=True)



    reset_epoch_cnt = 0
    epoch = args.start_epoch
    while(epoch < args.max_epoch):


        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)

        all_epoch = epoch+(reset_epoch_cnt*args.reset_epoch_periodically)

        if all_epoch == args.begin_keep_epoch:
            model.if_keep_box = True
            print('begin to keep boxes from epoch %d: ' % all_epoch)
            print(model.if_keep_box)

    # # -----------------just see eval-------------------
    #     curr_iter = epoch * len(dataloaders["train"])
    #     ap_calculator = evaluate(
    #         args,
    #         epoch,
    #         model,
    #         None,
    #         real_test_dataset_config,
    #         dataloaders["real_test"],
    #         logger,
    #         curr_iter,
    #         if_real_test = True
    #     )
    #     print('============1==============')
    #     metrics = ap_calculator.compute_metrics()
    #     print('============2==============')
    #     ap25 = metrics[0.25]["mAP"]
    #     print('============3==============')
    #     metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
    #     print('============4==============')
    #     metrics_dict = ap_calculator.metrics_to_dict(metrics)
    #     if is_primary():
    #         print("==" * 10)
    #         print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
    #         print("==" * 10)
    #         logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")
    # # -----------------just see eval-------------------

        aps = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            dataloaders["train"],
            logger,
        )

        if args.online_nms_update_novel_label_for_objectness_with_max_number and (
                (epoch % args.online_nms_update_max_num_epoch) == 0):
            if args.ngpus > 1:  # git the dic from all the gpus
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
            if is_primary():
                print('===============update and save box_number_dict==================')
                max_box_number_dir = args.checkpoint_dir + '/max_novel_dir/'
                os.makedirs(max_box_number_dir, exist_ok=True)
                print(model.module.max_box_number_dict)
                np.save(max_box_number_dir+'%04d.npy' % all_epoch, model.module.max_box_number_dict)

        curr_iter = epoch * len(dataloaders["train"])



        if epoch % 30 == 0:
            if not args.only_prompt_loss:
                metrics = aps.compute_metrics()
                metric_str = aps.metrics_to_str(metrics, per_class=False)
                metrics_dict = aps.metrics_to_dict(metrics)

            
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
                filename="checkpoint.pth",
            )

            
            if (not args.only_prompt_loss) and is_primary():
                print("==" * 10)
                print(f"Epoch [{all_epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)
                logger.log_scalars(metrics_dict, curr_iter, prefix="Train/")


        if (
                args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                all_epoch,
                args,
                best_val_metrics,
            )

            
        if (epoch % args.eval_every_epoch  == 0 and epoch > 0) or epoch == (args.max_epoch - 1):
            print('-----------begin to evaluate')
            ap_calculator = evaluate(
                args,
                epoch,
                model,
                criterion,
                dataset_config,
                dataloaders["test"],
                logger,
                curr_iter,
            )
            metrics = ap_calculator.compute_metrics()
            ap25 = metrics[0.25]["mAP"]
            metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ap_calculator.metrics_to_dict(metrics)
            if is_primary():
                print("==" * 10)
                print(f"Evaluate Epoch [{all_epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)
                logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")

            if is_primary() and (
                len(best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{all_epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25}"
                )
            

        if (not args.only_prompt_loss) and (((all_epoch % args.real_eval_every_epoch == 0 )) or all_epoch == (args.max_epoch - 1)) and (all_epoch > 0):
        # if (all_epoch % args.real_eval_every_epoch == 0) or all_epoch == (args.max_epoch - 1):
            ap_calculator = evaluate(
                args,
                all_epoch,
                model,
                None,
                real_test_dataset_config,
                dataloaders["real_test"],
                logger,
                curr_iter,
                if_real_test = True
            )

            metrics = ap_calculator.compute_metrics()
            metric_str = ap_calculator.metrics_to_str(metrics)

            log_file = args.checkpoint_dir + '/eval_%04d.lst' % all_epoch
            with open(log_file, 'w') as f:
                f.write(metric_str)

            if is_primary() and args.on_cloud:
                print("==" * 10)
                print(f"Test model; Metrics {metric_str}")
                print("==" * 10)

        if (not args.only_prompt_loss) and (((all_epoch % args.real_cmp_eval_every_epoch == 0 )) or all_epoch == (args.max_epoch - 1)) and (all_epoch > 0):
        # if (all_epoch % args.real_eval_every_epoch == 0) or all_epoch == (args.max_epoch - 1):
            ap_calculator = evaluate(
                args,
                all_epoch,
                model,
                None,
                real_cmp_test_dataset_config,
                dataloaders["real_cmp_test"],
                logger,
                curr_iter,
                if_cmp_class=True
            )

            metrics = ap_calculator.compute_metrics()
            if is_primary():
                # print(metrics[0.25])
                met_dict = {i: metrics[0.25][i] for i in metrics[0.25].keys()}
                metrics_excel = DataFrame(data=met_dict, index=[1])
                # print(1)
                metrics_excel.to_excel(args.checkpoint_dir+'/cmp_eval_%04d_025.xlsx' % all_epoch)
                # metric_str = ap_calculator.metrics_to_str(metrics)

                met_dict = {i: metrics[0.5][i] for i in metrics[0.5].keys()}
                metrics_excel = DataFrame(data=met_dict, index=[1])
                # print(1)
                metrics_excel.to_excel(args.checkpoint_dir+'/cmp_eval_%04d_05.xlsx' % all_epoch)
                metric_str = ap_calculator.metrics_to_str(metrics)

                log_file = args.checkpoint_dir + '/cmp_eval_%04d.lst' % all_epoch
                with open(log_file, 'w') as f:
                    f.write(metric_str)

            if is_primary() and args.on_cloud:
                print("==" * 10)
                print(f"Test model; Metrics {metric_str}")
                print("==" * 10)

        if args.if_reset_epoch_periodically:
            if epoch % args.reset_epoch_periodically == 0 and epoch > 0:
                print('Reset the epoch = 0 and train the model from beginning')
                epoch = 0
                reset_epoch_cnt += 1
                continue

        epoch += 1


    # always evaluate last checkpoint
    epoch = args.max_epoch - 1
    curr_iter = epoch * len(dataloaders["train"])
    ap_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )

    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    # latest checkpoint is always stored in checkpoint.pth
    save_checkpoint(
        args.checkpoint_dir,
        model_no_ddp,
        optimizer,
        epoch,
        args,
        best_val_metrics,
        filename="last_checkpoint.pth",
    )


    if is_primary():
        print("==" * 10)
        print(f"Evaluate Final [{all_epoch}/{args.max_epoch}]; Metrics {metric_str}")
        print("==" * 10)

        with open(final_eval, "w") as fh:
            fh.write("Training Finished.\n")
            fh.write("==" * 10)
            fh.write("Final Eval Numbers.\n")
            fh.write(metric_str)
            fh.write("\n")
            fh.write("==" * 10)
            fh.write("Best Eval Numbers.\n")
            fh.write(ap_calculator.metrics_to_str(best_val_metrics))
            fh.write("\n")

        with open(final_eval_pkl, "wb") as fh:
            pickle.dump(metrics, fh)

def crop_boxes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator = crop_image(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )
#   tmp_gt_map_cls = ap_calculator.gt_map_cls
#   tmp_pred_map_cls = ap_calculator.pred_map_cls
#   tmp_point_clouds = ap_calculator.point_clouds
#   #print(len(tmp_gt_map_cls)) # batch size's image
#   #print(len(tmp_pred_map_cls)) # batch size's image
#   #print(len(tmp_gt_map_cls[0])) # object num of gt
#   #print(len(tmp_gt_map_cls[0][0])) # list(class id, box array)
#   #print(tmp_gt_map_cls[0][0][0]) # int number, class
#   #print(tmp_gt_map_cls[0][0][1].shape) # box array (8, 3)
#   #print(len(tmp_pred_map_cls[0])) # object number of prediction
#   #print(len(tmp_pred_map_cls[0])) # list(class id, box array, )
#   #print(tmp_pred_map_cls[0][0][0]) # int number, class
#   #print(tmp_pred_map_cls[0][0][1].shape) # box array (8, 3)
#   #print(tmp_pred_map_cls[0][0][2]) # object prob
#   #print(tmp_pred_map_cls[0][0][3].shape)
#   #print('------------')
#   #print(len(tmp_point_clouds))
#   #print(len(tmp_point_clouds[0]))
#   #print(tmp_point_clouds[0].shape)
#   for batch_idx in range(len(tmp_pred_map_cls)):
#       #print(tmp_point_clouds[batch_idx].shape)
#       tmp_point_clouds_np = tmp_point_clouds[batch_idx].cpu().numpy()
#       write_ply_rgb(tmp_point_clouds_np[:, 0:3], (tmp_point_clouds_np[:, 3:]*256).astype(np.int8), out_dir+'/'+'%s_pc.obj' % batch_idx)
#       #break
#   for batch_idx in range(len(tmp_pred_map_cls)):
#       boxes_this_image = []
#       for box_idx in range(len(tmp_pred_map_cls[batch_idx])):
#           #print('==================')
#           #print(len(tmp_pred_map_cls))
#           #print(len(tmp_pred_map_cls[batch_idx]))
#           boxes_this_image.append(tmp_pred_map_cls[batch_idx][box_idx][3].cpu().numpy())
#           #break
#       boxes_this_image = np.array(boxes_this_image)
#       #print(boxes_this_image.shape)
#       write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_pred.ply' % batch_idx)
#       print(out_dir+'/'+'%s_pred.ply' % batch_idx)
#       #break
#       
#   for batch_idx in range(len(tmp_gt_map_cls)):
#       boxes_this_image = []
#       for box_idx in range(len(tmp_gt_map_cls[batch_idx])):
#           #print('==================')
#           #print(len(tmp_pred_map_cls))
#           #print(len(tmp_pred_map_cls[batch_idx]))
#           boxes_this_image.append(tmp_gt_map_cls[batch_idx][box_idx][2])
#           #break
#       boxes_this_image = np.array(boxes_this_image)
#       #print(boxes_this_image.shape)
#       write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_gt.ply' % batch_idx)
#       #break


def pred_boxes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # print('before show_boxes')
    ap_calculator = show_boxes(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )

def pred_box_points(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # print('before show_boxes')
    ap_calculator = save_box_points(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )


def save_novel_boxes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # print('before show_boxes')
    ap_calculator = save_novel(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )

def save_novel_boxes_with_class(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # print('before show_boxes')
    ap_calculator = save_novel_with_class(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )


def save_seen_feat(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # print('before show_boxes')
    ap_calculator = save_seen(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )

def cal_classes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, out_dir, if_after_nms):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    # print('before show_boxes')
    ap_calculator = calculate_wrong_class(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
        out_dir,
        if_after_nms
    )
#   tmp_gt_map_cls = ap_calculator.gt_map_cls
#   tmp_pred_map_cls = ap_calculator.pred_map_cls
#   tmp_point_clouds = ap_calculator.point_clouds
#   #print(len(tmp_gt_map_cls)) # batch size's image
#   #print(len(tmp_pred_map_cls)) # batch size's image
#   #print(len(tmp_gt_map_cls[0])) # object num of gt
#   #print(len(tmp_gt_map_cls[0][0])) # list(class id, box array)
#   #print(tmp_gt_map_cls[0][0][0]) # int number, class
#   #print(tmp_gt_map_cls[0][0][1].shape) # box array (8, 3)
#   #print(len(tmp_pred_map_cls[0])) # object number of prediction
#   #print(len(tmp_pred_map_cls[0])) # list(class id, box array, )
#   #print(tmp_pred_map_cls[0][0][0]) # int number, class
#   #print(tmp_pred_map_cls[0][0][1].shape) # box array (8, 3)
#   #print(tmp_pred_map_cls[0][0][2]) # object prob
#   #print(tmp_pred_map_cls[0][0][3].shape)
#   #print('------------')
#   #print(len(tmp_point_clouds))
#   #print(len(tmp_point_clouds[0]))
#   #print(tmp_point_clouds[0].shape)
#   for batch_idx in range(len(tmp_pred_map_cls)):
#       #print(tmp_point_clouds[batch_idx].shape)
#       tmp_point_clouds_np = tmp_point_clouds[batch_idx].cpu().numpy()
#       write_ply_rgb(tmp_point_clouds_np[:, 0:3], (tmp_point_clouds_np[:, 3:]*256).astype(np.int8), out_dir+'/'+'%s_pc.obj' % batch_idx)
#       #break
#   for batch_idx in range(len(tmp_pred_map_cls)):
#       boxes_this_image = []
#       for box_idx in range(len(tmp_pred_map_cls[batch_idx])):
#           #print('==================')
#           #print(len(tmp_pred_map_cls))
#           #print(len(tmp_pred_map_cls[batch_idx]))
#           boxes_this_image.append(tmp_pred_map_cls[batch_idx][box_idx][3].cpu().numpy())
#           #break
#       boxes_this_image = np.array(boxes_this_image)
#       #print(boxes_this_image.shape)
#       write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_pred.ply' % batch_idx)
#       print(out_dir+'/'+'%s_pred.ply' % batch_idx)
#       #break
#       
#   for batch_idx in range(len(tmp_gt_map_cls)):
#       boxes_this_image = []
#       for box_idx in range(len(tmp_gt_map_cls[batch_idx])):
#           #print('==================')
#           #print(len(tmp_pred_map_cls))
#           #print(len(tmp_pred_map_cls[batch_idx]))
#           boxes_this_image.append(tmp_gt_map_cls[batch_idx][box_idx][2])
#           #break
#       boxes_this_image = np.array(boxes_this_image)
#       #print(boxes_this_image.shape)
#       write_oriented_bbox(boxes_this_image, out_dir+'/'+'%s_gt.ply' % batch_idx)
#       #break


def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders, log_file=None, if_mini=False, real_cmp_test_dataset_config=None):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"], strict=False)
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    if if_mini:
        ap_calculator = evaluate(
            args,
            epoch,
            model,
            criterion,
            dataset_config,
            dataloaders["minitest"],
            logger,
            curr_iter,
            if_real_test=True
        )
    else:
        ap_calculator = evaluate(
            args,
            epoch,
            model,
            criterion,
            dataset_config,
            dataloaders["test"],
            logger,
            curr_iter,
            if_real_test=True
        )
        # ap_calculator = evaluate(
        #     args,
        #     epoch,
        #     model,
        #     None,
        #     real_cmp_test_dataset_config,
        #     dataloaders["real_cmp_test"],
        #     logger,
        #     curr_iter,
        #     if_cmp_class=True
        # )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if log_file != None:
        print('The log is saved in %s' % log_file)
        with open(log_file, 'w') as f:
            f.write(metric_str)

    else:
        print('Do not save log_file')
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)


def main(local_rank, args):
    if local_rank == 0:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
    if args.ngpus > 1:
        print("Using GPU: {}".format(local_rank))
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    datasets, dataset_config, real_test_dataset_config, real_cmp_test_dataset_config = build_dataset(args)
    model, _ = build_model(args, dataset_config)
    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] , find_unused_parameters=False
        )
    criterion = build_criterion(args, dataset_config)
    criterion = criterion.cuda(local_rank)


    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test","real_test", "real_cmp_test"]
    elif args.minitest_only:
        dataset_splits = ["minitest"]
    else:
        dataset_splits = ["train", "test", "real_test", "real_cmp_test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])
        if split == "train":
            tmp_bs_per_gpu = args.batchsize_per_gpu
            tmp_num_work = args.dataset_num_workers
        else:
            tmp_bs_per_gpu = args.batchsize_per_gpu_test
            tmp_num_work = args.dataset_num_workers_test
        #    sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=tmp_bs_per_gpu,
            num_workers=tmp_num_work,
            worker_init_fn=my_worker_init_fn,
            # pin_memory=True
        )
        dataloaders[split + "_sampler"] = sampler
    # print('========================')
    # print(args.show_dir)

    if args.test_only or args.minitest_only:
        criterion = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion, real_test_dataset_config, dataloaders, log_file=args.log_file, if_mini=args.minitest_only, real_cmp_test_dataset_config=real_cmp_test_dataset_config)
    elif args.show_only:
        os.makedirs(args.show_dir, exist_ok=True)
        pred_boxes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.show_dir, args.if_after_nms)
    elif args.show_box_points:
        os.makedirs(args.show_dir, exist_ok=True)
        pred_box_points(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.show_dir, args.if_after_nms)
    elif args.save_novel_only:
        os.makedirs(args.save_novel_dir, exist_ok=True)
        save_novel_boxes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.save_novel_dir, args.if_after_nms)
    elif args.save_novel_with_class_only:
        os.makedirs(args.save_novel_dir, exist_ok=True)
        save_novel_boxes_with_class(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.save_novel_dir, args.if_after_nms)
    elif args.save_seen_feat_only:
        os.makedirs(args.save_seen_dir, exist_ok=True)
        save_seen_feat(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.save_seen_dir, args.if_after_nms)
    elif args.cal_class_only:
        # os.makedirs(args.show_dir, exist_ok=True)
        cal_classes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.show_dir, args.if_after_nms)
    elif args.crop_only:
        os.makedirs(args.crop_dir, exist_ok=True)
        crop_boxes(args, model, model_no_ddp, criterion, dataset_config, dataloaders, args.crop_dir, args.if_after_nms)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer, args.checkpoint_file
        )
        if args.set_epoch == -1:
            args.start_epoch = loaded_epoch + 1
        else:
            args.start_epoch = args.set_epoch
        if args.only_image_class:
            do_train_only_image(
                args,
                model,
                model_no_ddp,
                optimizer,
                criterion,
                dataset_config,
                dataloaders,
                best_val_metrics,
            )
        else:
            do_train(
                args,
                model,
                model_no_ddp,
                optimizer,
                criterion,
                dataset_config,
                dataloaders,
                best_val_metrics,
                real_test_dataset_config,
                real_cmp_test_dataset_config
            )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark=True
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
