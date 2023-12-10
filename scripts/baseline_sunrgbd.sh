#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
mkdir Data/3d_indoor/sunrgb_d/sunrgbd_v1_revised_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_revised_0415_minival
python main.py --dataset_name sunrgbd_anonymous_aligned_image \
--model_name 3detrmulticlasshead \
--if_input_image \
--enc_dim 256 \
--dec_dim 512 \
--dataset_num_workers 4 \
--dataset_num_workers_test 4 \
--train_range_min 0 \
--train_range_max 10 \
--test_range_min 0 \
--test_range_max 46 \
--max_epoch 1080 \
--ngpus 8 \
--nqueries 128 \
--base_lr 1.97e-4 \
--warm_lr_epochs 18 \
--eval_every_epoch 100000000000000000 \
--batchsize_per_gpu 8 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.05 \
--loss_sem_focal_cls_weight 0 \
--loss_sem_cls_weight 0 \
--loss_sem_cls_softmax_weight 0 \
--loss_sem_cls_softmax_skip_none_gt_sample_weight 1 \
--save_separate_checkpoint_every_epoch 90 \
--dist_url tcp://localhost:15699 \
--checkpoint_dir outputs/baseline_sunrgbd \
--if_with_clip \
--real_eval_every_epoch 90 \
--real_cmp_eval_every_epoch 90 \
--if_use_v1 \
--test_num_semcls 46 \