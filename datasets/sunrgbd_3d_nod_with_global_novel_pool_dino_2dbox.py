# Copyright (c) Facebook, Inc. and its affiliates.


""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
import cv2

import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
    get_3d_box_batch_np_xyz,
    get_3d_box_batch_tensor_xyz
)
import datasets.sunrgbd_utils as sunrgbd_utils
import glob
from utils.box_util import extract_pc_in_box3d
from utils.votenet_pc_util import write_oriented_bbox, write_ply, write_ply_rgb
import pickle

MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
#DATA_PATH_V1 = "/cache/data/sunrgbd_moreclass_dealed/sunrgbd_pc_bbox_votes_50k_v1_more_classes" ## Replace with path to dataset
#DATA_PATH_V2 = "/cache/data/sunrgbd_moreclass_dealed/sunrgbd_pc_bbox_votes_50k_v2_more_classes" ## Not used in the codebase.
#DATA_PATH_V1 = "Data/3d_indoor/sunrgb_d/sunrgbd_moreclassv2_dealed/sunrgbd_pc_bbox_votes_50k_v1_more_classesv2"
#DATA_PATH_V2 = "Data/3d_indoor/sunrgb_d/sunrgbd_moreclassv2_dealed/sunrgbd_pc_bbox_votes_50k_v2_more_classesv2"
#DATA_PATH_V1 = "Data/3d_indoor/sunrgb_d/sunrgbd_dealed/sunrgbd_pc_bbox_votes_50k_v1"
#DATA_PATH_V2 = "Data/3d_indoor/sunrgb_d/sunrgbd_dealed/sunrgbd_pc_bbox_votes_50k_v2"
here_path = os.getcwd()
if here_path.find('user-job-dir')!=-1:
    # DATA_PATH_V1 = "/cache/data/sunrgbd_allclasses/sunrgbd_pc_bbox_votes_50k_v1_all_classes" ## Replace with path to dataset
    DATA_PATH_V1 = "/cache/data/sunrgbd_v1_revised_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_revised_0415"
    # DATA_PATH_V2 = "/cache/data/sunrgbd_v2_238classes/sunrgbd_pc_bbox_votes_50k_v2_238classes"
    DATA_PATH_V2 = "/cache/data/sunrgbd_v2_232classes/sunrgbd_pc_bbox_votes_50k_v2_232classes"
    BOX2D_PATH = '/cache/data/sunrgbd_groundingDino_2dboxes'
    # DATA_PATH_V2 = "/cache/data/sunrgbd_v2_allclasses/sunrgbd_pc_bbox_votes_50k_v2_all_classes" ## Not used in the codebase.
    CALIB_PATH = "/cache/data/sunrgbd_trainval/calib/"
    IMAGE_PATH = "/cache/data/sunrgbd_trainval/image/"
    # ALL_CLASS_PATH = "datasets/all_classes_trainval.npy"
    # ALL_CLASS_PATH = "datasets/all_classes_trainval_v2_del_nonval_classes.npy"
    # ALL_CLASS_PATH = "datasets/all_classes_trainval_v2_del_val_less_than_5_classes_anonymous.npy"
    ALL_CLASS_PATH_V1 = "datasets/all_classes_trainval_v1_revised.npy"
    ALL_CLASS_PATH_V2 = "datasets/all_classes_trainval_v2_revised_del_val_less_than_5_classes.npy"
    ALL_SUPERCLASS_PATH = "datasets/sunrgbd_seen_and_obj365_cate_to_idx.npy"
    # ALL_SUPERCLASS_PATH = "datasets/all_classes_trainval_v2_revised_del_val_less_than_5_classes_lvis_combined_1201classes.npy"
    ALL_GLOBAL_OBJECT_PATH = "/cache/data/sunrgbd_v1_revised_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_revised_0415_noveltrain_global_novel_setting0/"
    # ALL_global_OBJECT_PATH = "/cache/data/sunrgbd_v1_revised_0415/sunrgbdv1_novel_classes_768_full_body_in_the_room_cropped_point_e_1b/"
    # ALL_global_2d_OBJECT_PATH = "/cache/data/sunrgbd_v1_revised_0415/sunrgbdv1_novel_classes_768_full_body_in_the_room_obj365_cropped/"
    # ALL_global_2d_OBJECT_PATH = "/cache/data/sunrgbd_v1_revised_0415/sunrgbdv1_novel_classes_768_full_body_in_the_room_cropped/"
else:
    # DATA_PATH_V1 = "Data/3d_indoor/sunrgb_d/sunrgbd_allclasses/sunrgbd_pc_bbox_votes_50k_v1_all_classes"
    DATA_PATH_V1 = "Data/3d_indoor/sunrgb_d/sunrgbd_v1_revised_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_revised_0415"
    # DATA_PATH_V2 = "Data/3d_indoor/sunrgb_d/sunrgbd_v2_238classes/sunrgbd_pc_bbox_votes_50k_v2_238classes"
    DATA_PATH_V2 = "Data/3d_indoor/sunrgb_d/sunrgbd_v2_232classes/sunrgbd_pc_bbox_votes_50k_v2_232classes"
    BOX2D_PATH = 'Data/3d_indoor/sunrgb_d/sunrgbd_groundingDino_2dboxes'
    # DATA_PATH_V2 = "Data/3d_indoor/sunrgb_d/sunrgbd_v2_allclasses/sunrgbd_pc_bbox_votes_50k_v2_all_classes"
    CALIB_PATH = "Data/3d_indoor/sunrgb_d/sunrgbd_trainval/calib/"
    IMAGE_PATH = "Data/3d_indoor/sunrgb_d/sunrgbd_trainval/image/"
    # ALL_CLASS_PATH = "datasets/all_classes_trainval.npy"
    # ALL_CLASS_PATH = "datasets/all_classes_trainval_v2_del_nonval_classes.npy"
    # ALL_CLASS_PATH = "datasets/all_classes_trainval_v2_del_val_less_than_5_classes_anonymous.npy"
    ALL_CLASS_PATH_V1 = "datasets/all_classes_trainval_v1_revised.npy"
    ALL_CLASS_PATH_V2 = "datasets/all_classes_trainval_v2_revised_del_val_less_than_5_classes.npy"
    # ALL_CLASS_PATH = "datasets/ov_3detr.npy"
    ALL_SUPERCLASS_PATH = "datasets/sunrgbd_seen_and_obj365_cate_to_idx.npy"
    # ALL_SUPERCLASS_PATH = "datasets/all_classes_trainval_v2_revised_del_val_less_than_5_classes_lvis_combined_1201classes.npy"
    ALL_GLOBAL_OBJECT_PATH = "Data/3d_indoor/sunrgb_d/sunrgbd_v1_revised_0415/sunrgbd_pc_bbox_votes_50k_v1_all_classes_revised_0415_noveltrain_global_novel_setting0/"
    # ALL_global_OBJECT_PATH = "yang/codes/3d_generation/point-e-main/point_e/examples/sunrgbdv1_novel_classes_768_full_body_in_the_room_cropped_point_e_1b/"
    # ALL_global_2d_OBJECT_PATH = "yang/codes/grounded-SAM/outputs/sunrgbdv1_novel_classes_768_full_body_in_the_room_obj365_cropped/"
    # ALL_global_2d_OBJECT_PATH = "yang/codes/grounded-SAM/outputs/sunrgbdv1_novel_classes_768_full_body_in_the_room_cropped/"

#DATA_PATH_V1 = "/cache/data/sunrgbd_dealed/sunrgbd_pc_bbox_votes_50k_v1" ## Replace with path to dataset
#DATA_PATH_V2 = "/cache/data/sunrgbd_dealed/sunrgbd_pc_bbox_votes_50k_v2" ## Not used in the codebase.


class Sunrgbd3DNODGlobalNovelPoolDino2DboxConfig(object):
    def __init__(self, if_print=True, args=None):
        if args is not None:
            self.num_semcls = 1
        else:
            self.num_semcls = 1 # 238 #137 #690 #1140 #10 #42 #10
        self.num_angle_bin = 12
        self.max_num_obj = args.max_num_obj
        self.use_v1 = args.if_use_v1
        if if_print:
            print('=====================if use v1 version====================')
            print(self.use_v1)
        if self.use_v1:
            self.type2class = np.load(ALL_CLASS_PATH_V1, allow_pickle=True).item()
        else:
            self.type2class = np.load(ALL_CLASS_PATH_V2, allow_pickle=True).item()

        self.type2superclass = self.get_supercate(ALL_SUPERCLASS_PATH) #np.load(ALL_SUPERCLASS_PATH, allow_pickle=True).item()
        # self.type2class = dict(list(self.type2class.items())#[:137])
        if self.use_v1:
            self.type2onehotclass = np.load(ALL_CLASS_PATH_V1, allow_pickle=True).item()
        else:
            self.type2onehotclass = np.load(ALL_CLASS_PATH_V2, allow_pickle=True).item()

        self.type2onehotsuperclass = self.get_supercate(ALL_SUPERCLASS_PATH) #np.load(ALL_SUPERCLASS_PATH, allow_pickle=True).item()
        # self.type2onehotclass = dict(list(self.type2onehotclass.items())#[:137])
        self.pseudo_setting = args.pseudo_setting
        # if if_print:
            # print('Class&Label:')
            # print(self.type2class)
            # print('Resized image size:')
        # self.image_size = (730, 530)

        self.image_size = [args.image_size_width, args.image_size_height] #(730, 531)
        # print(self.image_size)
        # print('==================================')
        # self.image_size = (1000, 1000)
        # self.image_size = (600, 600)
        # self.image_size = (530, 530)
        if if_print:
            print(self.image_size)
            print('Class&Label:')
            print(self.type2class)
   #    self.type2class = {
   #       "bed": 0,
   #       "table": 1,
   #       "sofa": 2,
   #       "chair": 3,
   #       "toilet": 4,
   #       "desk": 5,
   #       "dresser": 6,
   #       "night_stand": 7,
   #       "bookshelf": 8,
   #       "bathtub": 9,
   #   }
   #    self.type2onehotclass = {
   #       "bed": 0,
   #       "table": 1,
   #       "sofa": 2,
   #       "chair": 3,
   #       "toilet": 4,
   #       "desk": 5,
   #       "dresser": 6,
   #       "night_stand": 7,
   #       "bookshelf": 8,
   #       "bathtub": 9,
   #   }
   #      self.test_range = [i for i in range(690)] # test range must be larger than train range, which is because it contains unseen class
        if args is not None:
            self.test_range = [i for i in range(args.test_range_min, args.test_range_max)]
            self.test_max = args.test_range_max
        else:
            self.test_range = [i for i in range(238)] # test range must be larger than train range, which is because it contains unseen class
        if if_print:
            print('Test label range:')
            print(self.test_range)
        # self.train_range = [i for i in range(144)]
        # if args is not None:
        #     print('***************************************')
        #     print(args.train_range_min)
        #     print(args.train_range_max)
        if args is not None:
            self.train_range = [i for i in range(args.train_range_min, args.train_range_max)]
            print('Attention!!!!!!  we adopt pseudo labels here=============')
            self.train_max = args.train_range_max
        else:
            self.train_range = [i for i in range(96)]
        # self.test_range = [i for i in range(238)]
        if if_print:
            print('Train label range:')
            print(self.train_range)
     #   self.type2class = {'table':0, 'night_stand':1, 'cabinet':2, 'counter':3, 'garbage_bin':4,
     #       'bookshelf':5, 'pillow':6, 'microwave':7, 'sink':8, 'stool':9}
     #   self.type2class = {
     #        'toilet':0, 'bed':1, 'chair':2, 'bathtub':3, 'sofa':4,
     #        'dresser':5, 'scanner':6, 'fridge':7, 'lamp':8, 'desk':9,
     #       'table':10, 'night_stand':11, 'cabinet':12, 'counter':13, 'garbage_bin':14,
     #       'bookshelf':15, 'pillow':16, 'microwave':17, 'sink':18, 'stool':19,
     #       'wall':20, 'floor':21, 'door':22, 'window':23, 'picture':24, 'blinds':25,
     #       'shelves':26, 'curtain':27, 'mirror':28, 'floor_mat':29, 'clothes':30,
     #       'ceiling':31, 'books':32, 'tv':33, 'paper':34, 'towel': 35, 'shower_curtain':36,
     #       'box':37, 'whiteboard':38, 'person':39,  'bag':40} 
     #  self.type2class = {
     #       'toilet':0, 'bed':1, 'chair':2, 'bathtub':3, 'sofa':4,
     #       'dresser':5, 'scanner':6, 'fridge':7, 'lamp':8, 'desk':9,
     #       'table':10, 'night stand':11, 'cabinet':12, 'counter':13, 'garbage bin':14,
     #       'bookshelf':15, 'pillow':16, 'microwave':17, 'sink':18, 'stool':19,
     #       'wall':20, 'floor':21, 'door':22, 'window':23, 'picture':24, 'blinds':25,
     #       'shelves':26, 'curtain':27, 'mirror':28, 'floor_mat':29, 'clothes':30,
     #       'ceiling':31, 'books':32, 'tv':33, 'paper':34, 'towel': 35, 'shower_curtain':36,
     #       'box':37, 'whiteboard':38, 'person':39, 'night_stand':40, 'bag':41} 
        self.class2type = {self.type2class[t]: t for t in self.type2class}
    #    self.type2onehotclass = {'table':0, 'night_stand':1, 'cabinet':2, 'counter':3, 'garbage_bin':4,
    #        'bookshelf':5, 'pillow':6, 'microwave':7, 'sink':8, 'stool':9}
     #   self.type2onehotclass = {
     #        'toilet':0, 'bed':1, 'chair':2, 'bathtub':3, 'sofa':4,
     #        'dresser':5, 'scanner':6, 'fridge':7, 'lamp':8, 'desk':9,
     #        'table':10, 'night_stand':11, 'cabinet':12, 'counter':13, 'garbage_bin':14,
     #        'bookshelf':15, 'pillow':16, 'microwave':17, 'sink':18, 'stool':19,
     #        'wall':20, 'floor':21, 'door':22, 'window':23, 'picture':24, 'blinds':25,
     #        'shelves':26, 'curtain':27, 'mirror':28, 'floor_mat':29, 'clothes':30,
     #        'ceiling':31, 'books':32, 'tv':33, 'paper':34, 'towel': 35, 'shower_curtain':36,
     #        'box':37, 'whiteboard':38, 'person':39,  'bag':40} 
#           "bed": 0,
#           "table": 1,
#           "sofa": 2,
#           "chair": 3,
#           "toilet": 4,
#           "desk": 5,
#           "dresser": 6,
#           "night_stand": 7,
#           "bookshelf": 8,
#           "bathtub": 9,}
        self.if_padding_image = True
        self.args = args
        if if_print:
            print('=================if paddng the input image==================')
            print(self.if_padding_image)
            print(self.image_size)

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle


    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_xyz(self, box_center_unnorm, box_size, box_angle):
        # box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor_xyz(box_size, box_angle, box_center_unnorm)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np_xyz(self, box_center_unnorm, box_size, box_angle):
        # box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np_xyz(box_size, box_angle, box_center_unnorm)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

    def get_supercate(self, ALL_SUPERCLASS_PATH):
        class_list = np.load(ALL_SUPERCLASS_PATH, allow_pickle=True).item()
        new_class = {}

        for key_item in class_list.keys():
            new_key_item = key_item.replace(' ', '_').replace('/', '_or_')
            new_class[new_key_item] = class_list[key_item]

        return new_class

class Sunrgbd3DNODGlobalNovelPoolDino2DboxDataLayer(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        use_v1=False,
        augment=False,
        if_input_image=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        meta_data_dir=None,
        if_image_augment=False,
    ):
        self.split_set = split_set
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval", "minival", "testtrain", "noveltrain", "minitrain", "toilettrain"]
        self.dataset_config = dataset_config
        self.type2superclass = self.dataset_config.type2superclass
        self.use_v1 = use_v1
        self.image_size = dataset_config.image_size
        self.if_padding_image = dataset_config.if_padding_image
        if root_dir is None:
            root_dir = DATA_PATH_V1 if use_v1 else DATA_PATH_V2
        print('Root dir:')
        print(root_dir)
        self.data_path = root_dir + "_%s" % (split_set)
        self.calib_path = CALIB_PATH
        self.image_path = IMAGE_PATH
        self.if_input_image = if_input_image
        if split_set in ["train", "val", "minival", "testtrain", "noveltrain", "minitrain", "toilettrain"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])
                )
            )
        elif split_set in ["trainval"]:
            # combine names from both
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(
                    list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
                )
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths

        self.if_input_image = if_input_image

        #print(self.scan_names)
        self.pseudo_setting = dataset_config.pseudo_setting
        print('Adopt pseudo_setting: ')
        assert self.pseudo_setting=='setting0' or self.pseudo_setting=='setting1' or self.pseudo_setting=='setting2' or self.pseudo_setting=='setting3' or self.pseudo_setting=='setting4' or self.pseudo_setting=='debug'
        print(self.pseudo_setting)
        self.pseudo_data_path = self.data_path.replace('train', 'noveltrain_pseudo_labels_'+self.pseudo_setting)
        self.global_novel_path = self.data_path.replace('train', 'noveltrain_global_novel_'+self.pseudo_setting)
        os.makedirs(self.global_novel_path, exist_ok = True)
        os.makedirs(self.pseudo_data_path, exist_ok = True)
        self.num_points = num_points
        self.augment = augment
        self.image_augment = if_image_augment
        print('====================If adopt image augmentation:=====================')
        print(self.image_augment)
        print('====================Adopt augmentation============')
        print(self.augment)
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = dataset_config.args.max_num_obj
        self.len_datasets = len(self.scan_names)
        self.confidence_type_in_datalayer = dataset_config.args.confidence_type_in_datalayer
        print('===============confidence_type_in_datalayer=================')
        print(self.confidence_type_in_datalayer)

        # print('=================global category number===================')
        # self.global_cate_num = self.dataset_config.args.global_cate_num
        # print(self.global_cate_num)
        # print('=====================global sample number======================')
        # self.global_sample_num = self.dataset_config.args.global_sample_num
        # print(self.global_sample_num)
        self.global_obj_path = ALL_GLOBAL_OBJECT_PATH + '%09d_%04d_box.npy' #glob.glob(ALL_GLOBAL_OBJECT_PATH+'*.npy')
        self.global_obj_num = len(glob.glob(ALL_GLOBAL_OBJECT_PATH + '*_box.npy')) // self.dataset_config.args.ngpus
        print('=========================the number of global obj==========================')
        # print(self.global_obj_num)
        # if self.global_sample_num > 0 and self.global_cate_num > 0:
        #     assert self.global_obj_num > 0
        self.conclusion_thres = self.dataset_config.args.conclusion_thres
        print('------------conclusion_thres--------------')
        print(self.conclusion_thres)
        self.max_global_obj = self.dataset_config.args.max_global_obj
        print('------------max global obj----------------')
        print(self.max_global_obj)

        self.min_points_num_of_global_obj = self.dataset_config.args.min_points_num_of_global_obj
        print('------------min_points_num_of_global_obj--------------')
        print(self.min_points_num_of_global_obj)

        self.box2d_path = BOX2D_PATH
        self.box2d_gt_score_thres = dataset_config.args.box2d_gt_score_thres
        print('===================box2d_gt_score_thres======================')
        print(self.box2d_gt_score_thres)

        self.if_not_paste_generated_image = self.dataset_config.args.if_not_paste_generated_image
        print('===================if_not_paste_generated_image===========================')
        print(self.if_not_paste_generated_image)
        self.sunrgbd10_and_captions_path = 'datasets/sunrgbd_10seen_and_captions_1009.npy'
        self.sunrgbd10_and_captions_cate_to_id = np.load(self.sunrgbd10_and_captions_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.scan_names)

    # def get_global_obj_list(self, ALL_GLOBAL_OBJECT_PATH):
    #     novel_list = sorted(glob.glob(ALL_GLOBAL_OBJECT_PATH + '*_box.npy'))
    #     self.global_obj_num = len(novel_list)
    #     return novel_list

    def load_boxes(self, scan_name):
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
            pseudo_scan_path = os.path.join(self.pseudo_data_path, scan_name)
        point_cloud = np.load(scan_path + "_pc.npz")["pc"]
        bboxes_source_tmp = np.load(scan_path + "_bbox.npy")  # K,8
        bboxes_source = np.ones((bboxes_source_tmp.shape[0], 11)) # make the cate prob and objectness of seen cate == 1
        bboxes_source[:, :bboxes_source_tmp.shape[1]] = bboxes_source_tmp
        # bboxes = bboxes_source.copy()
        if self.split_set == 'train':
            if os.path.exists(pseudo_scan_path + "_novel_bbox.npy"):
                pseudo_bboxes_source_tmp = np.load(pseudo_scan_path + "_novel_bbox.npy")  # K,8
            else:
                pseudo_bboxes_source_tmp = np.zeros((0, 8))
                np.save(pseudo_scan_path + "_novel_bbox.npy", pseudo_bboxes_source_tmp)
        # if self.split_set == 'train':
        #     pseudo_bboxes_source_tmp = np.load(pseudo_scan_path + "_novel_bbox.npy")  # K,8
            pseudo_bboxes_source_tmp[:, 3:6] = pseudo_bboxes_source_tmp[:, 3:6] / 2
            pseudo_bboxes_source = np.zeros((pseudo_bboxes_source_tmp.shape[0], 11)) # the last one to see if it is a pseudo label, 0 is a pseudo labels
            pseudo_bboxes_source[:, :pseudo_bboxes_source_tmp.shape[1]] = pseudo_bboxes_source_tmp
            if pseudo_bboxes_source.shape[0] > 0:
                combined_bboxes_source = np.concatenate((bboxes_source, pseudo_bboxes_source), axis=0)
            else:
                combined_bboxes_source = bboxes_source
            combined_bboxes = combined_bboxes_source.copy()
            pseudo_box_path = pseudo_scan_path + "_novel_bbox.npy"

        else:
            combined_bboxes_source = bboxes_source
            combined_bboxes = combined_bboxes_source.copy()
            pseudo_box_path = '_'
            # pseudo_bboxes = pseudo_bboxes_source.copy()
        ori_num = combined_bboxes_source.shape[0]
        return point_cloud, combined_bboxes_source, combined_bboxes, pseudo_box_path, ori_num #, global_pool_path


    def random_move(self, point_cloud_ori, global_obj_ori, global_boxes_ori):
        '''
        :param point_cloud: input point cloud scene
        :param global_obj: global object which is adopted for augmentation
        :return: the moved global_obj which is in the boundding box area of the point cloud scene
        Note that we keep the object is smaller than the scene
        '''
        global_obj = global_obj_ori.copy()
        point_cloud = point_cloud_ori.copy()
        global_boxes = global_boxes_ori.copy()

        ###########random rotate the box -90~+90 degree###########
        rot_angle = (np.random.random() * np.pi) - np.pi / 2.0  # -90 ~ +90 degree
        # rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
        rot_mat = pc_util.rotz(rot_angle)

        global_obj[:, 0:3] = np.dot(global_obj[:, 0:3], np.transpose(rot_mat))
        global_boxes[:, 0:3] = np.dot(global_boxes[:, 0:3], np.transpose(rot_mat))
        global_boxes[:, 6] -= rot_angle
        ###########random rotate the box -90~+90 degree###########

        ###########randomly scale: 0.75x-1x###########
        scale_ratio = np.random.random() * 0.25 + 0.75
        scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
        global_obj[:, 0:3] *= scale_ratio
        global_boxes[:, 0:3] *= scale_ratio
        global_boxes[:, 3:6] *= scale_ratio
        ###########randomly scale: 0.25x-1x###########

        scene_min_x = np.min(point_cloud[:, 0])
        scene_max_x = np.max(point_cloud[:, 0])
        scene_min_y = np.min(point_cloud[:, 1])
        scene_max_y = np.max(point_cloud[:, 1])
        scene_min_z = np.min(point_cloud[:, 2])
        scene_max_z = np.max(point_cloud[:, 2])

        object_min_x = np.min(global_obj[:, 0])
        object_max_x = np.max(global_obj[:, 0])
        object_min_y = np.min(global_obj[:, 1])
        object_max_y = np.max(global_obj[:, 1])
        object_min_z = np.min(global_obj[:, 2])
        object_max_z = np.max(global_obj[:, 2])

        while((object_max_x-object_min_x) > (scene_max_x-scene_min_x) or (object_max_y-object_min_y) > (scene_max_y-scene_min_y) or (object_max_z-object_min_z) > (scene_max_z-scene_min_z)):
            '''
            scale the point clouds when the object is larger than scene
            '''
            scale_ratio = 0.75
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            global_obj[:, 0:3] *= scale_ratio
            global_boxes[:, 0:3] *= scale_ratio
            global_boxes[:, 3:6] *= scale_ratio
            object_min_x = np.min(global_obj[:, 0])
            object_max_x = np.max(global_obj[:, 0])
            object_min_y = np.min(global_obj[:, 1])
            object_max_y = np.max(global_obj[:, 1])
            object_min_z = np.min(global_obj[:, 2])
            object_max_z = np.max(global_obj[:, 2])

        win_min_x = scene_min_x - object_min_x
        win_max_x = scene_max_x - object_max_x
        win_min_y = scene_min_y - object_min_y
        win_max_y = scene_max_y - object_max_y
        win_min_z = scene_min_z - object_min_z
        win_max_z = scene_max_z - object_max_z

        diff_x = np.random.uniform(low=min(win_min_x, win_max_x), high=max(win_min_x, win_max_x), size=1)[0]
        diff_y = np.random.uniform(low=min(win_min_y, win_max_y), high=max(win_min_y, win_max_y), size=1)[0]
        diff_z = np.random.uniform(low=min(win_min_z, win_max_z), high=max(win_min_z, win_max_z), size=1)[0]

        global_obj[:, 0] += diff_x
        global_obj[:, 1] += diff_y
        global_obj[:, 2] += diff_z
        global_boxes[:, 0] += diff_x
        global_boxes[:, 1] += diff_y
        global_boxes[:, 2] += diff_z

        return global_obj, global_boxes

    # def get_boxes(self, global_obj_ori, global_obj_class_id):
    #     '''
    #     :param point_cloud: input point cloud scene
    #     :param global_obj: global object which is adopted for augmentation
    #     :return: the moved global_obj which is in the boundding box area of the point cloud scene
    #     Noted that we conduct random rotation and scale here for the object point clouds and boxes
    #     '''
    #     global_obj = global_obj_ori.copy()
    #
    #     object_min_x = np.min(global_obj[:, 0])
    #     object_max_x = np.max(global_obj[:, 0])
    #     object_min_y = np.min(global_obj[:, 1])
    #     object_max_y = np.max(global_obj[:, 1])
    #     object_min_z = np.min(global_obj[:, 2])
    #     object_max_z = np.max(global_obj[:, 2])
    #     c_x = (object_max_x+object_min_x)/2
    #     c_y = (object_max_y+object_min_y)/2
    #     c_z = (object_max_z+object_min_z)/2
    #     w = object_max_x-object_min_x
    #     h = object_max_y-object_min_y
    #     l = object_max_z-object_min_z
    #     boxes = np.array([[c_x, c_y, c_z, w, h, l, 0, global_obj_class_id]])
    #
    #     # ###########random rotate the box -90~+90 degree###########
    #     # rot_angle = (np.random.random() * np.pi) - np.pi / 2.0  # -90 ~ +90 degree
    #     # # rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
    #     # rot_mat = pc_util.rotz(rot_angle)
    #     #
    #     # global_obj[:, 0:3] = np.dot(global_obj[:, 0:3], np.transpose(rot_mat))
    #     # boxes[:, 0:3] = np.dot(boxes[:, 0:3], np.transpose(rot_mat))
    #     # boxes[:, 6] -= rot_angle
    #     # ###########random rotate the box -90~+90 degree###########
    #     #
    #     # ###########randomly scale: 0.5x-1.1x###########
    #     # scale_ratio = np.random.random() * 0.6 + 0.5
    #     # scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
    #     # global_obj[:, 0:3] *= scale_ratio
    #     # boxes[:, 0:3] *= scale_ratio
    #     # boxes[:, 3:6] *= scale_ratio
    #     # ###########randomly scale: 0.5x-1.1x###########
    #
    #     return boxes

    # def random_select(self, global_obj_moved_ori):
    #     '''
    #     :param global_obj_moved_ori: the original points of global objects
    #     :return: the random selected points. number range[self.min_points_num_of_global_obj, original_num]
    #     '''
    #     global_obj_moved = global_obj_moved_ori.copy()
    #     ori_num = global_obj_moved.shape[0]
    #     tmp_global_obj_num = np.random.randint(ori_num-500, high=ori_num + 1, size=1)[0]
    #     global_obj_select, choices = pc_util.random_sampling(
    #         global_obj_moved, tmp_global_obj_num, return_choices=True
    #     )
    #     return  global_obj_select

    def flip_axis_to_depth(self, pc):
        pc2 = np.copy(pc)
        pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
        pc2[..., 2] *= -1
        return pc2

    def check_conclusion(self, global_obj_rotated_ori, global_boxes_ori):
        global_obj_rotated = global_obj_rotated_ori.copy()
        global_boxes = global_boxes_ori.copy()
        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            global_boxes[0, :3, ...],
            global_boxes[0, 3:6, ...],
            global_boxes[0, 6, ...],
        )
        box_corners = self.flip_axis_to_depth(box_corners)
        pc_tmp, _ = extract_pc_in_box3d(global_obj_rotated, box_corners) # check condition: use object points, all points should in. Check passed.
        return len(pc_tmp)>self.conclusion_thres

    def check_2d(self, input_image, global_boxes_ori, ori_height, ori_width, y_offset, x_offset, calib):
        # global_obj_rotated = global_obj_rotated_ori.copy()
        global_boxes = global_boxes_ori.copy()
        box_centers = global_boxes[:, :3]
        raw_sizes = global_boxes[:, 3:6]
        raw_angles = global_boxes[:, 6]
        box_corners_xyz = self.dataset_config.box_parametrization_to_corners_np_xyz(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners_xyz = box_corners_xyz.squeeze(0).squeeze(0)

        corners_uv, _, d = calib.project_upright_depth_to_image(box_corners_xyz)

        if np.min(d) < 1e-32:
            return True

        corners_uv[:, 0] = np.clip(corners_uv[:, 0], 0, ori_width - 1)
        corners_uv[:, 1] = np.clip(corners_uv[:, 1], 0, ori_height - 1)
        corners_uv[:, 0] += y_offset
        corners_uv[:, 1] += x_offset  # .unsqueeze(-1)  # short height

        xmin = int(np.min(corners_uv[:, 0]))
        ymin = int(np.min(corners_uv[:, 1]))
        xmax = int(np.max(corners_uv[:, 0]))
        ymax = int(np.max(corners_uv[:, 1]))
        if (xmax - xmin) < 30 or (ymax - ymin) < 30:
            return  True
        # box_corners = self.dataset_config.box_parametrization_to_corners_np(
        #     global_boxes[0, :3, ...],
        #     global_boxes[0, 3:6, ...],
        #     global_boxes[0, 6, ...],
        # )
        # box_corners = self.flip_axis_to_depth(box_corners)
        # pc_tmp, _ = extract_pc_in_box3d(global_obj_rotated, box_corners) # check condition: use object points, all points should in. Check passed.
        # return len(pc_tmp)>self.conclusion_thres
        return False

    def get_global_image(self, global_2d_obj_path):
        """
        :param global_2d_obj_path: the path of global images
        :return: global images augmented
        """
        global_2d_obj = cv2.imread(global_2d_obj_path)
        global_2d_obj = np.array(global_2d_obj, dtype=np.float32)
        # global_2d_obj[global_2d_obj < 1e-32]
        global_2d_obj = cv2.cvtColor(global_2d_obj, cv2.COLOR_BGR2RGB)
        global_2d_obj = np.array(global_2d_obj, dtype=np.uint8)
        if self.image_augment: # True
            # print( global_2d_obj.dtype)
            global_2d_obj =  global_2d_obj / 255.0
            # print( global_2d_obj.shape) #(538, 730, 3)
            # print(np.max( global_2d_obj))
            # print(np.min( global_2d_obj))
            if np.random.random() > 0.5: # flip
                global_2d_obj =  global_2d_obj[:, ::-1, :]
            # print('innnnnnnnnnnnnnnnnnnn')
            global_2d_obj *= (
                        1 + 0.4 * np.random.random(3) - 0.2
                    )  # brightness change for each channel
            # print((
            #             1 + 0.4 * np.random.random(3) - 0.2
            #         ).shape)
            global_2d_obj += (
                        0.1 * np.random.random(3) - 0.05
                    )  # color shift for each channel
            # print((
            #             0.1 * np.random.random(3) - 0.05
            #         ).shape)
            global_2d_obj += np.expand_dims(
                        (0.05 * np.random.random(( global_2d_obj.shape[0],  global_2d_obj.shape[1])) - 0.025), -1
                    )  # jittering on each pixel
            # print(np.expand_dims(
            #             (0.05 * np.random.random(( global_2d_obj.shape[0],  global_2d_obj.shape[1])) - 0.025), -1
            #         ).shape)
            global_2d_obj = np.clip( global_2d_obj, 0, 1)
            global_2d_obj =  global_2d_obj * 255.0
            # global_2d_obj = cv2.resize(global_2d_obj, (768, 768), cv2.INTER_LINEAR)
            # global_2d_obj_mask = cv2.resize(global_2d_obj_mask, (768, 768), cv2.INTER_NEAREST)
            # global_2d_obj_mask = global_2d_obj_mask.astype(np.uint8)
            global_2d_obj =  global_2d_obj.astype(np.uint8)
        return global_2d_obj

    def past_image(self, ret_dict, calib):
        # print(1)
        flip_array = ret_dict['flip_array'].copy()
        scale_array = ret_dict['scale_array'].copy()
        rot_array = ret_dict['rot_array'].copy()

        is_global_box = ret_dict['is_global_box'].copy()
        global_box_to_image_idx = ret_dict['global_box_to_image_idx'].copy()
        all_global_image = ret_dict['all_global_image'].copy()
        all_global_image_mask = ret_dict['all_global_image_mask'].copy()
        gt_box_corners_xyz = ret_dict['gt_box_corners_xyz'].copy()
        
        # input_image = ret_dict['input_image'].copy()
        # cv2.imwrite('before.png', input_image)

        for is_global_idx in range(is_global_box.shape[0]):
            is_global_item = is_global_box[is_global_idx]
            if is_global_item==0: # skip when is_not_global
                continue
            image_idx = global_box_to_image_idx[is_global_idx]
            image_item = all_global_image[image_idx]
            image_item_mask = all_global_image_mask[image_idx]
            corners_xyz_item_ori = gt_box_corners_xyz[is_global_idx]
            corners_xyz_item = corners_xyz_item_ori * scale_array  # [:, :2, :]
            corners_xyz_item = np.dot(corners_xyz_item, rot_array)
            corners_xyz_item[:, 0] = corners_xyz_item[:, 0] * flip_array  # [:, :2, :]
            corners_uv, _, d = calib.project_upright_depth_to_image(corners_xyz_item)

            corners_uv[:, 0] = np.clip(corners_uv[:, 0], 0, ret_dict["ori_width"] - 1)
            corners_uv[:, 1] = np.clip(corners_uv[:, 1], 0, ret_dict["ori_height"] - 1)
            corners_uv[:, 0] += ret_dict['y_offset']
            corners_uv[:, 1] += ret_dict['x_offset'] # .unsqueeze(-1)  # short height
            # print(1)

            image_flip_array = ret_dict["image_flip_array"]
            flip_length = ret_dict["flip_length"]
            corners_uv[:,  0] = corners_uv[:, 0] * image_flip_array + (1 - image_flip_array) * (flip_length - 1 - corners_uv[:, 0])

            xmin = int(np.min(corners_uv[:, 0]))
            ymin = int(np.min(corners_uv[:, 1]))
            xmax = int(np.max(corners_uv[:, 0]))
            ymax = int(np.max(corners_uv[:, 1]))
            if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                w_size = ymax - ymin
                h_size = xmax - xmin
                image_item_resized = cv2.resize(image_item, (h_size, w_size), cv2.INTER_LINEAR)
                image_item_mask_resized = cv2.resize(image_item_mask, (h_size, w_size), cv2.INTER_NEAREST)
                image_item_mask_resized = np.expand_dims(image_item_mask_resized, -1)
                ret_dict['input_image'][ymin:ymax, xmin:xmax, :] = (image_item_resized * image_item_mask_resized + ret_dict['input_image'][ymin:ymax, xmin:xmax, :] * (1 - image_item_mask_resized))
            else:
                continue

        # cv2.imwrite('after.png', ret_dict['input_image'])
        return ret_dict


    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        self.global_obj_num = len(glob.glob(ALL_GLOBAL_OBJECT_PATH + '*_box.npy')) // self.dataset_config.args.ngpus
        #print(self.image_name)
        if self.if_input_image:
            self.calib_name = os.path.join(self.calib_path, scan_name+'.txt')
            calib = sunrgbd_utils.SUNRGBD_Calibration(self.calib_name, if_tensor=False)
        #print(self.calib_name)
            self.image_name = os.path.join(self.image_path, scan_name+'.jpg')
            self.input_image = cv2.imread(self.image_name)
            self.input_image = np.array(self.input_image, dtype=np.float32)

            if self.image_size != None:
                if self.if_padding_image:
                    height, width = self.input_image.shape[:2]
                    self.ori_height = height
                    self.ori_width = width
                    # print(self.image_size)
                    # print(self.input_image.shape)
                    input_image_padded = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
                    # print(input_image_padded.shape)
                    self.x_offset = (self.image_size[1] - height) // 2
                    self.y_offset = (self.image_size[0] - width) // 2
                    input_image_padded[self.x_offset:self.x_offset + height, self.y_offset:self.y_offset + width,
                    :] = self.input_image
                    self.input_image = input_image_padded
                    trans_affine = [0]
                    # trans_mtx = np.zeros((1))
                    scale_h = 448 * 1.0 / self.image_size[1]
                    scale_w = 448 * 1.0 / self.image_size[0]
                    trans_mtx = np.eye(2, 2)
                    trans_mtx[0, 0] *= scale_w
                    trans_mtx[1, 1] *= scale_h
                    # print(self.input_image.shape)
                else:
                    height, width = self.input_image.shape[:2]
                    # print(self.input_image.shape)
                    scale_h = self.image_size[1] * 1.0 / height
                    scale_w = self.image_size[0] * 1.0 / width
                    trans_mtx = np.eye(2, 2)
                    trans_mtx[0, 0] *= scale_w
                    trans_mtx[1, 1] *= scale_h
                    # print(trans_mtx.shape)
                    self.input_image = cv2.resize(self.input_image, self.image_size, cv2.INTER_LINEAR)
                    # center = np.array([width / 2, height / 2], dtype=np.float32)
                    # size = np.array([width, height], dtype=np.float32)
                    # trans_affine = self._get_transform_matrix(center, size, self.image_size)
                    # self.input_image = cv2.warpAffine(self.input_image, trans_affine[:2, :], self.image_size)
            else:
                trans_affine = [0]


        #print(self.input_image.shape)
            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
            self.input_image = np.array(self.input_image, dtype=np.uint8)
        #calib = sunrgbd_utils.SUNRGBD_Calibration(self.calib_name)
         # Nx6

        point_cloud, bboxes_source, bboxes, pseudo_box_path, ori_num = self.load_boxes(scan_name)
        point_cloud_rbg_ori = point_cloud.copy()
        box2d_path = os.path.join(self.box2d_path, scan_name + '_2d_boxes.pkl')
        bboxes_seen = []
        bboxes_seenclass = []
        for idx_box in range(bboxes.shape[0]):
            #print('2-idx-box:')
            #print(idx_box)
            #    print(bboxes[idx_box, -1])
            #if (10 <= bboxes[idx_box, -1]) and (20 > bboxes[idx_box, -1]):
            #print('================================')
            #print(bboxes[idx_box, -1])
            #print(self.dataset_config.test_range)
            #print(bboxes[idx_box, -1] in self.dataset_config.test_range)

            cnt_bboxes = 0
            if (bboxes[idx_box, 7] in self.dataset_config.test_range): # and (20 > bboxes[idx_box, -1]):
                if (bboxes[idx_box, 7] in self.dataset_config.train_range):
                    bboxes_seenclass.append(bboxes[idx_box, 7])
                else:
                    bboxes_seenclass.append(self.dataset_config.train_max)
                bboxes[idx_box, 7] = 0 #bboxes[idx_box, -1] - self.dataset_config.test_range[0] # begin from zero
                cnt_bboxes += 1
                bboxes_seen.append(bboxes[idx_box])
        #        print('seen')
        #    else:
        #        print('unseen')
        # print('=============0===============')
        # print(len(bboxes_seenclass))
        if len(bboxes_seen) == 0:
            bboxes = np.zeros((0, 8))
        else:
            bboxes = np.array(bboxes_seen)
        #print('1-idx:')
        #print(idx)
        if self.split_set == 'train' or 'noveltrain':
            #print('==========================')
            #print(bboxes.shape)
            box2d_present = np.zeros(128)
            with open(box2d_path, 'rb') as f:
                box2d_predictions = pickle.load(f) #crop image use (1, 3, 2, 0)
            box2d_num = box2d_predictions.xyxy.shape[0]
            box2d_original = np.zeros((box2d_num, 6))
            box2d_original[:, :4] = box2d_predictions.xyxy
            box2d_original[:, 5] = box2d_predictions.confidence
            for idx_box2d in range(box2d_num):
                tmp_cate = box2d_predictions.class_name[idx_box2d]
                box2d_original[idx_box2d, 4] =  self.sunrgbd10_and_captions_cate_to_id[tmp_cate] # wait for debug


            box2d_tmp_list = []
            for box_idx in range(box2d_original.shape[0]):
                if box2d_original[box_idx, -1] > self.box2d_gt_score_thres:
                    box2d_tmp_list.append(box2d_original[box_idx][np.newaxis, :])

            if len(box2d_tmp_list) > 0:
                box2d_tmp = np.concatenate(box2d_tmp_list, axis=0)
            else:
                box2d_tmp = np.zeros((0, 6))

            if box2d_tmp.shape[0] > 128:
                box2d_scores = box2d_tmp[:,-1]
                box2d_scores_sorted = np.sort(box2d_scores, axis=-1)
                box2d_scores_sorted_kth = box2d_scores_sorted[-128]
                box2d = box2d_tmp[box2d_tmp[:,-1] > box2d_scores_sorted_kth,:]
            else:
                box2d = box2d_tmp
            box2d[:, 0] += self.y_offset
            box2d[:, 2] += self.y_offset #7xx
            box2d[:, 1] += self.x_offset #5xx
            box2d[:, 3] += self.x_offset
            box2d_final = np.zeros((128, 6))
            box2d_present[:box2d.shape[0]] = 1
            box2d_final[:box2d.shape[0]] = box2d

            ############ load global objects
            tmp_global_obj_num = np.random.randint(0, high=self.max_global_obj+1, size=1)[0]
            self.all_global_image = np.zeros((self.max_global_obj, 224, 224, 3))
            global_boxes_list = []
            # write_ply(point_cloud, 'debug_ori_7.ply')
            cnt_global_box = 0
            tmp_idx_obj = 0
            while(tmp_idx_obj < (tmp_global_obj_num)):
                if self.global_obj_num ==0: # there is no global objects in the pool
                    break
                global_idx = np.random.randint(0, high=self.global_obj_num, size=1)[0]
                gpu_idx = np.random.randint(0, high=self.dataset_config.args.ngpus, size=1)[0]
                global_box_path = self.global_obj_path % (global_idx, gpu_idx)
                #global_obj_class = os.path.basename(global_obj_path)[:-18]
                #assert global_obj_class in self.dataset_config.type2class.keys()

                global_2d_obj_path = global_box_path.replace('box.npy', 'im.png')
                global_obj_path = global_box_path.replace('box.npy', 'pc.npy')

                if not (os.path.exists(global_box_path) and os.path.exists(global_2d_obj_path) and os.path.exists(global_obj_path)):
                    continue

                try:
                    global_2d_obj = self.get_global_image(global_2d_obj_path)

                    global_obj = np.load(global_obj_path)
                    global_boxes = np.load(global_box_path)[np.newaxis, :]
                except:
                    continue
                # global_obj_class_id = global_boxes[-1]

                global_obj_moved, global_boxes_moved = self.random_move(point_cloud, global_obj, global_boxes)
                recheck_time = 0
                if_found_pos = True
                while self.check_conclusion(point_cloud, global_boxes_moved) or self.check_2d(self.input_image, global_boxes_moved, self.ori_height, self.ori_width, self.y_offset, self.x_offset, calib):
                    global_obj_moved, global_boxes_moved = self.random_move(point_cloud, global_obj, global_boxes)
                    recheck_time += 1
                    if recheck_time > 100: # here seems has a bug
                        if_found_pos = False
                        break
                if not if_found_pos:
                #     print('Found bad case!!!!!!!!!!1')
                    continue
                self.all_global_image[cnt_global_box, :, :, :] = global_2d_obj
                # self.all_global_image_mask[cnt_global_box, :, :] = global_2d_obj_mask
                cnt_global_box += 1
                global_obj_select = global_obj_moved #self.random_select(global_obj_moved)
                point_cloud = np.concatenate((point_cloud, global_obj_select), axis=0)
                global_boxes_list.append(global_boxes_moved)
                tmp_idx_obj += 1

            if len(global_boxes_list) == 0:
                global_boxes = np.zeros((0, 8))
            else:
                global_boxes = np.concatenate(global_boxes_list, axis=0)
            # ############ load global objects
            #
            # # ###debuging by saving points
            # write_ply(point_cloud, 'after_pc.ply')
            # global_boxes[:, 6] = -1 * global_boxes[:, 6]
            # write_oriented_bbox(global_boxes, 'global_box.ply')
            # # ###debuging by saving points
            global_boxes[:, 3:6] = global_boxes[:, 3:6] / 2 # need to divide 2 before concating

            cnt_bboxes = 0
            bboxes_seen = []
            bboxes_seenclass = []
            bboxes_seenconfi = []
            bboxes = bboxes_source.copy()

            for idx_box in range(bboxes.shape[0]):
                #print('2-idx-box:')
                #print(idx_box)
                #    print(bboxes[idx_box, -1])
                #if (10 <= bboxes[idx_box, -1]) and (20 > bboxes[idx_box, -1]):
                assert bboxes[idx_box, -1]==1 or bboxes[idx_box, -1]==0
                if bboxes[idx_box, -1]==1: # real label
                    if (bboxes[idx_box, 7] in self.dataset_config.train_range): # and (20 > bboxes[idx_box, -1]):
                        bboxes_seenclass.append(bboxes[idx_box, 7])
                        bboxes[idx_box, 7] = 0 #bboxes[idx_box, -1] - self.dataset_config.test_range[0]
                        cnt_bboxes += 1
                        bboxes_seen.append(bboxes[idx_box][:8])
                        bboxes_seenconfi.append(1.0) # confi for real label is 1
                elif bboxes[idx_box, -1]==0: # pseudo labels
                    bboxes_seenclass.append(bboxes[idx_box, 7])
                    bboxes[idx_box, 7] = 0  # bboxes[idx_box, -1] - self.dataset_config.test_range[0]
                    cnt_bboxes += 1
                    bboxes_seen.append(bboxes[idx_box][:8])
                    if self.confidence_type_in_datalayer=='clip-max-prob':
                        confi_tmp = bboxes[idx_box, 8] # confi for pseudo label, adopt class prob now, do not use objectness
                        bboxes_seenconfi.append(confi_tmp)
                    elif self.confidence_type_in_datalayer=='zero-out':
                        bboxes_seenconfi.append(0)
                    elif self.confidence_type_in_datalayer=='objectness':
                        confi_tmp = bboxes[idx_box, 9] # confi for pseudo label, adopt class prob now, do not use objectness
                        bboxes_seenconfi.append(confi_tmp)
                    elif self.confidence_type_in_datalayer=='clip+objectness':
                        confi_tmp = (bboxes[idx_box, 8]+bboxes[idx_box, 9])/2.0 # confi for pseudo label, adopt class prob now, do not use objectness
                        bboxes_seenconfi.append(confi_tmp)
                    elif self.confidence_type_in_datalayer=='weight_one':
                        bboxes_seenconfi.append(1)
            #        print('seen')
            #    else:
            #        print('unseen')

            num_boxes_before_global_aug = len(bboxes_seen)
            is_global_box = np.zeros(num_boxes_before_global_aug+global_boxes.shape[0])
            is_global_image = np.zeros(self.max_global_obj)
            for idx_box in range(global_boxes.shape[0]):
                is_global_box[num_boxes_before_global_aug+idx_box] = 1
                is_global_image[idx_box] = 1
                #print('2-idx-box:')
                #print(idx_box)
                #    print(bboxes[idx_box, -1])
                #if (10 <= bboxes[idx_box, -1]) and (20 > bboxes[idx_box, -1]):
                bboxes_seenclass.append(global_boxes[idx_box, -1])
                global_boxes[idx_box, -1] = 0 #bboxes[idx_box, -1] - self.dataset_config.test_range[0]
                cnt_bboxes += 1
                bboxes_seen.append(global_boxes[idx_box])
                bboxes_seenconfi.append(1) # all confi weights of global objects are set to 1
            # if self.confidence_type_in_datalayer == 'clip-max-prob':
            #     confi_tmp = bboxes[idx_box, 8]  # confi for pseudo label, adopt class prob now, do not use objectness
            #     bboxes_seenconfi.append(confi_tmp)
            # elif self.confidence_type_in_datalayer == 'zero-out':
            #     bboxes_seenconfi.append(0)
            # elif self.confidence_type_in_datalayer == 'objectness':
            #     confi_tmp = bboxes[idx_box, 9]  # confi for pseudo label, adopt class prob now, do not use objectness
            #     bboxes_seenconfi.append(confi_tmp)
            # elif self.confidence_type_in_datalayer == 'clip+objectness':
            #     confi_tmp = (bboxes[idx_box, 8] + bboxes[
            #         idx_box, 9]) / 2.0  # confi for pseudo label, adopt class prob now, do not use objectness
            #     bboxes_seenconfi.append(confi_tmp)
            # elif self.confidence_type_in_datalayer == 'weight_one':
            #     bboxes_seenconfi.append(1)

            if len(bboxes_seen) == 0:
                bboxes = np.zeros((0, 8))
            else:
                bboxes = np.array(bboxes_seen)
            #print(bboxes.shape)

            # ################skip the no-seen-object scene###################
            # while(bboxes.shape[0]==0):
            #     #print('3-idx:')
            #     #print(idx)
            #     idx = np.random.randint(0, high=self.len_datasets, size=1)[0]
            #     scan_name = self.scan_names[idx]
            #     point_cloud, bboxes_source, bboxes, pseudo_box_path, ori_num = self.load_boxes(scan_name)
            #     # conduct the resize
            #     if self.if_input_image:
            #         self.calib_name = os.path.join(self.calib_path, scan_name + '.txt')
            #         # print(self.calib_name)
            #         self.image_name = os.path.join(self.image_path, scan_name + '.jpg')
            #         self.input_image = cv2.imread(self.image_name)
            #         self.input_image = np.array(self.input_image, dtype=np.float32)
            #
            #
            #     if self.if_input_image and self.image_size != None:
            #         if self.if_padding_image:
            #             height, width = self.input_image.shape[:2]
            #             self.ori_height = height
            #             self.ori_width = width
            #             # print(self.image_size)
            #             # print(self.input_image.shape)
            #             input_image_padded = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
            #             # print(input_image_padded.shape)
            #             self.x_offset = (self.image_size[1] - height) // 2
            #             self.y_offset = (self.image_size[0] - width) // 2
            #             input_image_padded[self.x_offset:self.x_offset + height, self.y_offset:self.y_offset + width,
            #             :] = self.input_image
            #             self.input_image = input_image_padded
            #             trans_affine = [0]
            #             # scale_h = 448 * 1.0 / self.image_size[1]
            #             # scale_w = 448 * 1.0 / self.image_size[0]
            #             # trans_mtx = np.eye(2, 2)
            #             # trans_mtx[0, 0] *= scale_w
            #             # trans_mtx[1, 1] *= scale_h
            #             # print(self.input_image.shape)
            #         else:
            #             height, width = self.input_image.shape[:2]
            #             # print(self.input_image.shape)
            #             scale_h = self.image_size[1] * 1.0 / height
            #             scale_w = self.image_size[0] * 1.0 / width
            #             trans_mtx = np.eye(2, 2)
            #             trans_mtx[0, 0] *= scale_w
            #             trans_mtx[1, 1] *= scale_h
            #             # print(trans_mtx.shape)
            #             self.input_image = cv2.resize(self.input_image, self.image_size, cv2.INTER_LINEAR)
            #             # center = np.array([width / 2, height / 2], dtype=np.float32)
            #             # size = np.array([width, height], dtype=np.float32)
            #             # trans_affine = self._get_transform_matrix(center, size, self.image_size)
            #             # self.input_image = cv2.warpAffine(self.input_image, trans_affine[:2, :], self.image_size)
            #     else:
            #         trans_affine = [0]
            #
            #         # print(self.input_image.shape)
            #     if self.if_input_image:
            #         self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
            #         self.input_image = np.array(self.input_image, dtype=np.uint8)
            #
            #     cnt_bboxes = 0
            #     bboxes_seen = []
            #     bboxes_seenclass = []
            #     for idx_box in range(bboxes.shape[0]):
            #         #print('4-idx-box:')
            #         #print(idx_box)
            #         #    print(bboxes[idx_box, -1])
            #         if (bboxes[idx_box, -1] in self.dataset_config.train_range): #and (20 > bboxes[idx_box, -1]):
            #             if bboxes[idx_box, -1] == -1:
            #                 bboxes_seenclass.append(self.dataset_config.train_max)
            #             else:
            #                 bboxes_seenclass.append(bboxes[idx_box, -1])
            #             bboxes[idx_box, -1] = 0 #bboxes[idx_box, -1] - self.dataset_config.test_range[0]
            #         #if (10 > bboxes[idx_box, -1]):
            #             cnt_bboxes += 1
            #             bboxes_seen.append(bboxes[idx_box])
            #     #        print('seen')
            #     #    else:
            #     #        print('unseen')
            #     if len(bboxes_seen) == 0:
            #         bboxes = np.zeros((0, 8))
            #     else:
            #         bboxes = np.array(bboxes_seen)
        # print('=============1===============')
        # print(len(bboxes_seenclass))
        # ################skip the no-seen-object scene###################

        bboxes_seenclass = np.array(bboxes_seenclass)
        bboxes_seenconfi = np.array(bboxes_seenconfi)
        # print('=============2===============')
        # print(bboxes.shape)
        # print(bboxes_seenclass.shape)
        # scale_h = self.image_size[1] * 1.0 / self.image_size[1]
        # scale_w = self.image_size[0] * 1.0 / self.image_size[0]
        # trans_mtx = np.eye(2, 2)
        # trans_mtx[0, 0] *= scale_w
        # trans_mtx[1, 1] *= scale_h

        uv_2d, _, d = calib.project_upright_depth_to_image(point_cloud[:, :3])
        uv_2d[:, 0] += self.y_offset
        uv_2d[:, 1] += self.x_offset
        uv_2d[:, 0] = np.clip(uv_2d[:, 0], 0, self.image_size[0] - 1)
        uv_2d[:, 1] = np.clip(uv_2d[:, 1], 0, self.image_size[1] - 1)
        # uv_2d = np.dot(uv_2d, trans_mtx)

        point_cloud_rgb = point_cloud[:, 0:6]
        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            assert point_cloud.shape[1] == 6
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB


        if self.use_height: #False
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1
            )  # (N,4) or (N,7)
        # flip_array = np.ones()
        # ------------------------------- Image DATA AUGMENTATION ------------------------
        image_flip_array = np.ones(1)
        if self.if_input_image and self.image_augment: # True
            # print(self.input_image.dtype)
            self.input_image = self.input_image / 255.0
            # print(self.input_image.shape) #(538, 730, 3)
            # print(np.max(self.input_image))
            # print(np.min(self.input_image))
            if np.random.random() > 0.5: # flip
                self.input_image = self.input_image[:, ::-1, :]
                uv_2d[:, 0] = self.image_size[0] - 1 - uv_2d[:, 0]
                box2d_final[:, 0] = self.image_size[0] - 1 - box2d_final[:, 0]
                box2d_final[:, 2] = self.image_size[0] - 1 - box2d_final[:, 2]
                image_flip_array = np.zeros(1)
            # print('innnnnnnnnnnnnnnnnnnn')
            self.input_image *= (
                        1 + 0.4 * np.random.random(3) - 0.2
                    )  # brightness change for each channel
            # print((
            #             1 + 0.4 * np.random.random(3) - 0.2
            #         ).shape)
            self.input_image += (
                        0.1 * np.random.random(3) - 0.05
                    )  # color shift for each channel
            # print((
            #             0.1 * np.random.random(3) - 0.05
            #         ).shape)
            self.input_image += np.expand_dims(
                        (0.05 * np.random.random((self.input_image.shape[0], self.input_image.shape[1])) - 0.025), -1
                    )  # jittering on each pixel
            # print(np.expand_dims(
            #             (0.05 * np.random.random((self.input_image.shape[0], self.input_image.shape[1])) - 0.025), -1
            #         ).shape)
            self.input_image = np.clip(self.input_image, 0, 1)
            self.input_image = self.input_image * 255.0
            self.input_image = self.input_image.astype(np.uint8)
            # print(self.input_image.dtype)
            # print(self.input_image)
            # if np.random.random() > 0.5:  # transpose

        # ------------------------------- DATA AUGMENTATION ------------------------------
        rot_array = np.identity(3)
        scale_array = np.ones((1, 3))
        flip_array = np.ones(1)
        rot_angle = np.zeros(1)
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]
                flip_array = flip_array * -1


            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            rot_array = np.linalg.inv(np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                # print(np.max(rgb_color))
                # print(np.min(rgb_color))
                rgb_color *= (
                    1 + 0.4 * np.random.random(3) - 0.2
                )  # brightness change for each channel
                rgb_color += (
                    0.1 * np.random.random(3) - 0.05
                )  # color shift for each channel
                rgb_color += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(
                    np.random.random(point_cloud.shape[0]) > 0.3, -1
                )
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            scale_array = 1.0/scale_ratio
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio

            if self.use_height:  # False
                point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:  # True
                point_cloud, bboxes, _, bboxes_seenclass, bboxes_seenconfi, self.all_global_image, is_global_box, is_global_image = self.random_cuboid_augmentor(
                    point_cloud, bboxes, bboxes_seenclass=bboxes_seenclass, bboxes_seenconfi=bboxes_seenconfi, all_virtual_image=self.all_global_image, is_virtual_box=is_global_box, num_boxes_before_virtual_aug=num_boxes_before_global_aug, is_virtual_image=is_global_image
                )
        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : bboxes.shape[0]] = 1
        # max_bboxes = np.zeros((self.max_num_obj, 8))
        # max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        uv_2d = uv_2d[choices]
        uv_2d = np.round(uv_2d).astype(np.int64) - 1
        point_cloud_rgb = point_cloud_rgb[choices]

        point_cloud_dims_min = point_cloud[:, 0:3].min(axis=0)
        point_cloud_dims_max = point_cloud[:, 0:3].max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        box_corners_xyz = self.dataset_config.box_parametrization_to_corners_np_xyz(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners_xyz = box_corners_xyz.squeeze(0)

        ret_dict = {}
        ret_dict["global_pool_path"] = ALL_GLOBAL_OBJECT_PATH
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["point_clouds_rgb"] = point_cloud_rgb.astype(np.float32)
        ret_dict["point_clouds_rgb_ori"] = point_cloud_rbg_ori.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_corners_xyz"] = box_corners_xyz.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, 7]  # from 0 to 9

        target_bboxes_seen_semcls = np.zeros((self.max_num_obj))
        target_bboxes_seen_confi = np.zeros((self.max_num_obj))
        # print('=============3===============')
        # print(bboxes.shape)
        # print(bboxes_seenclass.shape)
        assert bboxes_seenclass.shape[0] == bboxes.shape[0]
        assert bboxes_seenconfi.shape[0] == bboxes.shape[0]

        # image_class_label = np.zeros(37)
        # if self.split_set == 'train':

        # else:
        #     image_class_label = np.zeros(self.dataset_config.test_max)

        image_class_label = np.zeros(self.dataset_config.train_max)
        target_bboxes_seen_semcls = target_bboxes_seen_semcls.astype(np.int64)
        target_bboxes_seen_confi = target_bboxes_seen_confi

        # if np.max(bboxes_seenclass) > 0:
        #     print(1)

        target_bboxes_seen_semcls[0: bboxes.shape[0]] = bboxes_seenclass
        target_bboxes_seen_confi[0: bboxes.shape[0]] = bboxes_seenconfi

        for present_idx, present_id in enumerate(target_bboxes_mask):
            if present_id > 0:
                class_id = target_bboxes_seen_semcls[present_idx]
                if class_id < self.dataset_config.train_max:
                    image_class_label[class_id] = 1

        final_is_global_box = np.zeros(self.max_num_obj)
        final_is_global_box[:is_global_box.shape[0]] = is_global_box
        ret_dict["gt_image_class_label"] = image_class_label.astype(np.int64)
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_seen_sem_cls_label"] = target_bboxes_seen_semcls.astype(np.int64)
        ret_dict["gt_box_seen_sem_cls_confi"] = target_bboxes_seen_confi.astype(np.float32)
        #print(ret_dict["gt_box_sem_cls_label"])
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["discovery_novel"] = np.zeros((self.dataset_config.args.nqueries))
        #print('5-idx:')
        #print(idx)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max
        ret_dict["pseudo_box_path"] = pseudo_box_path
        ret_dict["gt_ori_box_num"] = ori_num
        ret_dict["num_boxes_before_global_aug"] = num_boxes_before_global_aug
        ret_dict["all_global_image"] = self.all_global_image.astype(np.uint8) # shape: (5, 768, 768, 3) contents: which image is the keeping global image
        # ret_dict["all_global_image_mask"] = self.all_global_image_mask.astype(np.uint8) # shape: (5, 768, 768, 3) contents: which image is the keeping global image
        ret_dict["is_global_box"] = final_is_global_box # shape: (64) contents: which gt box is global box
        assert np.sum(final_is_global_box)==np.sum(is_global_image)
        image_idx = np.array(np.nonzero(is_global_image))[0]
        global_box_to_image_idx = np.zeros_like(final_is_global_box)
        global_box_to_image_idx[final_is_global_box>0] = image_idx
        ret_dict["is_global_image"] = is_global_image # shape: (5) contents: which image is the keeping global image
        ret_dict["global_box_to_image_idx"] = global_box_to_image_idx.astype(np.int64) # shape: (5) contents: which image is the keeping global image
        ret_dict["global_obj_num"] = self.global_obj_num
        # for idx_point in range(point_cloud.shape[0]):
        #     point_color_3d = point_cloud[idx_point, 3:] + MEAN_COLOR_RGB
        #     point_color_2d = self.input_image[uv_2d[idx_point, 1], uv_2d[idx_point, 0]] / 255.0
        #     print(1)
        # if if_tensor:
        #     self.K_tensor = torch.from_numpy(self.K).to('cuda').to(torch.float32)

        if self.if_input_image:
            lines = [line.rstrip() for line in open(self.calib_name)]
            Rtilt = np.array([float(x) for x in lines[0].split(' ')])
            Rtilt = np.reshape(Rtilt, (3, 3), order='F')
            # if if_tensor:
            #     self.Rtilt_tensor = torch.from_numpy(self.Rtilt).to('cuda').to(torch.float32)
            K = np.array([float(x) for x in lines[1].split(' ')])
            K = np.reshape(K, (3, 3), order='F')
            ret_dict["K"] = K
            ret_dict["Rtilt"] = Rtilt
            ret_dict["calib_name"] = self.calib_name
            ret_dict["uv_2d"] = uv_2d.astype(np.float32)
            ret_dict["input_image"] = self.input_image.astype(np.uint8)  # shape: [H, W, C] (R,G,B) dtype:uint8
            ret_dict["y_offset"] = self.y_offset
            ret_dict["x_offset"] = self.x_offset
            # ret_dict["trans_mtx"] = trans_mtx
            ret_dict["im_name"] = self.image_name
            ret_dict["ori_width"] = self.ori_width
            ret_dict["ori_height"] = self.ori_height
            ret_dict["flip_array"] = flip_array
            ret_dict["scale_array"] = scale_array
            ret_dict["rot_array"] = rot_array
            ret_dict["rot_angle"] = rot_angle
            ret_dict["image_flip_array"] = image_flip_array
            ret_dict["flip_length"] = self.image_size[0]
            if self.split_set == 'train' or 'toilettrain':
                ret_dict["box2d"] = box2d_final
                ret_dict["box2d_present"] = box2d_present
            # tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
            # write_ply_rgb(point_cloud[:, 0:3], ((point_cloud[:, 3:]+MEAN_COLOR_RGB)*255).astype(np.int8), 'show_outputs/tmp_point/'+'%s_pc.obj' % scan_name)
            # tmp_image = self.input_image
            # for idx_point in range(point_cloud.shape[0]):
            #     tmp_image[uv_2d[idx_point, 1], uv_2d[idx_point, 0]] = [0, 0, 255]
            # cv2.imwrite('show_outputs/tmp_point/'+'%s.png' % scan_name, tmp_image)
        # if not self.if_not_paste_generated_image:
        #     ret_dict = self.past_image(ret_dict, calib)
        return ret_dict
