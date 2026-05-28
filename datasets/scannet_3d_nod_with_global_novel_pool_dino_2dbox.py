# Copyright (c) Facebook, Inc. and its affiliates.

""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import cv2
import os
import sys
import glob
import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor, get_3d_box_batch_tensor_xyz, get_3d_box_batch_np_xyz, extract_pc_in_box3d)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid
from datasets.scannet_utils import load_txt
import datasets.scannet_utils as scannet_utils
import random
import pickle
from vis_color_pc import color_point
from utils.votenet_pc_util import write_oriented_bbox, write_ply

# IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
here_path = os.getcwd()
if here_path.find('user-job-dir')!=-1:
    DATASET_ROOT_DIR = "/cache/data/make_data_scannet200_cluster/scannet_frames_25k_200cls_train"  ## Replace with path to dataset
    DATASET_PARAMS_DIR = "/cache/data/make_data_scannet200_cluster/matrix"
    BOX2D_PATH = "/cache/data/make_data_scannet200_cluster/scannet_groundingDino_2dboxes"
    ALL_SUPERCLASS_PATH = "datasets/scannet_seen_and_obj365_cate_to_idx.npy"
    ALL_GLOBAL_OBJECT_PATH = "/cache/data/make_data_scannet200_cluster/scannet_frames_25k_200cls_noveltrain_global_novel_setting0/"
else:
    # DATASET_SCANS_DIR = "Data/scannet/"  ## Replace with path to dataset
    DATASET_ROOT_DIR = "Data/scannet/scannet_frames_25k/make_data_scannet200_cluster/scannet_frames_25k_200cls_train"
    DATASET_PARAMS_DIR = "Data/scannet/scannet_frames_25k/make_data_scannet200_cluster/matrix"
    BOX2D_PATH = 'Data/scannet/scannet_frames_25k/make_data_scannet200_cluster/scannet_groundingDino_2dboxes'
    ALL_SUPERCLASS_PATH = "datasets/scannet_seen_and_obj365_cate_to_idx.npy"
    ALL_GLOBAL_OBJECT_PATH = "Data/scannet/scannet_frames_25k/make_data_scannet200_cluster/scannet_frames_25k_200cls_noveltrain_global_novel_setting0/"

DATASET_CLASS_NAME_ID = "datasets/scannet_200_class2id.npy"
DATASET_METADATA_DIR = "datasets/meta_data" ## Replace with path to dataset


class Scannet3DNODGlobalNovelPoolDino2DboxConfig(object):
    def __init__(self, if_print=False, args=None):
        self.num_semcls = 1
        self.num_angle_bin = 12
        self.max_num_obj = 64
        self.args = args
        self.type2class = np.load(DATASET_CLASS_NAME_ID, allow_pickle=True).item()
        self.type2class['fg_only_for_anonymous'] = 0
        self.image_size = [args.image_size_width, args.image_size_height]  # (730, 531)
        self.pseudo_setting = args.pseudo_setting
        # self.type2class = {
        #     "cabinet": 0,
        #     "bed": 1,
        #     "chair": 2,
        #     "sofa": 3,
        #     "table": 4,
        #     "door": 5,
        #     "window": 6,
        #     "bookshelf": 7,
        #     "picture": 8,
        #     "counter": 9,
        #     "desk": 10,
        #     "curtain": 11,
        #     "refrigerator": 12,
        #     "showercurtrain": 13,
        #     "toilet": 14,
        #     "sink": 15,
        #     "bathtub": 16,
        #     "garbagebin": 17,
        # }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.train_range_list = args.train_range_list
        print('--------------------------train range list--------------------------')
        print(len(self.train_range_list))
        print(self.train_range_list)
        self.test_range_list = args.test_range_list
        print('--------------------------test range list---------------------------')
        print(len(self.test_range_list))
        print(self.test_range_list)
        self.class_id_to_idx = {2: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 13: 9, 14: 10, 15: 11, 16: 12,
                                17: 13, 18: 14, 19: 15, 21: 16, 22: 17, 23: 18, 24: 19, 26: 20, 27: 21, 28: 22, 29: 23,
                                31: 24, 32: 25, 33: 26, 34: 27, 35: 28, 36: 29, 38: 30, 39: 31, 40: 32, 41: 33, 42: 34,
                                44: 35, 45: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 54: 44, 55: 45,
                                56: 46, 57: 47, 58: 48, 59: 49, 62: 50, 63: 51, 64: 52, 65: 53, 66: 54, 67: 55, 68: 56,
                                69: 57, 70: 58, 71: 59, 72: 60, 73: 61, 74: 62, 75: 63, 76: 64, 77: 65, 78: 66, 79: 67,
                                80: 68, 82: 69, 84: 70, 86: 71, 87: 72, 88: 73, 89: 74, 90: 75, 93: 76, 95: 77, 96: 78,
                                97: 79, 98: 80, 99: 81, 100: 82, 101: 83, 102: 84, 103: 85, 104: 86, 105: 87, 106: 88,
                                107: 89, 110: 90, 112: 91, 115: 92, 116: 93, 118: 94, 120: 95, 121: 96, 122: 97,
                                125: 98, 128: 99, 130: 100, 131: 101, 132: 102, 134: 103, 136: 104, 138: 105, 139: 106,
                                140: 107, 141: 108, 145: 109, 148: 110, 154: 111, 155: 112, 156: 113, 157: 114,
                                159: 115, 161: 116, 163: 117, 165: 118, 166: 119, 168: 120, 169: 121, 170: 122,
                                177: 123, 180: 124, 185: 125, 188: 126, 191: 127, 193: 128, 195: 129, 202: 130,
                                208: 131, 213: 132, 214: 133, 221: 134, 229: 135, 230: 136, 232: 137, 233: 138,
                                242: 139, 250: 140, 261: 141, 264: 142, 276: 143, 283: 144, 286: 145, 300: 146,
                                304: 147, 312: 148, 323: 149, 325: 150, 331: 151, 342: 152, 356: 153, 370: 154,
                                392: 155, 395: 156, 399: 157, 408: 158, 417: 159, 488: 160, 540: 161, 562: 162,
                                570: 163, 572: 164, 581: 165, 609: 166, 748: 167, 776: 168, 1156: 169, 1163: 170,
                                1164: 171, 1165: 172, 1166: 173, 1167: 174, 1168: 175, 1169: 176, 1170: 177, 1171: 178,
                                1172: 179, 1173: 180, 1174: 181, 1175: 182, 1176: 183, 1178: 184, 1179: 185, 1180: 186,
                                1181: 187, 1182: 188, 1183: 189, 1184: 190, 1185: 191, 1186: 192, 1187: 193, 1188: 194,
                                1189: 195, 1190: 196, 1191: 197}
        self.seen_idx_list = []
        self.novel_idx_list = []
        for class_id in self.train_range_list:
            self.seen_idx_list.append(self.class_id_to_idx[class_id])


        for class_id in self.test_range_list:
            if class_id not in self.train_range_list:
                self.novel_idx_list.append(self.class_id_to_idx[class_id])

        self.type2superclass = self.get_supercate(ALL_SUPERCLASS_PATH)  # np.load(ALL_SUPERCLASS_PATH, allow_pickle=True).item()

        self.type2onehotsuperclass = self.get_supercate(ALL_SUPERCLASS_PATH)
        # print(1)
        # self.nyu40ids = np.array(
        #     [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        # )
        # self.nyu40id2class = {
        #     nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        # }

        # Semantic Segmentation Classes. Not used in 3DETR
        # self.num_class_semseg = 20
        # self.type2class_semseg = {
        #     "wall": 0,
        #     "floor": 1,
        #     "cabinet": 2,
        #     "bed": 3,
        #     "chair": 4,
        #     "sofa": 5,
        #     "table": 6,
        #     "door": 7,
        #     "window": 8,
        #     "bookshelf": 9,
        #     "picture": 10,
        #     "counter": 11,
        #     "desk": 12,
        #     "curtain": 13,
        #     "refrigerator": 14,
        #     "showercurtrain": 15,
        #     "toilet": 16,
        #     "sink": 17,
        #     "bathtub": 18,
        #     "garbagebin": 19,
        # }
        # self.class2type_semseg = {
        #     self.type2class_semseg[t]: t for t in self.type2class_semseg
        # }
        # self.nyu40ids_semseg = np.array(
        #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        # )
        # self.nyu40id2class_semseg = {
        #     nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        # }

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

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        # box_center_upright = box_center_unnorm
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np_xyz(self, box_center_unnorm, box_size, box_angle):
        # box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        # box_center_upright = box_center_unnorm
        boxes = get_3d_box_batch_np_xyz(box_size, -1 * box_angle, box_center_unnorm)
        return boxes

    def box_parametrization_to_corners_xyz(self, box_center_unnorm, box_size, box_angle):
        # box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor_xyz(box_size, -1 * box_angle, box_center_unnorm)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)

    def get_supercate(self, ALL_SUPERCLASS_PATH):
        class_list = np.load(ALL_SUPERCLASS_PATH, allow_pickle=True).item()
        new_class = {}

        for key_item in class_list.keys():
            new_key_item = key_item.replace(' ', '_').replace('/', '_or_')
            new_class[new_key_item] = class_list[key_item]

        return new_class

class Scannet3DNODGlobalNovelPoolDino2DboxDataLayer(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
            root_dir=None,
            meta_data_dir=None,
            num_points=20000,
            use_color=False,
            use_height=False,
            use_v1=False,
            augment=False,
            if_input_image=False,
            if_image_augment=False,
            use_random_cuboid=True,
            random_cuboid_min_points=30000,
    ):

        self.dataset_config = dataset_config
        self.type2superclass = self.dataset_config.type2superclass
        print(split_set)
        assert split_set in ["train", "val", "minival", "test", "minitest", "real_test", "real_cmp_test"]
        if root_dir is None:
            # scans_dir = DATASET_SCANS_DIR
            root_dir = DATASET_ROOT_DIR
        if split_set.find('train') == -1:
            root_dir = root_dir.replace('train', split_set)
        print('----------------------root dir-------------------------')
        print(split_set)
        print(split_set.find('train'))
        print(root_dir)
        self.max_num_obj = dataset_config.max_num_obj
        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR

        # self.scans_path = scans_dir
        self.data_path = root_dir
        self.pseudo_setting = dataset_config.pseudo_setting
        print('Adopt pseudo_setting: ')
        assert self.pseudo_setting=='setting0' or self.pseudo_setting=='setting1' or self.pseudo_setting=='setting2' or self.pseudo_setting=='setting3' or self.pseudo_setting=='setting4' or self.pseudo_setting=='debug'
        print(self.pseudo_setting)
        self.pseudo_data_path = self.data_path.replace('train', 'noveltrain_pseudo_labels_' + self.pseudo_setting)
        self.global_novel_path = self.data_path.replace('train', 'noveltrain_global_novel_'+self.pseudo_setting)
        os.makedirs(self.global_novel_path, exist_ok = True)
        os.makedirs(self.pseudo_data_path, exist_ok = True)
        self.split_set = split_set
        # all_scan_names = list(
        #     set(
        #         [
        #             os.path.basename(x)[0:12]
        #             for x in os.listdir(self.scans_path)
        #             if x.startswith("scene")
        #         ]
        #     )
        # )
        # if split_set == "all":
        #     self.scan_names = all_scan_names
        if split_set in ["train", "val", "minival", "test", "minitest", "real_test", "real_cmp_test"]:
            split_filenames = os.path.join(meta_data_dir, f"scannetv2_{split_set}.txt")
            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            # self.scan_names = [
            #     sname for sname in self.scan_names if sname in all_scan_names
            # ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
            self.data_names = []
            pc_list = glob.glob(self.data_path + '/*_pc.npy')
            for pc_item in pc_list:
                pc_name = os.path.basename(pc_item)[:-7]
                if pc_name not in self.data_names:
                    self.data_names.append(pc_name)
            print(f"kept {len(self.data_names)} images")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        if split_set.find('train') != -1:
            self.select_range_list = np.array(dataset_config.train_range_list)
        else:
            self.select_range_list = np.array(dataset_config.test_range_list)

        print('--------------------print select range-------------------')
        print(self.select_range_list)
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.image_augment = if_image_augment
        print('====================If adopt image augmentation:=====================')
        print(self.image_augment)
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.if_input_image = if_input_image
        self.reorder_list_seen_for_modal_align = {}
        for idx, class_id in enumerate(dataset_config.train_range_list):
            self.reorder_list_seen_for_modal_align[class_id] = idx
        print('---------------------if_input_image---------------------')
        print(self.if_input_image)
        self.image_size = dataset_config.image_size
        self.if_padding_image = True
        self.param_path = DATASET_PARAMS_DIR
        print('=================if paddng the input image==================')
        print(self.if_padding_image)
        print(self.image_size)
        self.class_id_to_idx = {2: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 13: 9, 14: 10, 15: 11, 16: 12,
                                17: 13, 18: 14, 19: 15, 21: 16, 22: 17, 23: 18, 24: 19, 26: 20, 27: 21, 28: 22, 29: 23,
                                31: 24, 32: 25, 33: 26, 34: 27, 35: 28, 36: 29, 38: 30, 39: 31, 40: 32, 41: 33, 42: 34,
                                44: 35, 45: 36, 46: 37, 47: 38, 48: 39, 49: 40, 50: 41, 51: 42, 52: 43, 54: 44, 55: 45,
                                56: 46, 57: 47, 58: 48, 59: 49, 62: 50, 63: 51, 64: 52, 65: 53, 66: 54, 67: 55, 68: 56,
                                69: 57, 70: 58, 71: 59, 72: 60, 73: 61, 74: 62, 75: 63, 76: 64, 77: 65, 78: 66, 79: 67,
                                80: 68, 82: 69, 84: 70, 86: 71, 87: 72, 88: 73, 89: 74, 90: 75, 93: 76, 95: 77, 96: 78,
                                97: 79, 98: 80, 99: 81, 100: 82, 101: 83, 102: 84, 103: 85, 104: 86, 105: 87, 106: 88,
                                107: 89, 110: 90, 112: 91, 115: 92, 116: 93, 118: 94, 120: 95, 121: 96, 122: 97,
                                125: 98, 128: 99, 130: 100, 131: 101, 132: 102, 134: 103, 136: 104, 138: 105, 139: 106,
                                140: 107, 141: 108, 145: 109, 148: 110, 154: 111, 155: 112, 156: 113, 157: 114,
                                159: 115, 161: 116, 163: 117, 165: 118, 166: 119, 168: 120, 169: 121, 170: 122,
                                177: 123, 180: 124, 185: 125, 188: 126, 191: 127, 193: 128, 195: 129, 202: 130,
                                208: 131, 213: 132, 214: 133, 221: 134, 229: 135, 230: 136, 232: 137, 233: 138,
                                242: 139, 250: 140, 261: 141, 264: 142, 276: 143, 283: 144, 286: 145, 300: 146,
                                304: 147, 312: 148, 323: 149, 325: 150, 331: 151, 342: 152, 356: 153, 370: 154,
                                392: 155, 395: 156, 399: 157, 408: 158, 417: 159, 488: 160, 540: 161, 562: 162,
                                570: 163, 572: 164, 581: 165, 609: 166, 748: 167, 776: 168, 1156: 169, 1163: 170,
                                1164: 171, 1165: 172, 1166: 173, 1167: 174, 1168: 175, 1169: 176, 1170: 177, 1171: 178,
                                1172: 179, 1173: 180, 1174: 181, 1175: 182, 1176: 183, 1178: 184, 1179: 185, 1180: 186,
                                1181: 187, 1182: 188, 1183: 189, 1184: 190, 1185: 191, 1186: 192, 1187: 193, 1188: 194,
                                1189: 195, 1190: 196, 1191: 197}
        self.confidence_type_in_datalayer = dataset_config.args.confidence_type_in_datalayer
        print('===============confidence_type_in_datalayer=================')
        print(self.confidence_type_in_datalayer)

        self.global_obj_path = ALL_GLOBAL_OBJECT_PATH + '%09d_%04d_box.npy' #glob.glob(ALL_GLOBAL_OBJECT_PATH+'*.npy')
        self.global_obj_num = len(glob.glob(ALL_GLOBAL_OBJECT_PATH + '*_box.npy')) // self.dataset_config.args.ngpus
        print('=========================the number of global obj==========================')
        #
        # self.virtual_objs = self.get_virtual_obj_list(ALL_VIRTUAL_OBJECT_PATH) #glob.glob(ALL_VIRTUAL_OBJECT_PATH+'*.npy')
        # self.virtual_obj_num = len(self.virtual_objs)
        # print('=========================the number of virtual obj==========================')
        # print(self.virtual_obj_num)
        # if self.virtual_sample_num > 0 and self.virtual_cate_num > 0:
        #     assert self.virtual_obj_num > 0
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

        self.if_need_2d_box = not self.dataset_config.args.if_not_need_2d_box
        print('===================if_need_2d_box=====================')
        print(self.if_need_2d_box)
        self.scannet10_and_captions_path = 'datasets/scannet_10seen_and_captions_953.npy'
        self.scannet10_and_captions_cate_to_id = np.load(self.scannet10_and_captions_path, allow_pickle=True).item()
        assert len(self.scannet10_and_captions_cate_to_id) == 953


    def __len__(self):
        return len(self.data_names)


    def load_boxes(self, scan_name):
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
            pseudo_scan_path = os.path.join(self.pseudo_data_path, scan_name)
        point_cloud = np.load(scan_path + "_pc.npy")
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

            pseudo_bboxes_source_tmp[:, 3:6] = pseudo_bboxes_source_tmp[:, 3:6] / 2
            pseudo_bboxes_source_tmp[:, 6] = pseudo_bboxes_source_tmp[:, 6] * -1
            pseudo_bboxes_source = np.zeros((pseudo_bboxes_source_tmp.shape[0], 11)) # the last one to see if it is a pseudo label, 0 is a pseudo labels
            pseudo_bboxes_source[:, :pseudo_bboxes_source_tmp.shape[1]] = pseudo_bboxes_source_tmp
            combined_bboxes_source = np.concatenate((bboxes_source, pseudo_bboxes_source), axis=0)
            combined_bboxes = combined_bboxes_source.copy()
            pseudo_box_path = pseudo_scan_path + "_novel_bbox.npy"

        else:
            combined_bboxes_source = bboxes_source
            combined_bboxes = combined_bboxes_source.copy()
            pseudo_box_path = '_'
            # pseudo_bboxes = pseudo_bboxes_source.copy()
        ori_num = combined_bboxes_source.shape[0]
        return point_cloud, combined_bboxes_source, combined_bboxes, pseudo_box_path, ori_num

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


    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scan_name = data_name.split('_')[0] + '_' + data_name.split('_')[1]
        self.global_obj_num = len(glob.glob(ALL_GLOBAL_OBJECT_PATH + '*_box.npy')) // self.dataset_config.args.ngpus
        squence_name = data_name.split('_')[-1]

         # = np.load(os.path.join(self.data_path, data_name) + "_pc.npy")  # (num_points, 6) (x,y,z,r,g,b)
        # instance_bboxes_read = np.load(os.path.join(self.data_path, data_name) + "_bbox.npy")  # (num_boxes, 7)
        mesh_vertices, instance_bboxes_read_source, instance_bboxes_read, pseudo_box_path, ori_num = self.load_boxes(data_name)
        point_cloud_rbg_ori = mesh_vertices.copy()
        point_cloud_rbg_ori, choices = pc_util.random_sampling(
            point_cloud_rbg_ori, 120000, return_choices=True
        )
        mesh_vertices_check_pos, choices = pc_util.random_sampling(
            mesh_vertices, 50000, return_choices=True
        )
        # color_point(mesh_vertices, save_path='1.ply')
        # color_point(mesh_vertices_check_pos, save_path='2.ply')
        if self.if_input_image:
            scan_param_path = os.path.join(self.param_path, scan_name)
            self.image_name = os.path.join(self.data_path,
                                           data_name) + ".jpg"  # remember to change it, just set 0.jpg now
            self.input_image = cv2.imread(self.image_name)
            self.input_image = np.array(self.input_image, dtype=np.float32)

            calib = scannet_utils.SCANNET_Calibration(scan_param_path, squence_name, if_tensor=False)

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
                    trans_mtx = np.zeros((1))
                    # scale_h = 448 * 1.0 / self.image_size[1]
                    # scale_w = 448 * 1.0 / self.image_size[0]
                    # trans_mtx = np.eye(2, 2)
                    # trans_mtx[0, 0] *= scale_w
                    # trans_mtx[1, 1] *= scale_h
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

            self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
            self.input_image = np.array(self.input_image, dtype=np.uint8)

        # instance_labels = np.load(
        #     os.path.join(self.data_path, scan_name) + "_ins_label.npy" # (num_points) label of instances, 0 means unannotated
        # )
        # semantic_labels = np.load(
        #     os.path.join(self.data_path, scan_name) + "_sem_label.npy" # (num_points) label of semantic, 0 means unannotated
        # )

        # bbox_mask = np.in1d(instance_bboxes_read[:, -1], self.select_range_list)
        #
        # instance_bboxes = instance_bboxes_read[bbox_mask]
        # instance_bboxes[:, -1] = 0
        # if self.if_need_2d_box:
        box2d_path = os.path.join(self.box2d_path, '%s_%s_2d_boxes.pkl' % (scan_name, squence_name))
        assert os.path.exists(box2d_path)

        box2d_present = np.zeros(128)
        with open(box2d_path, 'rb') as f:
            box2d_predictions = pickle.load(f) #crop image use (1, 3, 2, 0)
        box2d_num = box2d_predictions.xyxy.shape[0]
        box2d_original = np.zeros((box2d_num, 6))
        box2d_original[:, :4] = box2d_predictions.xyxy
        box2d_original[:, 5] = box2d_predictions.confidence
        for idx_box2d in range(box2d_num):
            tmp_cate = box2d_predictions.class_name[idx_box2d]
            box2d_original[idx_box2d, 4] = self.scannet10_and_captions_cate_to_id[tmp_cate]  # wait for debug

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
        box2d[:, 2] += self.y_offset
        box2d[:, 1] += self.x_offset
        box2d[:, 3] += self.x_offset
        box2d_final = np.zeros((128, 6))
        box2d_present[:box2d.shape[0]] = 1
        box2d_final[:box2d.shape[0]] = box2d


        # # ###debuging by saving points
        # write_ply(mesh_vertices, 'check_tmp/before_pc.ply')
        # global_boxes[:, 6] = -1 * global_boxes[:, 6]
        # write_oriented_bbox(global_boxes, 'global_box.ply')
        # # ###debuging by saving points

        ############ load global objects
        tmp_global_obj_num = np.random.randint(0, high=self.max_global_obj+1, size=1)[0]
        self.all_global_image = np.zeros((self.max_global_obj, 224, 224, 3))
        global_boxes_list = []
        # write_ply(point_cloud, 'debug_ori_7.ply')
        cnt_global_box = 0
        tmp_idx_obj = 0
        while (tmp_idx_obj < (tmp_global_obj_num)):
            if self.global_obj_num == 0:  # there is no global objects in the pool
                break
            global_idx = np.random.randint(0, high=self.global_obj_num, size=1)[0]
            gpu_idx = np.random.randint(0, high=self.dataset_config.args.ngpus, size=1)[0]
            global_box_path = self.global_obj_path % (global_idx, gpu_idx)
            # global_obj_class = os.path.basename(global_obj_path)[:-18]
            # assert global_obj_class in self.dataset_config.type2class.keys()

            global_2d_obj_path = global_box_path.replace('box.npy', 'im.png')
            global_obj_path = global_box_path.replace('box.npy', 'pc.npy')

            if not (os.path.exists(global_box_path) and os.path.exists(global_2d_obj_path) and os.path.exists(
                    global_obj_path)):
                continue

            try:
                global_2d_obj = self.get_global_image(global_2d_obj_path)
                # # # ###debuging by saving points
                # cv2.imwrite('check_tmp/%04d.png' % tmp_idx_obj, global_2d_obj)
                # # # ###debuging by saving points

                global_obj = np.load(global_obj_path)
                global_boxes = np.load(global_box_path)[np.newaxis, :]
            except:
                continue
            # global_obj_class_id = global_boxes[-1]

            global_obj_moved, global_boxes_moved = self.random_move(mesh_vertices, global_obj, global_boxes)
            recheck_time = 0
            if_found_pos = True
            while self.check_conclusion(mesh_vertices_check_pos, global_boxes_moved) or self.check_2d(self.input_image,
                                                                                          global_boxes_moved,
                                                                                          self.ori_height,
                                                                                          self.ori_width, self.y_offset,
                                                                                          self.x_offset, calib):
                global_obj_moved, global_boxes_moved = self.random_move(mesh_vertices, global_obj, global_boxes)
                recheck_time += 1
                if recheck_time > 100:  # here seems has a bug
                    if_found_pos = False
                    break
            if not if_found_pos:
                #     print('Found bad case!!!!!!!!!!1')
                continue
            self.all_global_image[cnt_global_box, :, :, :] = global_2d_obj
            # self.all_global_image_mask[cnt_global_box, :, :] = global_2d_obj_mask
            cnt_global_box += 1
            global_obj_select = global_obj_moved  # self.random_select(global_obj_moved)
            mesh_vertices = np.concatenate((mesh_vertices, global_obj_select), axis=0)
            global_boxes_list.append(global_boxes_moved)
            tmp_idx_obj += 1

        if len(global_boxes_list) == 0:
            global_boxes = np.zeros((0, 8))
        else:
            global_boxes = np.concatenate(global_boxes_list, axis=0)

        # # ###debuging by saving points
        # write_ply(mesh_vertices, 'check_tmp/after_pc.ply')
        # global_boxes[:, 6] = -1 * global_boxes[:, 6]
        # write_oriented_bbox(global_boxes, 'check_tmp/global_box.ply')
        # # ###debuging by saving points

        global_boxes[:, 3:6] = global_boxes[:, 3:6] / 2  # need to divide 2 before concating

        cnt_bboxes = 0
        bboxes_seen = []
        bboxes_seenclass = []
        bboxes_seenconfi = []
        bboxes = instance_bboxes_read_source.copy()

        for idx_box in range(bboxes.shape[0]):
            # print('2-idx-box:')
            # print(idx_box)
            #    print(bboxes[idx_box, -1])
            # if (10 <= bboxes[idx_box, -1]) and (20 > bboxes[idx_box, -1]):
            assert bboxes[idx_box, -1] == 1 or bboxes[idx_box, -1] == 0
            if bboxes[idx_box, -1] == 1:  # real label
                if (bboxes[idx_box, 7] in self.select_range_list):  # and (20 > bboxes[idx_box, -1]):
                    bboxes_seenclass.append(self.reorder_list_seen_for_modal_align[bboxes[idx_box, 7]]) # need to recode for the alignment supercategory list
                    bboxes[idx_box, 7] = 0  # bboxes[idx_box, -1] - self.dataset_config.test_range[0]
                    cnt_bboxes += 1
                    bboxes_seen.append(bboxes[idx_box][:8])
                    bboxes_seenconfi.append(1.0)  # confi for real label is 1
            elif bboxes[idx_box, -1] == 0:  # pseudo labels
                bboxes_seenclass.append(bboxes[idx_box, 7]) # it is saved pseudo labels, no need to reorder
                bboxes[idx_box, 7] = 0  # bboxes[idx_box, -1] - self.dataset_config.test_range[0]
                cnt_bboxes += 1
                bboxes_seen.append(bboxes[idx_box][:8])
                if self.confidence_type_in_datalayer == 'clip-max-prob':
                    confi_tmp = bboxes[
                        idx_box, 8]  # confi for pseudo label, adopt class prob now, do not use objectness
                    bboxes_seenconfi.append(confi_tmp)
                elif self.confidence_type_in_datalayer == 'zero-out':
                    bboxes_seenconfi.append(0)
                elif self.confidence_type_in_datalayer == 'objectness':
                    confi_tmp = bboxes[
                        idx_box, 9]  # confi for pseudo label, adopt class prob now, do not use objectness
                    bboxes_seenconfi.append(confi_tmp)
                elif self.confidence_type_in_datalayer == 'clip+objectness':
                    confi_tmp = (bboxes[idx_box, 8] + bboxes[
                        idx_box, 9]) / 2.0  # confi for pseudo label, adopt class prob now, do not use objectness
                    bboxes_seenconfi.append(confi_tmp)
                elif self.confidence_type_in_datalayer == 'weight_one':
                    bboxes_seenconfi.append(1)
        #        print('seen')
        #    else:
        #        print('unseen')
        num_boxes_before_global_aug = len(bboxes_seen)
        is_global_box = np.zeros(num_boxes_before_global_aug + global_boxes.shape[0])
        is_global_image = np.zeros(self.max_global_obj)
        for idx_box in range(global_boxes.shape[0]):
            is_global_box[num_boxes_before_global_aug + idx_box] = 1
            is_global_image[idx_box] = 1
            # print('2-idx-box:')
            # print(idx_box)
            #    print(bboxes[idx_box, -1])
            # if (10 <= bboxes[idx_box, -1]) and (20 > bboxes[idx_box, -1]):
            bboxes_seenclass.append(global_boxes[idx_box, -1])
            global_boxes[idx_box, -1] = 0  # bboxes[idx_box, -1] - self.dataset_config.test_range[0]
            cnt_bboxes += 1
            bboxes_seen.append(global_boxes[idx_box])
            bboxes_seenconfi.append(1)  # all confi weights of global objects are set to 1

        if len(bboxes_seen) == 0:
            instance_bboxes = np.zeros((0, 8))
        else:
            instance_bboxes = np.array(bboxes_seen)
        # for box_idx in range(instance_bboxes.shape[0]):
        #     instance_bboxes[box_idx, -1] = self.class_id_to_idx[instance_bboxes[box_idx, -1]]
        # instance_bboxes[:, -1] = 0
        bboxes_seenclass = np.array(bboxes_seenclass)
        bboxes_seenconfi = np.array(bboxes_seenconfi)

        uv_2d, _, d = calib.project_upright_depth_to_image(mesh_vertices[:, :3])
        uv_2d[:, 0] += self.y_offset
        uv_2d[:, 1] += self.x_offset
        uv_2d[:, 0] = np.clip(uv_2d[:, 0], 0, self.image_size[0] - 1)
        uv_2d[:, 1] = np.clip(uv_2d[:, 1], 0, self.image_size[1] - 1)


        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            point_cloud_rgb = mesh_vertices[:, 0:6]
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)


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
                if self.if_need_2d_box:
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

        #        ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 7), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)


        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
                bboxes_seenclass,
                bboxes_seenconfi,
                self.all_global_image, is_global_box, is_global_image
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes, bboxes_seenclass=bboxes_seenclass, bboxes_seenconfi=bboxes_seenconfi, all_virtual_image=self.all_global_image, is_virtual_box=is_global_box, num_boxes_before_virtual_aug=num_boxes_before_global_aug, is_virtual_image=is_global_image # [instance_labels, semantic_labels]
            )
            # instance_labels = per_point_labels[0]
            # semantic_labels = per_point_labels[1]
        assert bboxes_seenclass.shape[0] == instance_bboxes.shape[0]
        assert bboxes_seenconfi.shape[0] == instance_bboxes.shape[0]

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        uv_2d = uv_2d[choices]
        uv_2d = np.round(uv_2d).astype(np.int64) - 1
        # instance_labels = instance_labels[choices]
        # semantic_labels = semantic_labels[choices]

        # sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

        # for _c in self.dataset_config.nyu40ids_semseg:
        #     sem_seg_labels[
        #         semantic_labels == _c
        #     ] = self.dataset_config.nyu40id2class_semseg[_c]

        pcl_color = pcl_color[choices]
        point_cloud_rgb = point_cloud_rgb[choices]
        target_bboxes_mask[0: instance_bboxes.shape[0]] = 1
        target_bboxes[0: instance_bboxes.shape[0], :] = instance_bboxes[:, 0:7]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        rot_array = np.identity(3)
        scale_array = np.ones((1, 3))
        flip_array = np.ones(1)
        rot_angle = np.zeros(1)
        zx_flip_array = np.ones(1)

        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                point_cloud_rgb[:, 0] = -1 * point_cloud_rgb[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]
                flip_array = flip_array * -1
                target_bboxes[:, 6] = np.pi - target_bboxes[:, 6]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                point_cloud_rgb[:, 1] = -1 * point_cloud_rgb[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]
                zx_flip_array = zx_flip_array * -1
                target_bboxes[:, 6] = np.pi - target_bboxes[:, 6]

            # # Rotation along up-axis/Z-axis
            # rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            # rot_mat = pc_util.rotz(rot_angle)
            # point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            # target_bboxes = self.dataset_config.rotate_aligned_boxes(
            #     target_bboxes, rot_mat
            # )

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            point_cloud_rgb[:, 0:3] = np.dot(point_cloud_rgb[:, 0:3], np.transpose(rot_mat))
            target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
            rot_array = np.linalg.inv(np.transpose(rot_mat))
            target_bboxes[:, 6] -= rot_angle

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
            point_cloud_rgb[:, 0:3] *= scale_ratio
            target_bboxes[:, 0:3] *= scale_ratio
            target_bboxes[:, 3:6] *= scale_ratio


        raw_sizes = target_bboxes[:, 3:6] * 2 * target_bboxes_mask[..., None]  # size times 2
        raw_angles = target_bboxes[:, 6] * -1 * target_bboxes_mask  # angle times -1
        raw_angles_copy = raw_angles.copy()
        for box_idx in range(instance_bboxes.shape[0]):
            angle_class, angle_residual = self.dataset_config.angle2class(raw_angles_copy[box_idx])
            angle_classes[box_idx] = angle_class
            angle_residuals[box_idx] = angle_residual

        angle_classes = angle_classes * target_bboxes_mask
        angle_residuals = angle_residuals * target_bboxes_mask
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

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
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

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

        target_bboxes_seen_semcls = np.zeros((self.max_num_obj))
        target_bboxes_seen_confi = np.zeros((self.max_num_obj))
        target_bboxes_seen_semcls = target_bboxes_seen_semcls.astype(np.int64)
        target_bboxes_seen_confi = target_bboxes_seen_confi
        target_bboxes_seen_semcls[0: instance_bboxes.shape[0]] = bboxes_seenclass
        target_bboxes_seen_confi[0: instance_bboxes.shape[0]] = bboxes_seenconfi

        final_is_global_box = np.zeros(self.max_num_obj)
        final_is_global_box[:is_global_box.shape[0]] = is_global_box

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
        ret_dict["gt_box_seen_sem_cls_label"] = target_bboxes_seen_semcls.astype(np.int64)
        ret_dict["gt_box_seen_sem_cls_confi"] = target_bboxes_seen_confi.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0: instance_bboxes.shape[0]] = [
            # self.dataset_config.nyu40id2class[int(x)]
            x
            for x in instance_bboxes[:, 7][0: instance_bboxes.shape[0]]
        ]
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_seen_sem_cls_label"] = target_bboxes_seen_semcls.astype(np.int64)
        ret_dict["gt_box_seen_sem_cls_confi"] = target_bboxes_seen_confi.astype(np.float32)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
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

        if self.if_input_image:
            R_path = scan_param_path + '/pose/' + squence_name + '.txt'  # calib_filepath + '/pose/1.txt'  # change later, set 0.txt now.  contents:  camera_to_world matrix
            Rtilt = load_txt(R_path)
            K_path = scan_param_path + '/intrinsic/intrinsic_color.txt'
            K = load_txt(K_path)
            ret_dict["K"] = K
            ret_dict["Rtilt"] = Rtilt
            ret_dict["calib_name"] = scan_param_path
            ret_dict["uv_2d"] = uv_2d.astype(np.float32)
            ret_dict["squence_name"] = squence_name
            ret_dict["input_image"] = self.input_image
            ret_dict["y_offset"] = self.y_offset
            ret_dict["x_offset"] = self.x_offset
            ret_dict["ori_width"] = self.ori_width
            ret_dict["ori_height"] = self.ori_height
            ret_dict["trans_mtx"] = trans_mtx
            ret_dict["im_name"] = self.image_name
            ret_dict["data_name"] = data_name
            ret_dict["flip_array"] = flip_array
            ret_dict["zx_flip_array"] = zx_flip_array
            ret_dict["scale_array"] = scale_array
            ret_dict["rot_array"] = rot_array
            ret_dict["rot_angle"] = rot_angle
            ret_dict["image_flip_array"] = image_flip_array
            ret_dict["flip_length"] = self.image_size[0]

        if self.if_need_2d_box:
            ret_dict["box2d"] = box2d_final
            ret_dict["box2d_present"] = box2d_present

        # point_cloud_rgb
        # tmp_point_clouds_np = point_cloud_rgb_this_batch[sample_idx].cpu().numpy()
        # write_ply_rgb(point_cloud[:, 0:3], ((point_cloud[:, 3:]+MEAN_COLOR_RGB)*255).astype(np.int8), 'show_outputs/tmp_point/'+'%s_pc.obj' % scan_name)
        # center_uv2d, _, d = calib.project_upright_depth_to_image(instance_bboxes_read_source[:, :3])
        # center_uv2d = np.round(center_uv2d).astype(np.int64) - 1
        # color_point(point_cloud_rgb, save_path='check_points/'+'%s_%s.ply' % (scan_name, squence_name))
        #
        # instance_bboxes_read_source[:-1] = instance_bboxes_read_source[:-1]*-1
        # write_oriented_bbox(instance_bboxes_read_source,
        #                     'check_points/'+'%s_%s_bbox.ply' % (scan_name, squence_name))
        #
        # tmp_image = self.input_image
        #
        #
        # for idx_point in range(center_uv2d.shape[0]):
        #     tmp_image[center_uv2d[idx_point, 1], center_uv2d[idx_point, 0], :] = [0, 0, 255]
        # cv2.imwrite('check_points/'+'%s_%s.png' % (scan_name, squence_name), tmp_image)
        # if not self.if_not_paste_generated_image:
        #     ret_dict = self.past_image(ret_dict, calib)
        return ret_dict

