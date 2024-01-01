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
                            get_3d_box_batch_np, get_3d_box_batch_tensor, get_3d_box_batch_tensor_xyz, get_3d_box_batch_np_xyz)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid
from datasets.scannet_utils import load_txt
# IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
here_path = os.getcwd()
if here_path.find('user-job-dir')!=-1:
    DATASET_ROOT_DIR = "/cache/data/make_data_scannet200_cluster/scannet_frames_25k_200cls_train"  ## Replace with path to dataset
    DATASET_PARAMS_DIR = "/cache/data/make_data_scannet200_cluster/matrix"
else:
    # DATASET_SCANS_DIR = "Data/scannet/scans/scans/"  ## Replace with path to dataset
    DATASET_ROOT_DIR = "Data/scannet/make_data_scannet200_cluster/scannet_frames_25k_200cls_train"
    DATASET_PARAMS_DIR = "Data/scannet/make_data_scannet200_cluster/matrix"

DATASET_CLASS_NAME_ID = "datasets/scannet_200_class2id.npy"
DATASET_METADATA_DIR = "datasets/meta_data" ## Replace with path to dataset


class ScannetAnonymousAlignedImageDatasetConfigWithNovelCateConfi(object):
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


class ScannetDetectionAlignedImageAnonymousDatasetWithNovelCateConfi(Dataset):
    def __init__(
            self,
            dataset_config,
            split_set="train",
            root_dir=None,
            meta_data_dir=None,
            num_points=40000,
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
            pseudo_bboxes_source_tmp = np.load(pseudo_scan_path + "_novel_bbox.npy")  # K,8
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

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scan_name = data_name.split('_')[0] + '_' + data_name.split('_')[1]
        squence_name = data_name.split('_')[-1]
         # = np.load(os.path.join(self.data_path, data_name) + "_pc.npy")  # (num_points, 6) (x,y,z,r,g,b)
        # instance_bboxes_read = np.load(os.path.join(self.data_path, data_name) + "_bbox.npy")  # (num_boxes, 7)
        mesh_vertices, instance_bboxes_read_source, instance_bboxes_read, pseudo_box_path, ori_num = self.load_boxes(data_name)
        if self.if_input_image:
            scan_param_path = os.path.join(self.param_path, scan_name)
            self.image_name = os.path.join(self.data_path,
                                           data_name) + ".jpg"  # remember to change it, just set 0.jpg now
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
        if len(bboxes_seen) == 0:
            instance_bboxes = np.zeros((0, 8))
        else:
            instance_bboxes = np.array(bboxes_seen)
        # for box_idx in range(instance_bboxes.shape[0]):
        #     instance_bboxes[box_idx, -1] = self.class_id_to_idx[instance_bboxes[box_idx, -1]]
        # instance_bboxes[:, -1] = 0
        bboxes_seenclass = np.array(bboxes_seenclass)
        bboxes_seenconfi = np.array(bboxes_seenconfi)

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
                # uv_2d[:, 0] = self.image_size[0] - 1 - uv_2d[:, 0]
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
                bboxes_seenconfi
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes, bboxes_seenclass=bboxes_seenclass, bboxes_seenconfi=bboxes_seenconfi # [instance_labels, semantic_labels]
            )
            # instance_labels = per_point_labels[0]
            # semantic_labels = per_point_labels[1]
        assert bboxes_seenclass.shape[0] == instance_bboxes.shape[0]
        assert bboxes_seenconfi.shape[0] == instance_bboxes.shape[0]

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
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
        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["point_clouds_rgb"] = point_cloud_rgb.astype(np.float32)
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
        if self.if_input_image:
            R_path = scan_param_path + '/pose/' + squence_name + '.txt'  # calib_filepath + '/pose/1.txt'  # change later, set 0.txt now.  contents:  camera_to_world matrix
            Rtilt = load_txt(R_path)
            K_path = scan_param_path + '/intrinsic/intrinsic_color.txt'
            K = load_txt(K_path)
            ret_dict["K"] = K
            ret_dict["Rtilt"] = Rtilt
            ret_dict["calib_name"] = scan_param_path
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
        return ret_dict

