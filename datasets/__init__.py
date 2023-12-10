# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet_anonymous import ScannetDetectionAnonymousDataset, ScannetAnonymousDatasetConfig
from .scannet_anonymous_aligned_image import ScannetDetectionAlignedImageAnonymousDataset, ScannetAnonymousAlignedImageDatasetConfig
from .scannet_anonymous_aligned_image_with_novel_cate_confi import ScannetDetectionAlignedImageAnonymousDatasetWithNovelCateConfi, ScannetAnonymousAlignedImageDatasetConfigWithNovelCateConfi
from .scannet50_image import Scannet50DetectionImageDataset, Scannet50ImageDatasetConfig
from .sunrgbd_image import SunrgbdImageDetectionDataset, SunrgbdImageDatasetConfig
from .sunrgbd_anonymous_aligned_image import SunrgbdAnonymousAlignedImageDetectionDataset, SunrgbdAnonymousAlignedImageDatasetConfig
from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi import SunrgbdAnonymousAlignedImageDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageDatasetConfigWithNovelCateConfi
from .sunrgbd_cmp_image import SunrgbdImageCmpDetectionDataset, SunrgbdImageCmpDatasetConfig
from .scannet_cmp_image import ScannetDetectionImageCmpDataset, ScannetImageCmpDatasetConfig
from .sunrgbd_anonymous_aligned_image_object_aug import SunrgbdAnonymousAlignedImageObjectAugDetectionDataset, SunrgbdAnonymousAlignedImageObjectAugDatasetConfig
DATASET_FUNCTIONS = {
    "scannet_anonymous": [ScannetDetectionAnonymousDataset, ScannetAnonymousDatasetConfig],
    "scannet50_image": [Scannet50DetectionImageDataset, Scannet50ImageDatasetConfig],
    "scannet_anonymous_aligned_image": [ScannetDetectionAlignedImageAnonymousDataset, ScannetAnonymousAlignedImageDatasetConfig],
    "scannet_anonymous_aligned_image_with_novel_cate_confi": [ScannetDetectionAlignedImageAnonymousDatasetWithNovelCateConfi, ScannetAnonymousAlignedImageDatasetConfigWithNovelCateConfi],
    "sunrgbd_image": [SunrgbdImageDetectionDataset, SunrgbdImageDatasetConfig],
    "sunrgbd_anonymous_aligned_image": [SunrgbdAnonymousAlignedImageDetectionDataset, SunrgbdAnonymousAlignedImageDatasetConfig],
    "sunrgbd_anonymous_aligned_image_with_novel_cate_confi": [SunrgbdAnonymousAlignedImageDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageDatasetConfigWithNovelCateConfi],
    "sunrgbd_anonymous_aligned_image_object_aug": [SunrgbdAnonymousAlignedImageObjectAugDetectionDataset, SunrgbdAnonymousAlignedImageObjectAugDatasetConfig]
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1](if_print=True, args=args)

    # if args.dataset_name=='sunrgbd_image' or args.dataset_name=="sunrgbd_anonymous_image":
    #     if_augment = True #False  # TODO： need to train it with augmentation(right for both image and PC)
    # else:
    #     if_augment = True

    if args.dataset_name.find('scannet')!=-1:
        # real_test_config = ScannetImageDatasetConfig(if_print=True, args=args)
        real_cmp_config = ScannetImageCmpDatasetConfig(if_print=True, args=args)
        # real_dataset_builder = ScannetDetectionImageDataset # wait to change
        real_cmp_builder = ScannetDetectionImageCmpDataset # wait to change
        real_test_config = Scannet50ImageDatasetConfig(if_print=True, args=args)
        real_dataset_builder = Scannet50DetectionImageDataset # wait to change
    else:
        real_test_config = SunrgbdImageDatasetConfig(if_print=True, args=args)
        real_cmp_config = SunrgbdImageCmpDatasetConfig(if_print=True, args=args)
        real_dataset_builder = SunrgbdImageDetectionDataset
        real_cmp_builder = SunrgbdImageCmpDetectionDataset

    dataset_dict = {
        "train": dataset_builder(
            dataset_config,
            # split_set="noveltrain",
            split_set="train",
            # split_set="minival",
            # split_set="toilettrain",
            root_dir=args.dataset_root_dir, 
            meta_data_dir=args.meta_data_dir, 
            use_color=args.use_color,
            use_v1=args.if_use_v1,
            augment=True,
            if_input_image=args.if_input_image,
            if_image_augment=args.if_image_augment,
        ),
        "test": dataset_builder(
            dataset_config,
            # split_set="novelval",
            split_set="val",
            # split_set="chairval",
            # split_set="train",
            # split_set="minival",
            # split_set="papershow",
            # split_set="minival",
            # split_set="noveltrain",
            # split_set="toiletval",
            # split_set="tableval",
            # split_set="coffeetableval",
            # split_set="endtableval",
            # split_set="train",
            root_dir=args.dataset_root_dir, 
            use_color=args.use_color,
            use_v1=args.if_use_v1,
            augment=False,
            if_input_image=args.if_input_image
        ),
        "minitest": dataset_builder(
            dataset_config,
            split_set="minival",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            use_v1=args.if_use_v1,
            augment=False,
            if_input_image=args.if_input_image
        ),
        "real_test": real_dataset_builder(
            real_test_config,
            # split_set="novelval",
            split_set="val",
            # split_set="papershow",
            # split_set="minival",
            # split_set="noveltrain",
            # split_set="train",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            use_v1=args.if_use_v1,
            augment=False,
            if_input_image=args.if_input_image
        ),
        "real_cmp_test": real_cmp_builder(
            real_cmp_config,
            # split_set="novelval",
            split_set="val",
            # split_set="minival",
            # split_set="noveltrain",
            # split_set="train",
            root_dir=args.dataset_root_dir,
            use_color=args.use_color,
            use_v1=args.if_use_v1,
            augment=False,
            if_input_image=args.if_input_image
        ),
    }
    return dataset_dict, dataset_config, real_test_config, real_cmp_config
    
