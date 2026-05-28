# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet_anonymous import ScannetDetectionAnonymousDataset, ScannetAnonymousDatasetConfig
from .scannet_anonymous_aligned_image import ScannetDetectionAlignedImageAnonymousDataset, ScannetAnonymousAlignedImageDatasetConfig
from .scannet_anonymous_aligned_image_with_novel_cate_confi import ScannetDetectionAlignedImageAnonymousDatasetWithNovelCateConfi, ScannetAnonymousAlignedImageDatasetConfigWithNovelCateConfi
from .scannet50_image import Scannet50DetectionImageDataset, Scannet50ImageDatasetConfig
from .sunrgbd_image import SunrgbdImageDetectionDataset, SunrgbdImageDatasetConfig
from .sunrgbd_anonymous_aligned_image import SunrgbdAnonymousAlignedImageDetectionDataset, SunrgbdAnonymousAlignedImageDatasetConfig
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi import SunrgbdAnonymousAlignedImageDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageDatasetConfigWithNovelCateConfi
from .sunrgbd_cmp_image import SunrgbdImageCmpDetectionDataset, SunrgbdImageCmpDatasetConfig
from .scannet_cmp_image import ScannetDetectionImageCmpDataset, ScannetImageCmpDatasetConfig
# from .sunrgbd_anonymous_aligned_image_object_aug import SunrgbdAnonymousAlignedImageObjectAugDetectionDataset, SunrgbdAnonymousAlignedImageObjectAugDatasetConfig
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug import SunrgbdAnonymousAlignedImageObjectAugDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugDatasetConfigWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_not_align import SunrgbdAnonymousAlignedImageObjectAugNotAlignDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugNotAlignDatasetConfigWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_load_2d_box import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteReal2dBoxWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteReal2dBoxWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPosWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPosWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPos2dBoxWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPos2dBoxWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_imsize_prior import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPos2dBoxImsizePriorWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPos2dBoxImsizePriorWithNovelCateConfi
# from .sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_density_adaptation import SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPos2dBoxDensityAdaptWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPos2dBoxDensityAdaptWithNovelCateConfi
from .sunrgbd_sample import SunrgbdSampleDataset, SunrgbdSampleCateConfi
# from .scannet_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box import ScannetAnonymousAlignedImageDatasetVirtual3DImagePasteRealCheckPos2dBoxConfigWithNovelCateConfi, ScannetDetectionAlignedImageAnonymousDatasetVirtual3DImagePasteRealCheckPos2dBoxWithNovelCateConfi
# from .scannet_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_less_data import ScannetAnonymousAlignedImageDatasetVirtual3DImagePasteRealCheckPos2dBoxConfigWithNovelCateConfiLessData, ScannetDetectionAlignedImageAnonymousDatasetVirtual3DImagePasteRealCheckPos2dBoxWithNovelCateConfiLessData
# from .scannet_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_less_data_reload import ScannetAnonymousAlignedImageDatasetVirtual3DImagePasteRealCheckPos2dBoxConfigWithNovelCateConfiLessDataReload, ScannetDetectionAlignedImageAnonymousDatasetVirtual3DImagePasteRealCheckPos2dBoxWithNovelCateConfiLessDataReload
from .sunrgbd_3d_nod_with_global_novel_pool import Sunrgbd3DNODGlobalNovelPoolDataLayer, Sunrgbd3DNODGlobalNovelPoolConfig
from .sunrgbd_3d_nod_with_global_novel_pool_dino_2dbox import Sunrgbd3DNODGlobalNovelPoolDino2DboxDataLayer, Sunrgbd3DNODGlobalNovelPoolDino2DboxConfig
# from .sunrgbd_3d_nod_with_global_novel_pool_dino_2dbox_with_layout_prior import Sunrgbd3DNODGlobalNovelPoolDino2DboxLayoutPriorDataLayer, Sunrgbd3DNODGlobalNovelPoolDino2DboxLayoutPriorConfig
from .scannet_3d_nod_with_global_novel_pool_dino_2dbox import Scannet3DNODGlobalNovelPoolDino2DboxDataLayer, Scannet3DNODGlobalNovelPoolDino2DboxConfig
# from .sunrgbd_3d_nod_with_global_novel_pool_dino_2dbox_show_im import Sunrgbd3DNODGlobalNovelPoolDino2DboxShowImDataLayer, Sunrgbd3DNODGlobalNovelPoolDino2DboxShowImConfig

DATASET_FUNCTIONS = {
    "scannet_anonymous": [ScannetDetectionAnonymousDataset, ScannetAnonymousDatasetConfig],
    "scannet50_image": [Scannet50DetectionImageDataset, Scannet50ImageDatasetConfig],
    "scannet_anonymous_aligned_image": [ScannetDetectionAlignedImageAnonymousDataset, ScannetAnonymousAlignedImageDatasetConfig],
    "scannet_anonymous_aligned_image_with_novel_cate_confi": [ScannetDetectionAlignedImageAnonymousDatasetWithNovelCateConfi, ScannetAnonymousAlignedImageDatasetConfigWithNovelCateConfi],
    "sunrgbd_image": [SunrgbdImageDetectionDataset, SunrgbdImageDatasetConfig],
    "sunrgbd_anonymous_aligned_image": [SunrgbdAnonymousAlignedImageDetectionDataset, SunrgbdAnonymousAlignedImageDatasetConfig],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi": [SunrgbdAnonymousAlignedImageDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageDatasetConfigWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_object_aug": [SunrgbdAnonymousAlignedImageObjectAugDetectionDataset, SunrgbdAnonymousAlignedImageObjectAugDatasetConfig],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug": [SunrgbdAnonymousAlignedImageObjectAugDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugDatasetConfigWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_virtual_image": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_not_align": [SunrgbdAnonymousAlignedImageObjectAugNotAlignDetectionDatasetWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugNotAlignDatasetConfigWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_load_2d_box": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteReal2dBoxWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteReal2dBoxWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPosWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPosWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPos2dBoxWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPos2dBoxWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_density_adapt": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPos2dBoxDensityAdaptWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPos2dBoxDensityAdaptWithNovelCateConfi],
    # "sunrgbd_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_imsize_prior": [SunrgbdAnonymousAlignedImageObjectAugDetectionVirtualImageDatasetPasteRealCheckPos2dBoxImsizePriorWithNovelCateConfi, SunrgbdAnonymousAlignedImageObjectAugVirtualImageDatasetConfigPasteRealCheckPos2dBoxImsizePriorWithNovelCateConfi],
    "sunrgbd_sample": [SunrgbdSampleDataset, SunrgbdSampleCateConfi],
    "sunrgbd_3d_nod_with_global_novel_pool": [Sunrgbd3DNODGlobalNovelPoolDataLayer, Sunrgbd3DNODGlobalNovelPoolConfig],
    "sunrgbd_3d_nod_with_global_novel_pool_dino_2d_box": [Sunrgbd3DNODGlobalNovelPoolDino2DboxDataLayer, Sunrgbd3DNODGlobalNovelPoolDino2DboxConfig],
    # "sunrgbd_3d_nod_with_global_novel_pool_dino_2d_box_layout_prior": [Sunrgbd3DNODGlobalNovelPoolDino2DboxLayoutPriorDataLayer, Sunrgbd3DNODGlobalNovelPoolDino2DboxLayoutPriorConfig],
    "scannet_3d_nod_with_global_novel_pool_dino_2d_box": [Scannet3DNODGlobalNovelPoolDino2DboxDataLayer, Scannet3DNODGlobalNovelPoolDino2DboxConfig],
    # "sunrgbd_3d_nod_with_global_novel_pool_dino_2d_box_show_im": [Sunrgbd3DNODGlobalNovelPoolDino2DboxShowImDataLayer, Sunrgbd3DNODGlobalNovelPoolDino2DboxShowImConfig],
    # "scannet_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box": [ScannetDetectionAlignedImageAnonymousDatasetVirtual3DImagePasteRealCheckPos2dBoxWithNovelCateConfi, ScannetAnonymousAlignedImageDatasetVirtual3DImagePasteRealCheckPos2dBoxConfigWithNovelCateConfi],
    # "scannet_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_less_data": [ScannetDetectionAlignedImageAnonymousDatasetVirtual3DImagePasteRealCheckPos2dBoxWithNovelCateConfiLessData, ScannetAnonymousAlignedImageDatasetVirtual3DImagePasteRealCheckPos2dBoxConfigWithNovelCateConfiLessData],
    # "scannet_anonymous_aligned_image_with_novel_cate_confi_object_aug_with_virtual_image_paste_real_check_pos_load_2d_box_less_data_reload": [ScannetDetectionAlignedImageAnonymousDatasetVirtual3DImagePasteRealCheckPos2dBoxWithNovelCateConfiLessDataReload, ScannetAnonymousAlignedImageDatasetVirtual3DImagePasteRealCheckPos2dBoxConfigWithNovelCateConfiLessDataReload],
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
            # split_set="printerval",
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
        # "minitest": dataset_builder(
        #     dataset_config,
        #     split_set="minival",
        #     root_dir=args.dataset_root_dir,
        #     use_color=args.use_color,
        #     use_v1=args.if_use_v1,
        #     augment=False,
        #     if_input_image=args.if_input_image
        # ),
        "real_test": real_dataset_builder(
            real_test_config,
            # split_set="novelval",
            split_set="val",
            # split_set="printerval",
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
    
