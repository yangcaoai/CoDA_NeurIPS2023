# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr_predictedbox_distillation_head, build_3detr_multiclasshead, build_3detr_predictedbox_distillation_globalpool_2dbox_head
MODEL_FUNCS = {
    "3detrmulticlasshead": build_3detr_multiclasshead, #####
    "3detr_predictedbox_distillation": build_3detr_predictedbox_distillation_head, ###
    "3detr_predictedbox_distillation_head_globalpool_2dbox": build_3detr_predictedbox_distillation_globalpool_2dbox_head
}

def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor