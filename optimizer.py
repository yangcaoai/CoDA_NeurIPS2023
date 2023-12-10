# Copyright (c) Facebook, Inc. and its affiliates.
import torch


def build_optimizer(args, model):

    params_with_decay = []
    params_without_decay = []
    # print(model.named_parameters())
    for name, param in model.named_parameters():
        # print(name)
        # if name.find('bg_embed')!=-1:
            # print('=============================')
            # print(name)
            # print(name.find('bg_embed'))
            # print(param.requires_grad)
        if param.requires_grad is False:
            continue
        if args.only_prompt_loss and name.find('clip_model') != -1:
            continue
        if args.filter_biases_wd and (len(param.shape) == 1 or name.endswith("bias")):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    if args.filter_biases_wd:
        param_groups = [
            {"params": params_without_decay, "weight_decay": 0.0},
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    else:
        param_groups = [
            {"params": params_with_decay, "weight_decay": args.weight_decay},
        ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.base_lr)
    return optimizer
