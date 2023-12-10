# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import os
from utils.dist import is_primary


def save_checkpoint(
    checkpoint_dir,
    model_no_ddp,
    optimizer,
    epoch,
    args,
    best_val_metrics,
    filename=None,
):
    if not is_primary():
        return
    if filename is None:
        filename = f"checkpoint_{epoch:04d}.pth"
    checkpoint_name = os.path.join(checkpoint_dir, filename)

    sd = {
        "model": model_no_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(sd, checkpoint_name)


def resume_if_possible(checkpoint_dir, model_no_ddp, optimizer, checkpoint_file):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = -1
    best_val_metrics = {}
    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics

    if checkpoint_file is not None:
        last_checkpoint = checkpoint_file
        if not os.path.isfile(last_checkpoint):
            return epoch, best_val_metrics
    else:
        last_checkpoint = os.path.join(checkpoint_dir, "checkpoint.pth")
        if not os.path.isfile(last_checkpoint):
            return epoch, best_val_metrics

    sd = torch.load(last_checkpoint, map_location=torch.device("cpu"))
    epoch = sd["epoch"]
    best_val_metrics = sd["best_val_metrics"]
    print(f"Found checkpoint at {epoch}. Resuming.")
    # optimizer.load_state_dict(sd["optimizer"])
    # model_no_ddp.load_state_dict(sd["model"], strict=True)

    try:
        model_no_ddp.load_state_dict(sd["model"], strict=True)
    except:
        print('load by strict=False')
        model_no_ddp.load_state_dict(sd["model"], strict=False)
    try:
        optimizer.load_state_dict(sd["optimizer"])
    except:
        print('can not load the optimizer')
    print(
        f"Loaded model and optimizer state at {epoch}. Loaded best val metrics so far."
    )
    return epoch, best_val_metrics
