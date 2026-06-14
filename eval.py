"""
ITC-OVAD — reproduce the zero-shot / open-vocabulary detection metrics.

Each validation set in the config is evaluated separately and the per-split
AP50 (Base / Novel / HM) and Recall are printed (same protocol as the paper).

Checkpoint pairing:
    DIOR & DOTA splits -> weights/itc_ovad_dior_dota.pth
    xView split        -> weights/itc_ovad_xview.pth

Example:
    python eval.py -c configs/itc-ovad/eval.yml -r weights/itc_ovad_dior_dota.pth
"""
import os
import sys
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

import src.misc.dist  # noqa: F401
from src.core import YAMLConfig
from src.data import ConcatDataset, get_coco_api_from_dataset
from src.solver import evaluate


def get_logger():
    logger = logging.getLogger("itc-ovad-eval")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
    return logger


def load_weights(path):
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        return ckpt["model"]
    if "ema" in ckpt:
        return ckpt["ema"]["module"]
    return ckpt


def main(args):
    logger = get_logger()
    cfg = YAMLConfig(args.config)
    cfg.model.load_state_dict(load_weights(args.resume))

    module = cfg.model.to(args.device).eval()          # EMA weights, eval mode (no reparam)
    postprocessor = cfg.postprocessor
    criterion = nn.Module()                            # evaluate() only calls .eval() on it
    val_loader = cfg.val_dataloader

    # ConcatDataset -> evaluate each sub-dataset independently
    base_ds_list, base_ds_lens = get_coco_api_from_dataset(val_loader.dataset)
    novel_list = val_loader.novel_classNum_list

    start = 0
    for i, (ds_coco, ds_len, novelN) in enumerate(zip(base_ds_list, base_ds_lens, novel_list)):
        end = start + ds_len
        sub_loader = DataLoader(
            Subset(val_loader.dataset, range(start, end)),
            batch_size=val_loader.batch_size, shuffle=False,
            num_workers=val_loader.num_workers, collate_fn=val_loader.collate_fn,
        )
        logger.info(f"===== evaluating sub-dataset {i} ({ds_len} images) =====")
        evaluate(module, criterion, postprocessor, sub_loader, ds_coco,
                 args.device, None, logger, novelN)
        start = end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITC-OVAD evaluation")
    parser.add_argument("-c", "--config", default="configs/itc-ovad/eval.yml")
    parser.add_argument("-r", "--resume", required=True, help="ITC-OVAD checkpoint")
    parser.add_argument("-d", "--device", default="cuda")
    main(parser.parse_args())
