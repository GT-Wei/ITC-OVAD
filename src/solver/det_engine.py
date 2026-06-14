"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable
import time
import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, logger, novel_classNum=-1):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types, novel_classNum)

    panoptic_evaluator = None
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger):
        samples = samples.to(device)
        # targets = [{k: (v.to(device) if k != "class_texts" and k != "file_name" and k != "class_start_idx" and k != "class_end_idx" and k != "Over_GT_Flag" else v) for k, v in t.items()} for t in targets]

        for t in targets:
            for key, value in t.items():
                if isinstance(value, str) or isinstance(value, list) or isinstance(value, bool) or isinstance(value, int):
                    t[key] = value
                elif value==None:
                    t[key] = value
                else:
                    t[key] = value.to(device)
        
        class_texts_batch = [target['class_texts'] for target in targets]
        
  
        outputs = model(samples, None, class_texts_batch)
            
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize(logger)

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    
    return stats, coco_evaluator



