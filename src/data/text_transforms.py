# Copyright (c) OpenMMLab. All rights reserved.
import torch
import json
from src.core import register
from typing import Tuple, Any
import torchvision.transforms.v2 as T

try:
    from transformers import AutoTokenizer
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

import random
import re

import numpy as np

# @register
# class RandomClassText:

#     def __call__(self, *inputs: Any) -> Any:
#         outputs = inputs[0]
        
#         class_texts = outputs[1].get('class_texts')
#         gt_labels = outputs[1].get('labels').clone().detach()

#         original_ids = list(range(len(class_texts)))
#         random.shuffle(original_ids)
        
#         # 打乱的class_texts
#         shuffled_class_texts = [class_texts[i] for i in original_ids]
        
#         mapping = {old_id: new_id for new_id, old_id in enumerate(original_ids)}
        
#         updated_gt_labels = [mapping[int(old_id)] for old_id in gt_labels]

#         outputs[1]['class_texts'] = shuffled_class_texts
#         outputs[1]['labels'] = torch.tensor(updated_gt_labels, dtype=torch.int64, device=gt_labels.device)
#         outputs[1]['denosing_orig_labels'] = gt_labels
        
#         return outputs


@register
class RandomLoadText:
    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, *inputs: Any) -> Any:
        results = inputs[0][1]
        
        class_texts = results.get('class_texts')

        # 记录非空元素的索引
        non_empty_indices = [idx for idx, text in enumerate(class_texts) if text[0]]

        # 记录起始和结束索引
        start_idx = non_empty_indices[0]
        end_idx = non_empty_indices[-1]

        # 检查非空元素是否连续
        if non_empty_indices != list(range(start_idx, end_idx + 1)):
            raise ValueError("Non-empty elements are not consecutive.")

        gt_labels = results.get('labels').clone().detach()
        
        num_classes = len(class_texts)

        positive_labels = set(results['labels'].tolist())
        
        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(len(non_empty_indices), self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels and class_texts[idx][0] != "":
                candidate_neg_labels.append(idx)
        num_neg_samples = min(num_neg_samples, len(candidate_neg_labels))
        # num_neg_samples = len(candidate_neg_labels)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['labels']), dtype=bool)
        
        for idx, label in enumerate(results['labels'].tolist()):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results['labels'][idx] = label2ids[label]
        results['boxes'] = results['boxes'][gt_valid_mask]
        results['labels'] = results['labels'][gt_valid_mask]

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['class_texts'] = texts

        inputs[0][1]['boxes'] = results['boxes']
        inputs[0][1]['labels'] = results['labels']
        inputs[0][1]['class_texts'] = results['class_texts']
        inputs[0][1]['denosing_orig_labels'] = gt_labels[gt_valid_mask]
        
        inputs[0][1]['class_start_idx'] = start_idx
        inputs[0][1]['class_end_idx'] = end_idx
        return inputs[0]

@register
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.captions = json.load(f)

    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            inputs = inputs[0]
        image, results = inputs  # results=target

        captions = results.get(
            'class_texts',
            None)

        texts = []
        for idx, cls_caps in enumerate(captions):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            # sel_cls_cap = cls_caps
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['class_texts'] = texts

        return image, results


@register
class RandomLoadText_LAE_1M:
    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 48,
                 padding_to_max: bool = False,
                 denosing_index_range: Tuple[int, int] = (1, 80),
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        self.denosing_index_range = denosing_index_range  # 该数据集denosing是伪标签取值范围
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, *inputs: Any) -> Any:
        results = inputs[0][1]
        
        class_texts = results.get('class_texts')

        # 记录非空元素的索引
        non_empty_indices = [idx for idx, text in enumerate(class_texts) if text[0]]

        # 记录起始和结束索引  初始定义一个区间段，保证数据集间类别不冲突
        start_idx = self.denosing_index_range[0]
        end_idx = self.denosing_index_range[1]

        gt_labels = results.get('labels').clone().detach()
        
        num_classes = len(class_texts)

        positive_labels = set(results['labels'].tolist())
        
        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels and class_texts[idx][0] != "":
                candidate_neg_labels.append(idx)
        
        if self.num_neg_samples == [-1, -1]:
            num_neg_samples = min(
                min(len(non_empty_indices), self.max_num_samples) - len(positive_labels),
                len(candidate_neg_labels))
        else:
            num_neg_samples = min(
                min(len(non_empty_indices), self.max_num_samples) - len(positive_labels),
                random.randint(*self.num_neg_samples))
            num_neg_samples = min(num_neg_samples, len(candidate_neg_labels))
            
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['labels']), dtype=bool)
        
        for idx, label in enumerate(results['labels'].tolist()):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results['labels'][idx] = label2ids[label]
        results['boxes'] = results['boxes'][gt_valid_mask]
        results['labels'] = results['labels'][gt_valid_mask]

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['class_texts'] = texts

        inputs[0][1]['boxes'] = results['boxes']
        inputs[0][1]['labels'] = results['labels']
        inputs[0][1]['class_texts'] = results['class_texts']
        inputs[0][1]['denosing_orig_labels'] = start_idx + gt_labels[gt_valid_mask]
        
        inputs[0][1]['class_start_idx'] = start_idx
        inputs[0][1]['class_end_idx'] = end_idx
        return inputs[0]