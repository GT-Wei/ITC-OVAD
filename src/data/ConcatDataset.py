from torch.utils.data import ConcatDataset as TorchConcatDataset
from src.core import register, create, merge_config, GLOBAL_CONFIG
from src.data import *

@register
class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets_list):
        self.coco = []
        self.sub_dataset_len = []
        dataset = self.build_dataset(datasets_list)
        super().__init__(dataset)
        
    def build_dataset(self, datasets_list):
        dataset = []
        for _, cfg in datasets_list.items():
            _cfg: dict = GLOBAL_CONFIG[cfg['type']]
            _cfg.update(cfg) # update global cls default args 
            name = _cfg.pop('type')
            
            temple_dataset = create(name)
            self.coco.append(temple_dataset.coco)
            self.sub_dataset_len.append(len(temple_dataset))
            dataset.append(temple_dataset)
        return dataset