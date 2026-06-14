# VOC
# datasets_name
#     train/
#         images/
#             img1.jpg
#             ...
#         labelXML/
#             img1.xml
#             ...
#     val/
#         images/
#             img3.jpg
#             ...
#         labelXML/
#             img3.xml

# =0 / 
# difficult=2(COCOdifficult)
import random
import os
import json
import math
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import datapoints
from torchvision.datasets.vision import VisionDataset
from src.core import register
from pycocotools.coco import COCO
from tqdm import tqdm
__all__ = ['ITCOVAD_Detection']

@register
class ITCOVAD_Detection(torch.utils.data.Dataset):
    __inject__ = ['transforms']
    
    def __init__(self, img_folder, ann_file, transforms, class_json_path, coco_ann_json_path, val_dataset_flag=False):
        if not os.path.exists(class_json_path):
            raise FileNotFoundError(f" {class_json_path} ")

        with open(class_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.CLASSES = [item for item in data]
        print(self.CLASSES)
        self.image_dir = img_folder
        self.annotation_dir = ann_file
        self._transforms = transforms
        
        self.val_dataset_flag = val_dataset_flag
        
        self.img_extensions = ['.jpg', '.png', '.tif']  # 

        #  XML 
        self.image_annotation_pairs = self._load_image_annotation_pairs()

        # xml
        self.image_annotation_pairs = [
            pair for pair in self.image_annotation_pairs if os.path.exists(pair[1])
        ]

        if len(self.image_annotation_pairs) != len(self.image_annotation_pairs):
            assert ValueError("Image annotations do not correspond one to one")
        
        self.prepare = Pre_Coco_Data()
        
        # cocojson
        if os.path.exists(coco_ann_json_path):
            self.coco = COCO(coco_ann_json_path)
        else:
            print('None')
            # VOC->COCO200
            # self.coco = self.convert_to_coco_api(max_instances_per_image=300)
            # if not os.path.exists(os.path.dirname(coco_ann_json_path)):
            #     os.makedirs(os.path.dirname(coco_ann_json_path), exist_ok=True)
            # with open(coco_ann_json_path, "w", encoding="utf-8") as f:
            #     json.dump(self.coco.dataset, f, ensure_ascii=False, indent=4)
        # self.coco = self.convert_to_coco_api()
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image_annotation_pairs(self):
        xml_files = [
            os.path.join(self.annotation_dir, fname)
            for fname in os.listdir(self.annotation_dir)
            if fname.lower().endswith('.xml')
        ]
        
        image_annotation_pairs = []
        for xml_path in xml_files:
            base_name = os.path.splitext(os.path.basename(xml_path))[0]
            
            # 
            image_path = None
            for ext in self.img_extensions:
                potential_image = os.path.join(self.image_dir, f"{base_name}{ext}")
                if os.path.exists(potential_image):
                    image_path = potential_image
                    break  # 
            
            if image_path:
                image_annotation_pairs.append((image_path, xml_path))
            else:
                raise ValueError(f":  {xml_path} ")
        return image_annotation_pairs

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.image_dir, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        
        image_info = self.coco.loadImgs(image_id)[0]
        image_name = image_info['file_name']
        # Over_GT_Flag = image_info['Over_GT']
        Over_GT_Flag = False

        image = self._load_image(image_id)
        target = self._load_target(image_id)
        
        if len(target) > 300 and self.val_dataset_flag==False:
            # target = random.sample(target, 300)

            def _ann_area(ann):
                if 'area' in ann and ann['area'] > 0:
                    return ann['area']
                x, y, w, h = ann['bbox']
                return w * h
            
            # 按面积降序排序并保留前 300
            target = sorted(target, key=_ann_area, reverse=True)[:300]

        
        target = {'image_id': image_id, 'annotations': target}
        
        # 
        image, target = self.prepare(image, target)
        target['file_name'] = image_name
        target['Over_GT_Flag'] = Over_GT_Flag
        # 
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=image.size[::-1]  # 
            )

        target['class_texts'] = self.CLASSES

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target

    def convert_to_coco_api(self, max_instances_per_image=300):
        # VOCCOCOmax_instances_per_image
        coco_ds = COCO()
        # annotation IDs need to start at 1, not 0, see torchvision issue #1530
        ann_id = 1
        dataset = {"images": [], "categories": [], "annotations": []}

        #  for image_id in range(...): image_id ID
        # VOCCOCO
        new_image_id = 0

        # COCOcategories
        dataset["categories"] = [
            {"id": idx, "name": cls_name, "supercategory": "none"}
            for idx, cls_name in enumerate(self.CLASSES)
        ]

        for i, (img_path, ann_path) in enumerate(tqdm(self.image_annotation_pairs, desc="conver data to coco api")):
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            # XML
            tree = ET.parse(ann_path)
            root = tree.getroot()

            # 
            objects = root.findall('object')
            image_annotations = []

            for obj in objects:
                name = obj.find('name').text.lower().strip()
                bndbox = obj.find('bndbox')

                xmin = int(float(bndbox.find('xmin').text)) - 1
                ymin = int(float(bndbox.find('ymin').text)) - 1
                xmax = int(float(bndbox.find('xmax').text)) + 1
                ymax = int(float(bndbox.find('ymax').text)) + 1
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin
                bbox = [xmin, ymin, bbox_w, bbox_h]

                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 2:
                    continue

                if name in self.CLASSES:
                    category_id = self.CLASSES.index(name)
                else:
                    # self.CLASSES
                    if not self.val_dataset_flag:
                        raise ValueError(f"{name} not in {self.CLASSES}")
                    continue

                ann = {
                    'image_id': None,  # 
                    'bbox': bbox,
                    'category_id': category_id,
                    'area': bbox_w * bbox_h,
                    'iscrowd': 0,
                    'id': None  # 
                }
                image_annotations.append(ann)

            #  max_instances_per_image 
            total_anns = len(image_annotations)
            
            if not self.val_dataset_flag:  # 训练
                if total_anns == 0:
                    # image
                    img_dict = {
                        "id": new_image_id,
                        "file_name": os.path.basename(img_path),
                        "height": orig_h,
                        "width": orig_w,
                        "Over_GT": False
                    }
                    dataset["images"].append(img_dict)
                    dataset["annotations"].extend(image_annotations)
                    new_image_id += 1
                else:
                    # 
                    img_dict = {
                        "id": new_image_id,
                        "file_name": os.path.basename(img_path),
                        "height": orig_h,
                        "width": orig_w,
                        "Over_GT": True
                    }
                    dataset["images"].append(img_dict)

                    for ann in image_annotations:
                        ann["image_id"] = new_image_id
                        ann["id"] = ann_id
                        ann_id += 1

                    dataset["annotations"].extend(image_annotations)
                    new_image_id += 1
                # elif total_anns <= max_instances_per_image:
                #     # 
                #     img_dict = {
                #         "id": new_image_id,
                #         "file_name": os.path.basename(img_path),
                #         "height": orig_h,
                #         "width": orig_w,
                #         "Over_GT": False
                #     }
                #     dataset["images"].append(img_dict)

                #     for ann in image_annotations:
                #         ann["image_id"] = new_image_id
                #         ann["id"] = ann_id
                #         ann_id += 1

                #     dataset["annotations"].extend(image_annotations)
                #     new_image_id += 1
                # else:
                #     chunk_size = max_instances_per_image
                #     for idx in range(0, total_anns, chunk_size):
                #         chunk_anns = image_annotations[idx: idx + chunk_size]

                #         # base_name = os.path.basename(img_path)
                #         # root_name, ext = os.path.splitext(base_name)
                #         # new_file_name = f"{root_name}_{idx}{ext}"

                #         img_dict = {
                #             "id": new_image_id,
                #             "file_name": os.path.basename(img_path),
                #             "height": orig_h,
                #             "width": orig_w,
                #             "Over_GT": True
                #         }
                #         dataset["images"].append(img_dict)

                #         for ann in chunk_anns:
                #             ann["image_id"] = new_image_id
                #             ann["id"] = ann_id
                #             ann_id += 1

                #         dataset["annotations"].extend(chunk_anns)
                #         new_image_id += 1
            else:
                if len(image_annotations) > 0:
                    img_dict = {
                        "id": new_image_id,
                        "file_name": os.path.basename(img_path),
                        "height": orig_h,
                        "width": orig_w,
                        "Over_GT": False
                    }
                    dataset["images"].append(img_dict)

                    for ann in image_annotations:
                        ann["image_id"] = new_image_id
                        ann["id"] = ann_id
                        ann_id += 1

                    dataset["annotations"].extend(image_annotations)
                    new_image_id += 1


        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds

    def extra_repr(self) -> str:
        s = f'Image folder: {self.image_dir}\nAnnotation folder: {self.annotation_dir}\n'
        if self._transforms is not None:
            s += f'Transforms: {repr(self._transforms)}'
        return s


class Pre_Coco_Data(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])

        return image, target