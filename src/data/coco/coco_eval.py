# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.
增加:
  1. FP 混淆来源分析 (analyze_fp_confusion) - prediction-centric
  2. GT 归属分析 (analyze_gt_assignment) - GT-centric
  无条件支持 GZSD 和 ZSD
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from src.misc import dist


__all__ = ['CocoEvaluator',]


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, novel_class_num=4):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        
        self.novel_class = novel_class_num

        # ===== 累积所有 bbox 预测结果 =====
        self.all_bbox_results = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            if iou_type == "bbox":
                self.all_bbox_results.extend(results)

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

        all_results = dist.all_gather(self.all_bbox_results)
        merged = []
        for r in all_results:
            merged.extend(r)
        seen = set()
        deduped = []
        for r in merged:
            key = (r["image_id"], tuple(r["bbox"]), r["score"], r["category_id"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        self.all_bbox_results = deduped

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self, logger):
        for iou_type, coco_eval in self.coco_eval.items():

            # ---------- 官方 12 项指标 ----------
            logger.info(f"IoU metric: {iou_type}")
            coco_eval.summarize()

            # ---------- 自定义统计 ----------
            if not getattr(coco_eval, "eval", None):
                logger.info(f"[Warning] {iou_type} does not have valid eval results.")
                continue

            precision = coco_eval.eval["precision"]
            recall    = coco_eval.eval["recall"]
            if precision is None or recall is None:
                logger.info(f"[Warning] {iou_type} missing precision or recall.")
                continue

            iou_thrs    = coco_eval.params.iouThrs
            if 0.5 not in iou_thrs:
                logger.info("[Warning] 0.5 not in iouThrs, skip IoU=0.5 stats.")
                continue
            iou50_idx   = int(np.where(iou_thrs == 0.5)[0][0])

            area_labels = coco_eval.params.areaRngLbl
            area_idx    = {lbl: i for i, lbl in enumerate(area_labels)}
            max_det_idx = len(coco_eval.params.maxDets) - 1

            K        = precision.shape[2]
            cat_ids  = coco_eval.params.catIds
            cat_info = []

            for k in range(K):
                cls_prec = precision[iou50_idx, :, k, area_idx["all"], max_det_idx]
                if (cls_prec > -1).any():
                    ap50 = float(np.mean(cls_prec[cls_prec > -1]))
                else:
                    ap50 = float("nan")

                cls_rec = recall[iou50_idx, k, area_idx["all"], max_det_idx]
                rec50   = float(cls_rec) if cls_rec > -1 else float("nan")

                cat_info.append(
                    {
                        "cat_id":   cat_ids[k],
                        "cat_name": coco_eval.cocoGt.cats[cat_ids[k]]["name"],
                        "ap50":     ap50,
                        "recall50": rec50,
                    }
                )

            # ---------- GZSD ----------
            if self.novel_class not in (-1, 0) and self.novel_class < K:
                base_cats   = cat_info[:-self.novel_class]
                novel_cats  = cat_info[-self.novel_class:]

                def _avg(cats, key):
                    vals = [c[key] for c in cats if not np.isnan(c[key])]
                    return float(np.mean(vals)) if vals else float("nan")

                base_ap   = _avg(base_cats,  "ap50")
                novel_ap  = _avg(novel_cats, "ap50")
                base_rec  = _avg(base_cats,  "recall50")
                novel_rec = _avg(novel_cats, "recall50")

                def _hm(a, b):
                    return 2 * a * b / (a + b) if not (np.isnan(a) or np.isnan(b) or (a + b) == 0) else np.nan

                hm_ap  = _hm(base_ap,  novel_ap)
                hm_rec = _hm(base_rec, novel_rec)

                logger.info(
                    f"[GZSD] Base ({K-self.novel_class}) vs Novel ({self.novel_class}) "
                    f"@ IoU=0.50 (area=all, maxDet=100)"
                )
                logger.info(
                    f"  AP50   – Base: {base_ap:.4f} | Novel: {novel_ap:.4f} | HM: {hm_ap:.4f}"
                )
                logger.info(
                    f"  Recall – Base: {base_rec:.4f} | Novel: {novel_rec:.4f} | HM: {hm_rec:.4f}"
                )

            # ---------- per-class 表格 ----------
            logger.info("## Per-category AP50 & Recall50 (area=all, IoU=0.5)")
            logger.info("| cat_id | cat_name             |  AP50 | Recall50 |")
            logger.info("|-------:|-----------------------|------:|---------:|")

            ap_vals = []
            rec_vals = []

            for c in cat_info:
                ap = c['ap50']
                rec = c['recall50']
                ap_str  = f"{ap:.4f}"  if not np.isnan(ap)  else "nan"
                rec_str = f"{rec:.4f}" if not np.isnan(rec) else "nan"

                logger.info(
                    f"| {c['cat_id']:>6} | {c['cat_name']:<21} | {ap_str:>5} | {rec_str:>8} |"
                )

                if not np.isnan(ap): ap_vals.append(ap)
                if not np.isnan(rec): rec_vals.append(rec)

            mean_ap  = np.mean(ap_vals) if ap_vals else float('nan')
            mean_rec = np.mean(rec_vals) if rec_vals else float('nan')
            mean_ap_str  = f"{mean_ap:.4f}" if not np.isnan(mean_ap) else "nan"
            mean_rec_str = f"{mean_rec:.4f}" if not np.isnan(mean_rec) else "nan"

            logger.info("|--------|-----------------------|-------|----------|")
            logger.info(f"|  Mean  | {'(all categories)':<21} | {mean_ap_str:>5} | {mean_rec_str:>8} |")
            logger.info("|--------|-----------------------|-------|----------|")

            # =============================================
            # ===== FP 分析 + GT 归属分析 =====
            # ===== 无条件触发: 只要是 bbox 且有预测结果 =====
            # =============================================
            if iou_type == "bbox" and self.all_bbox_results:
                # 确定 novel / base
                if self.novel_class > 0 and self.novel_class < K:
                    # GZSD: 有 base 和 novel
                    novel_cat_ids = set(cat_ids[-self.novel_class:])
                    base_cat_ids  = set(cat_ids[:-self.novel_class])
                    target_cat_ids = novel_cat_ids
                else:
                    # ZSD 或全类别: 所有类都当 target 分析
                    target_cat_ids = set(cat_ids)
                    novel_cat_ids  = set(cat_ids)
                    base_cat_ids   = set()

                self.analyze_fp_confusion(coco_eval, logger,
                                          target_cat_ids=target_cat_ids,
                                          other_cat_ids=base_cat_ids,
                                          iou_thresh=0.5, score_thresh=0.3)
                self.analyze_gt_assignment(coco_eval, logger,
                                           target_cat_ids=target_cat_ids,
                                           iou_thresh=0.5, score_thresh=0.1)

                # ===== 清空累积结果，防止跨 subdataset 污染 =====
                self.all_bbox_results = []

    def analyze_fp_confusion(self, coco_eval, logger, target_cat_ids, other_cat_ids,
                              iou_thresh=0.5, score_thresh=0.3):
        """
        Prediction-centric 分析: 每个 target class 的预测框，其 FP 来自哪里？
        """
        from collections import defaultdict

        coco_gt = self.coco_gt
        cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

        gt_by_img = defaultdict(list)
        for ann in coco_gt.dataset["annotations"]:
            if not ann.get("iscrowd", 0):
                gt_by_img[ann["image_id"]].append(ann)

        dt_by_img = defaultdict(list)
        for ann in self.all_bbox_results:
            if ann["score"] >= score_thresh:
                dt_by_img[ann["image_id"]].append(ann)

        all_img_ids = set(list(gt_by_img.keys()) + list(dt_by_img.keys()))

        has_other = len(other_cat_ids) > 0
        setting_name = "GZSD" if has_other else "ZSD"

        logger.info("")
        logger.info("=" * 65)
        logger.info(f"  FP SOURCE ANALYSIS ({setting_name})")
        logger.info(f"  IoU={iou_thresh}, score>={score_thresh}")
        logger.info("=" * 65)

        for cid in sorted(target_cat_ids):
            cname = cat_id_to_name.get(cid, f"class_{cid}")

            stats = {
                "TP": 0,
                "FP_bg": 0,
                "FP_loc": 0,
                "FP_dup": 0,
                "FP_other": defaultdict(int),   # base-class 混淆 (GZSD) 
                "FP_intra": defaultdict(int),    # target 类间混淆
                "total_preds": 0,
                "total_gt": 0,
            }

            for img_id in all_img_ids:
                gts  = gt_by_img[img_id]
                dets = [d for d in dt_by_img[img_id] if d["category_id"] == cid]

                this_gts = [g for g in gts if g["category_id"] == cid]
                stats["total_gt"] += len(this_gts)

                if not dets:
                    continue

                stats["total_preds"] += len(dets)
                dets = sorted(dets, key=lambda x: x["score"], reverse=True)
                matched_gt_ids = set()

                for det in dets:
                    best_iou, best_gt = 0.0, None
                    for gt in gts:
                        iou = self._iou(det["bbox"], gt["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt

                    if best_iou >= iou_thresh and best_gt is not None:
                        gt_cid = best_gt["category_id"]
                        gt_id  = best_gt["id"]

                        if gt_cid == cid:
                            if gt_id not in matched_gt_ids:
                                stats["TP"] += 1
                                matched_gt_ids.add(gt_id)
                            else:
                                stats["FP_dup"] += 1
                        elif gt_cid in other_cat_ids:
                            stats["FP_other"][cat_id_to_name.get(gt_cid, f"class_{gt_cid}")] += 1
                        elif gt_cid in target_cat_ids:
                            stats["FP_intra"][cat_id_to_name.get(gt_cid, f"class_{gt_cid}")] += 1
                        else:
                            stats["FP_bg"] += 1

                    elif best_iou > 0.1 and best_gt is not None and best_gt["category_id"] == cid:
                        stats["FP_loc"] += 1
                    else:
                        stats["FP_bg"] += 1

            fp_other_total = sum(stats["FP_other"].values())
            fp_intra_total = sum(stats["FP_intra"].values())
            total_fp = stats["FP_bg"] + stats["FP_loc"] + stats["FP_dup"] + fp_other_total + fp_intra_total

            logger.info(f"\n--- {cname} ---")
            logger.info(f"  GT: {stats['total_gt']}  |  Preds: {stats['total_preds']}  |  TP: {stats['TP']}  |  FP: {total_fp}")

            if total_fp > 0:
                logger.info(f"  FP breakdown:")
                logger.info(f"    Background:   {stats['FP_bg']:>5}  ({100*stats['FP_bg']/total_fp:.1f}%)")
                logger.info(f"    Localization:  {stats['FP_loc']:>5}  ({100*stats['FP_loc']/total_fp:.1f}%)")
                logger.info(f"    Duplicate:     {stats['FP_dup']:>5}  ({100*stats['FP_dup']/total_fp:.1f}%)")
                if has_other and fp_other_total > 0:
                    logger.info(f"    Base-class:    {fp_other_total:>5}  ({100*fp_other_total/total_fp:.1f}%)")
                    for cls, cnt in sorted(stats["FP_other"].items(), key=lambda x: -x[1])[:10]:
                        logger.info(f"      -> {cls}: {cnt}")
                if fp_intra_total > 0:
                    label = "Novel-class" if has_other else "Inter-class"
                    logger.info(f"    {label}:   {fp_intra_total:>5}  ({100*fp_intra_total/total_fp:.1f}%)")
                    for cls, cnt in sorted(stats["FP_intra"].items(), key=lambda x: -x[1])[:10]:
                        logger.info(f"      -> {cls}: {cnt}")

            total = stats["TP"] + total_fp
            if total > 0:
                logger.info(f"  Precision: {stats['TP']/total:.4f}")
            if stats["total_gt"] > 0:
                logger.info(f"  Recall:    {stats['TP']/stats['total_gt']:.4f}")

        logger.info("=" * 65)

    def analyze_gt_assignment(self, coco_eval, logger, target_cat_ids, iou_thresh=0.5, score_thresh=0.1):
        """
        GT-centric 分析: 每个 target class 的 GT，被模型预测为了哪个类别？
        """
        from collections import defaultdict

        coco_gt = self.coco_gt
        cat_id_to_name = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

        gt_by_img = defaultdict(list)
        for ann in coco_gt.dataset["annotations"]:
            if not ann.get("iscrowd", 0):
                gt_by_img[ann["image_id"]].append(ann)

        dt_by_img = defaultdict(list)
        for ann in self.all_bbox_results:
            if ann["score"] >= score_thresh:
                dt_by_img[ann["image_id"]].append(ann)

        logger.info("")
        logger.info("=" * 65)
        logger.info(f"  GT ASSIGNMENT ANALYSIS")
        logger.info(f"  IoU={iou_thresh}, score>={score_thresh}")
        logger.info("=" * 65)

        for cid in sorted(target_cat_ids):
            cname = cat_id_to_name.get(cid, f"class_{cid}")

            stats = {
                "total_gt": 0,
                "correct": 0,
                "misclassified": defaultdict(int),
                "missed": 0,
                "low_iou_detected": 0,
            }

            for img_id, gts in gt_by_img.items():
                this_gts = [g for g in gts if g["category_id"] == cid]
                if not this_gts:
                    continue

                all_dets = dt_by_img[img_id]
                stats["total_gt"] += len(this_gts)

                for gt in this_gts:
                    best_iou, best_det = 0.0, None

                    for det in all_dets:
                        iou = self._iou(gt["bbox"], det["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_det = det

                    if best_iou >= iou_thresh and best_det is not None:
                        pred_cid = best_det["category_id"]
                        if pred_cid == cid:
                            stats["correct"] += 1
                        else:
                            pred_name = cat_id_to_name.get(pred_cid, f"unknown_{pred_cid}")
                            stats["misclassified"][pred_name] += 1
                    elif best_iou > 0.1:
                        stats["low_iou_detected"] += 1
                    else:
                        stats["missed"] += 1

            total_gt = stats["total_gt"]
            misclass_total = sum(stats["misclassified"].values())

            logger.info(f"\n--- {cname} (GT-centric) ---")
            logger.info(f"  Total GT: {total_gt}")

            if total_gt > 0:
                logger.info(f"  Correctly classified:  {stats['correct']:>5}  ({100*stats['correct']/total_gt:.1f}%)")
                logger.info(f"  Misclassified:         {misclass_total:>5}  ({100*misclass_total/total_gt:.1f}%)")
                if stats["misclassified"]:
                    for cls, cnt in sorted(stats["misclassified"].items(), key=lambda x: -x[1])[:15]:
                        logger.info(f"    -> predicted as {cls}: {cnt}  ({100*cnt/total_gt:.1f}%)")
                logger.info(f"  Low-IoU overlap:       {stats['low_iou_detected']:>5}  ({100*stats['low_iou_detected']/total_gt:.1f}%)")
                logger.info(f"  Completely missed:     {stats['missed']:>5}  ({100*stats['missed']/total_gt:.1f}%)")

        logger.info("=" * 65)

    @staticmethod
    def _iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
        inter = max(0, xi2-xi1) * max(0, yi2-yi1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0.0

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = dist.all_gather(img_ids)
    all_eval_imgs = dist.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(self):
    p = self.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    return p.imgIds, evalImgs