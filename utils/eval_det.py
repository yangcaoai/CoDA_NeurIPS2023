# Copyright (c) Facebook, Inc. and its affiliates.

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box
    
    Output:
    For each class:
        precision-recal and average precision
    
    Author: Charles R. Qi
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
import numpy as np
from utils.box_util import box3d_iou


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_iou_obb(bb1, bb2):
    iou3d, iou2d = box3d_iou(bb1, bb2)
    return iou3d


def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)


def eval_det_cls(
    pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou_obb
):
    """Generic functions to compute precision/recall for object detection
    for a single class.
    Input:
        pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
        gt: map of {img_id: [bbox]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if True use VOC07 11 point method
    Output:
        rec: numpy array of length nd
        prec: numpy array of length nd
        ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id]) # the gt_bbox of image_id
        det = [False] * len(bbox) # False of the same length with gt_bbox of image_id
        npos += len(bbox) # count the gt boxes of all the image
        class_recs[img_id] = {"bbox": bbox, "det": det} 
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box, score in pred[img_id]:
            image_ids.append(img_id) # image ID list
            confidence.append(score) # the max probability of all the class
            BB.append(box) # bounding box list
    confidence = np.array(confidence)
    BB = np.array(BB)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence) # sort the value from biger to smaller ones and return the idx
    sorted_scores = np.sort(-confidence) # sort the max probability
    # top_k = confidence.shape[0]//6
    # sorted_ind = np.argsort(-confidence)[:top_k] # sort the value from biger to smaller ones and return the idx
    # sorted_scores = np.sort(-confidence)[:top_k] # sort the max probability
    BB = BB[sorted_ind, ...] # sort the bbox according to the max probability
    image_ids = [image_ids[x] for x in sorted_ind] # sort the image id according the max probability

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # the number of bbox
    tp = np.zeros(nd) # true  positive
    fp = np.zeros(nd) # false positive
    for d in range(nd):
        # if d%100==0: print(d)
        R = class_recs[image_ids[d]] # the GT info of image_id
        bb = BB[d, ...].astype(float) # the predicted bbox of the specific box
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float) # all the bbox gt of image_id

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                try:
                    iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                except:
                    print('The wrong image is:')
                    print(image_ids[d])
                if iou > ovmax: # find the bbox gt with the bigest IOU with the predicted bbox
                    ovmax = iou
                    jmax = j

        # print d, ovmax
        if ovmax > ovthresh:
            if not R["det"][jmax]: # if find a gt box, which has a IOU bigger than ovthresh AND the gt_box is not used for other predicted box, tp=1, else fp=1
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp) # qian xu sum from the bbox with the biggest score to the smallest score
    tp = np.cumsum(tp)
    if npos == 0:
        rec = np.zeros_like(tp)
    else:
        rec = tp / float(npos) # recall 
    # print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) # precision
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_det_cls_more(pred, gt, ovthresh=0.25, score_thresh=0.05, use_07_metric=False, get_iou_func=get_iou_obb):
    """
    评估单一类别的 AP、FPR、FNR 和 F1 Score
    """
    # 构建 Ground Truth 信息
    class_recs = {}
    npos = 0  # GT 框的总数
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])  # 当前图片的 GT 框
        det = [False] * len(bbox)  # 初始化 GT 框匹配状态
        npos += len(bbox)
        class_recs[img_id] = {"bbox": bbox, "det": det}

    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {"bbox": np.array([]), "det": []}

    # 构建预测框
    image_ids = []
    confidence = []
    BB = []

    for img_id in pred.keys():
        for bbox, score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(bbox)
    confidence = np.array(confidence)
    BB = np.array(BB)

    # 按置信度排序
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[i] for i in sorted_ind]

    # 逐个预测框进行匹配
    nd = len(image_ids)
    tp = np.zeros(nd)  # True Positives
    fp = np.zeros(nd)  # False Positives
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, ...].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # 计算 IoU
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j, ...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        if ovmax > ovthresh:
            if not R["det"][jmax]:  # 如果 GT 框未被匹配
                tp[d] = 1.0
                R["det"][jmax] = 1  # 标记 GT 框已匹配
            else:
                fp[d] = 1.0  # 重复匹配的预测框为 FP
        else:
            fp[d] = 1.0  # 没有匹配到任何 GT 框的预测框为 FP

    # 计算 Precision 和 Recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    if len(confidence) == 0 or len(BB) == 0:
        return 0, 0, 0, 0.0, 1.0, 0.0
    # rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    # 计算 AP
    # ap = voc_ap(rec, prec, use_07_metric)

    # 计算 FPR、FNR 和 F1 Score（固定阈值）
    filtered_tp = tp[-1]  # 最终 TP 数

    filtered_fp = fp[-1]  # 最终 FP 数
    filtered_fn = npos - filtered_tp  # 最终 FN 数
    # fpr = filtered_fp / float(filtered_fp + filtered_tp + np.finfo(np.float64).eps)
    # fnr = filtered_fn / float(filtered_tp + filtered_fn + np.finfo(np.float64).eps)
    final_precision = filtered_tp / float(filtered_tp + filtered_fp + np.finfo(np.float64).eps)
    final_recall = filtered_tp / float(npos)
    f1 = (2 * final_precision * final_recall) / (final_precision + final_recall + np.finfo(np.float64).eps)

    return 0, 0, 0, filtered_fp, filtered_fn, f1


def eval_det_cls_wrapper(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)


def eval_det(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou_obb):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
        #del pred_all[img_id]
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)
        #del gt_all[img_id]

    rec = {}
    prec = {}
    ap = {}
    gt_keys = list(gt.keys())
    for classname in gt_keys:
    # for classname in gt.keys():
        # print('Computing AP for class: ', classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(
            pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func
        )
        del gt[classname]
        del pred[classname]
        # print(classname, ap[classname])

    return rec, prec, ap


def eval_det_more(pred_all, gt_all, ovthresh=0.25, use_07_metric=False, score_thresh=0.25, get_iou_func=get_iou_obb):
    """
    多类别目标检测评估，确保每个预测框只分配到一个类别
    """
    pred = {}
    gt = {}

    for img_id in pred_all.keys():
        # 临时存储每个框的最佳类别
        max_class_per_bbox = {}

        for classname, bbox, score in pred_all[img_id]:
            if score < score_thresh:  # 过滤置信度低于阈值的框
                continue

            # 确保 bbox 是一维列表或数组
            if isinstance(bbox, np.ndarray):
                bbox_flated = bbox.flatten()  # 将 NumPy 数组展平为一维
            elif isinstance(bbox, list):
                bbox_flated = [float(coord) for coord in bbox]  # 确保列表中的值为浮点数
            else:
                print(f"Invalid bbox format: {bbox}, skipping...")
                continue

            # 转换 bbox 为唯一键
            try:
                bbox_key = tuple(map(float, bbox_flated))  # 用作键
            except TypeError as e:
                print(f"Error processing bbox {bbox}: {e}")
                continue

            # 如果当前框的概率更高，则更新其所属类别
            if bbox_key not in max_class_per_bbox or max_class_per_bbox[bbox_key]["score"] < score:
                max_class_per_bbox[bbox_key] = {"classname": classname, "bbox": bbox, "score": score}

        # 将每个框分配到其最大概率类别
        for bbox_key, data in max_class_per_bbox.items():
            classname = data["classname"]
            bbox = data["bbox"]
            score = data["score"]

            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))

    # 2. 构建 GT 框
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []

    # 3. 逐类别评估
    rec = {}
    prec = {}
    ap = {}
    fpr = {}
    fnr = {}
    f1 = {}
    for classname in gt.keys():
        rec[classname], prec[classname], ap[classname], fpr[classname], fnr[classname], f1[classname] = eval_det_cls_more(
            pred[classname], gt[classname], ovthresh, score_thresh, use_07_metric,  get_iou_func
        )

    return rec, prec, ap, fpr, fnr, f1


from multiprocessing import Pool


def eval_det_multiprocessing(
    pred_all, gt_all, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou_obb
):
    """Generic functions to compute precision/recall for object detection
    for multiple classes.
    Input:
        pred_all: map of {img_id: [(classname, bbox, score)]}
        gt_all: map of {img_id: [(classname, bbox)]}
        ovthresh: scalar, iou threshold
        use_07_metric: bool, if true use VOC07 11 point method
    Output:
        rec: {classname: rec}
        prec: {classname: prec_all}
        ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    pred_all_key = list(pred_all.keys())
    for img_id in pred_all_key:
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred:
                pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
        del pred_all[img_id]
    gt_all_key = list(gt_all.keys())
    for img_id in gt_all_key:
        for classname, bbox in gt_all[img_id]:
            if classname not in gt:
                gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)
        del gt_all[img_id]

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(
        eval_det_cls_wrapper,
        [
            (pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func)
            for classname in gt.keys()
            if classname in pred
        ],
    )
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        # print(classname, ap[classname])

    return rec, prec, ap
