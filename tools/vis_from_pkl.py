import os
import torch
import numpy as np
import mmengine
import mmcv
import argparse
from pycocotools.coco import COCO
from mmengine.fileio import get
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmdet.datasets import CocoDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--pkl_file', type=str, default='test.pkl')
    parser.add_argument(
        '--img_root', type=str, default='data/coco/val/tir')
    parser.add_argument(
        '--ann_file', type=str, default='data/coco/annotations/instances_val2017.json')
    parser.add_argument(
        '--wait_time', type=int, default=2)
    parser.add_argument(
        '--pred_score_thr', type=float, default=0.1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # Load pred_results
    pred_results = mmengine.load(args.pkl_file)
    # Load Visualizer
    visuliazer = DetLocalVisualizer()
    # Load coco
    coco = COCO(args.ann_file)
    # Vis
    for pred in pred_results:
        # process img_info
        img_id = pred['img_id']
        img_info = coco.loadImgs(img_id)
        img_file_name = img_info[0]['file_name']
        img_tir_file_path = os.path.join(args.img_root, img_file_name)

        img_bytes = get(img_tir_file_path, backend_args=None)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        # load gt
        annids = coco.getAnnIds(imgIds=img_info[0]['id'])
        anns = coco.loadAnns(annids)
        gt_bboxes = []
        gt_labels = []
        for ann in anns:
            # change bbox format from x,y,w,h -> x1,y1,x2,y2
            x, y, w, h = ann['bbox']
            x2, y2 = x+w, y+h
            gt_bboxes.append([x, y, x2, y2])
            gt_labels.append(ann['category_id'])
        gt_bboxes = torch.from_numpy(np.array(gt_bboxes))
        gt_labels = torch.from_numpy(np.array(gt_labels))
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels

        # create data sample
        data_sample = DetDataSample()
        data_sample.batch_input_shape = pred['img_shape']
        data_sample.img_id = img_id
        data_sample.img_path = img_tir_file_path
        pred_instances = InstanceData()
        pred_instances.bboxes = pred['pred_instances']['bboxes']
        pred_instances.labels = pred['pred_instances']['labels']
        pred_instances.scores = pred['pred_instances']['scores']
        data_sample.pred_instances = pred_instances
        data_sample.ori_shape = pred['ori_shape']
        data_sample.pad_shape = pred['pad_shape']
        data_sample.gt_instances = gt_instances

        visuliazer.add_datasample(
            str(img_id),
            image=img,
            data_sample=data_sample,
            show=True,
            draw_gt=True,
            wait_time=args.wait_time,
            pred_score_thr=args.pred_score_thr
        )





