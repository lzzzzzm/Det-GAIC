import os
import mmengine
import argparse
from mmdet.evaluation.metrics import CocoMetric
from mmdet.datasets import CocoDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--pkl_file', type=str, default='test.pkl')
    parser.add_argument(
        '--ann_file', type=str, default='data/coco/annotations/instances_val2017.json')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # Load evaluator
    evaluator = CocoMetric(
        ann_file=args.ann_file,
        dataset_meta=CocoDataset.METAINFO,
        proposal_nums=(10, 100, 500)
    )
    # Load pred_results
    pred_results = mmengine.load(args.pkl_file)
    zip_results = []
    index = 0
    for res in pred_results:
        gt = dict()
        pred = dict()
        gt['height'] = res['ori_shape'][0]
        gt['width'] = res['ori_shape'][1]
        gt['img_id'] = res['img_id']
        pred['bboxes'] = res['pred_instances']['bboxes']
        pred['scores'] = res['pred_instances']['scores']
        pred['labels'] = res['pred_instances']['labels']
        pred['img_id'] = res['img_id']
        zip_res = tuple([gt, pred])
        zip_results.append(zip_res)
    evaluator.compute_metrics(zip_results)



