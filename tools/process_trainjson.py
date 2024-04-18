import json
import mmengine
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--train_json_path', type=str, default='data/coco/train/train.json')
    parser.add_argument(
        '--train_out_json', type=str, default='data/coco/annotations/instances_train2017.json')
    parser.add_argument(
        '--val_json_path', type=str, default='data/coco/val/val.json')
    parser.add_argument(
        '--val_out_json', type=str, default='data/coco/annotations/instances_val2017.json')

    args = parser.parse_args()

    return args



def process(args):
    json_file_path = args.train_json_path
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    out_json_data = dict()
    out_json_data['images'] = json_data['images']
    out_json_data['categories'] = json_data['categories']
    anno_list = []
    for anno in json_data['annotations']:
        anno['category_id'] = anno['category_id'] - 1
        anno_list.append(anno)
    out_json_data['annotations'] = anno_list
    out_json_data['categories'][0] = {'id': 0, 'name': 'car'}
    out_json_data['categories'][1] = {'id': 1, 'name': 'truck'}
    out_json_data['categories'][2] = {'id': 2, 'name': 'bus'}
    out_json_data['categories'][3] = {'id': 3, 'name': 'van'}
    out_json_data['categories'][4] = {'id': 4, 'name': 'freight_car'}

    with open(args.train_out_json, 'w') as fp:
        json.dump(out_json_data, fp)


    # ---------------------------------------
    json_file_path = args.val_json_path
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    out_json_data = dict()
    out_json_data['images'] = json_data['images']
    out_json_data['categories'] = json_data['categories']
    anno_list = []
    for anno in json_data['annotations']:
        anno['category_id'] = anno['category_id'] - 1
        anno_list.append(anno)
    out_json_data['annotations'] = anno_list
    out_json_data['categories'][0] = {'id': 0, 'name': 'car'}
    out_json_data['categories'][1] = {'id': 1, 'name': 'truck'}
    out_json_data['categories'][2] = {'id': 2, 'name': 'bus'}
    out_json_data['categories'][3] = {'id': 3, 'name': 'van'}
    out_json_data['categories'][4] = {'id': 4, 'name': 'freight_car'}

    with open(args.val_out_json, 'w') as fp:
        json.dump(out_json_data, fp)



if __name__ == '__main__':
    args = parse_args()
    process(args)