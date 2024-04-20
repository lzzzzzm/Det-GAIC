import json
import mmengine
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument(
        '--pkl_file', type=str, default='test.pkl')
    parser.add_argument(
        '--out_json', type=str, default='out_json/project/out.json')
    parser.add_argument(
        '--score_threshold', type=float, default=0.1)

    args = parser.parse_args()

    return args

def check_image_id(json_data):
    image_id = set()
    for data in json_data:
        image_id.add(data['image_id'])
    for i in range(1, 1001):
        if i not in image_id:
            miss_data_dict = dict()
            miss_data_dict['image_id'] = i
            miss_data_dict['category_id'] = None
            miss_data_dict['bbox'] = None
            miss_data_dict['score'] = None
            json_data.append(miss_data_dict)
            print('Miss object in image_id:{}'.format(i))
    return json_data


def convert_pkl2json(args):
    pkl_file_path = args.pkl_file
    pkl_data = mmengine.load(pkl_file_path)
    out_json_data = []
    for data in pkl_data:
        for box, label, score in zip(data['pred_instances']['bboxes'], data['pred_instances']['labels'], data['pred_instances']['scores']):
            if score < args.score_threshold:
                continue
            json_dict = dict()
            json_dict['image_id'] = int(data['img_id'])
            box = np.around(box.cpu().numpy(), 1)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            x1, y1, w, h = np.around(float(x1), 1), np.around(float(y1), 1), np.around(float(w), 1), np.around(float(h), 1)
            write_box = [x1, y1, w, h]
            json_dict['bbox'] = write_box
            json_dict['category_id'] = int(label.item() + 1)
            json_dict['score'] = np.around(float(score), 2)
            out_json_data.append(json_dict)
    out_json_data = check_image_id(out_json_data)
    with open(args.out_json, 'w') as fp:
        json.dump(out_json_data, fp)



if __name__ == '__main__':
    args = parse_args()
    convert_pkl2json(args)