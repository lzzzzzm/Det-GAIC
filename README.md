# Det-GAIC

## Prepare Data

Download the dataset from [GAIC](https://www.heywhale.com/org/2024gaiic/competition/area/65f7abcf019d8282037f3924/content/2).

Make the data directory like this:

```
Det-GAIC
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train
│   │   │   ├── rgb
│   │   │   ├── tir
│   │   │   ├── train.json
│   │   ├── val
│   │   │   ├── rgb
│   │   │   ├── tir
│   │   │   ├── val.json
```
## Install

```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
pip install -v -e .
```

## Training

```bash
bash tools/dist_train.sh configs 8
```

## Testing

```bash
bash tools/dist_test.sh config epochs 8 --eval bbox
```

or output the pkl results

```bash
python tools/test.py config epochs --out test.pkl
```

## Evaluation with pkl

```bash
python tools/eval_from_pkl.py --pkl_file test.pkl --ann_file data/coco/annotations/instances_val2017.json
```

## Visualization with pkl

```bash
python tools/vis_from_pkl.py --pkl_file test.pkl --ann_file data/coco/annotations/instances_val2017.json
```
