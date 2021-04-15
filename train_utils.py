import cv2
import numpy as np
import json
import os
import os.path as osp

from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.apis import set_random_seed

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet.apis import single_gpu_test
from mmdet.apis import train_detector
#from mmdet.datasets import build_dataloader
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

import datetime

DATASET_NAME = 'Particles'

def register_dataset(path_to_train_coco):
    with open(path_to_train_coco, 'r') as f:
        train_coco = json.load(f)
    
    clses = tuple([cat['name'] for cat in train_coco['categories']])

    try:
        @DATASETS.register_module()
        class Particles(CocoDataset):
            CLASSES = clses
        print(f'registered dataset {DATASET_NAME} with classes {clses}')    
    except KeyError:
        print(f'dataset {DATASET_NAME} with classes {clses} have been already registered')
    return clses
        
def modify_cfg(cfg, ckpt_path, dataset_name, path_to_train_coco, num_epochs):
    root_dir = osp.dirname(path_to_train_coco)
    train_coco_name = osp.basename(path_to_train_coco)
    
    # load train coco
    with open(path_to_train_coco, 'r') as f:
        train_coco = json.load(f)
        
    # Modify dataset type and path
    cfg.dataset_type = dataset_name
    cfg.data_root = root_dir

    cfg.data.train.type = dataset_name
    cfg.data.train.data_root = root_dir
    cfg.data.train.ann_file = train_coco_name
    cfg.data.train.img_prefix = 'images'

    cfg.data.val.type = dataset_name
    cfg.data.val.data_root = root_dir
    cfg.data.val.ann_file = train_coco_name
    cfg.data.val.img_prefix = 'images'

    cfg.data.test.type = dataset_name
    cfg.data.test.data_root = root_dir
    cfg.data.test.ann_file = ''
    cfg.data.test.img_prefix = ''

    # modify num classes of the model in box head
    num_classes = len(train_coco['categories'])
    for bb_head in cfg.model.roi_head.bbox_head:
        bb_head['num_classes'] = num_classes
    
    cfg.model.roi_head.mask_head.num_classes = num_classes
    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = ckpt_path

    # Set up working dir to save files and logs.
    cfg.work_dir = root_dir


    # Set image size
    w, h = train_coco['images'][0]['width'], train_coco['images'][0]['height']
    img_size = w * h
    cfg.train_pipeline[2]['img_scale'] = w, h
    cfg.test_pipeline[1]['img_scale'] = w, h
    cfg.data.train.pipeline[2]['img_scale'] = w, h
    cfg.data.val.pipeline[1]['img_scale'] = w, h
    cfg.data.test.pipeline[1]['img_scale'] = w, h
    
    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # set general learning parameters 
    cfg.lr_config.step = [int(num_epochs*0.75), int(num_epochs*0.95)]
    cfg.total_epochs = num_epochs
    cfg.runner.max_epochs = num_epochs
    cfg.checkpoint_config.interval = cfg.total_epochs
    cfg.data.workers_per_gpu = 2
    cfg.data.samples_per_gpu = min(2, len(train_coco['images']), 2**20 // img_size)
    cfg.gpu_ids = range(1)

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU
    # and multiply by number of gpus
    cfg.optimizer.lr = 0.01 * 2/ cfg.data.samples_per_gpu * len(list(cfg.gpu_ids)) / 8 
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 1

    # at present evaluation doesn't work on multiple GPUs
    # so disable evaluation
    cfg.evaluation.interval = 10e6

    cfg.data.test.pipeline[0].type = 'LoadImageFromFile'

    max_per_img = 4000
    max_rpn = 16000
    cfg.model.test_cfg.rpn.nms_pre = max_rpn
    cfg.model.test_cfg.rpn.nms_post = max_rpn
    cfg.model.test_cfg.rpn.max_per_img = max_rpn
    cfg.model.test_cfg.rcnn.max_per_img = max_per_img

    nms_thr = 0.15 #0.15
    iou_thr = 0.2 #0.2
    cfg.model.test_cfg.rpn.iou_threshold = nms_thr
    cfg.model.test_cfg.rcnn.nms.iou_threshold = iou_thr
    
    return cfg

def train_loop(path_to_cfg, path_to_ckpt, proj_dir, num_epochs):
    path_to_train_coco = osp.join(proj_dir, 'train_coco.json')
    clses = register_dataset(path_to_train_coco)    
          
    # read and modify config
    cfg = Config.fromfile(path_to_cfg)
    cfg = modify_cfg(cfg, path_to_ckpt, DATASET_NAME, path_to_train_coco, num_epochs)
    path_to_dst_cfg = osp.join(proj_dir, 'train_coco.py')
    cfg.dump(path_to_dst_cfg)

    # building dataset
    coco_name = osp.basename(path_to_train_coco)
    print(f'building dataset for {coco_name}')
    datasets = [build_dataset(cfg.data.train)]
    
    print(f'building model for {coco_name}')
    model = build_detector(cfg.model)
    model.CLASSES = datasets[0].CLASSES

    # rename output weights file to preserve it from overwriting in the next loop
    print(f'training model for {coco_name}')
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    
if __name__ == '__main__':
    train_loop(path_to_cfg, path_to_ckpt, proj_dir, num_epochs)