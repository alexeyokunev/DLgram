import argparse
import json
import os
import os.path as osp
import shutil
import cv2
import numpy as np
import subprocess
    
from preprocess import extract_imgs_from_labelme
from preprocess import preprocess
from train_utils import train_loop

from predict import predict
from inference import inference
from mAP import mAP

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net", help="unique name for network model")
    parser.add_argument("-t", "--thresh", default='0.3', help="prediction's threshold probability")
    parser.add_argument("-e", "--epochs", default='100', help="number of training epochs")
    parser.add_argument("-r", "--ratio", default='1.5', help="ration of crop size to bounding box size")
    parser.add_argument("-a", "--augment", default='1', help="number of augmented crops")
    parser.add_argument("-m", "--merge", default='0', help="merge classes")
    parser.add_argument("-g", "--gt", default='0', help="calculate mAP for test dataset")

    args = parser.parse_args()
    args.thresh = float(args.thresh)
    args.epochs = int(args.epochs)
    args.ratio = float(args.ratio)
    args.augment = int(args.augment)
    args.merge = int(args.merge)
    args.gt = int(args.gt)

    #specify here the actual folder with data
    # labelme annotations file
    model_dir = osp.join('../data', args.net)
    
    path_to_imgs = preprocess(model_dir, args.ratio, args.augment)
    path_to_img = path_to_imgs[0]
    print(path_to_img)

    #specify here the actual folder with data
    src_cfg_path = '/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py'
    src_ckpt_path = '../data/stock/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth'
    
    train_loop(src_cfg_path, src_ckpt_path, model_dir, args.epochs)

    path_to_config = osp.join(model_dir, 'train_coco.py')
    path_to_train_coco = osp.join(model_dir, 'train_coco.json')
    with open(path_to_train_coco, 'r') as f:
        coco = json.load(f)
    
    classes = [cat['name'] for cat in coco['categories']]
    with open('config.json', 'r') as f:
        cfg = json.load(f)
        
    cfg['net'][args.net] = dict(  model_dir = model_dir,
                                  path_to_config = path_to_config,
                                  path_to_checkpoint = osp.join(model_dir, f"epoch_{args.epochs}.pth"),
                                  classes = classes
                               )
    
    with open('config.json', 'w') as f:
        json.dump(cfg, f)
    
    # test block
    # copy image into test_dir
    img_name = osp.basename(path_to_img)
    test_dir = osp.join(model_dir, 'test')
    
    try:
        os.makedirs(test_dir)
    except:
        pass
    
    dst_img_path = osp.join(test_dir, img_name)
    print(path_to_img, dst_img_path)
    shutil.copyfile(path_to_img, dst_img_path)
    
    inference(dst_img_path, args.net, args.thresh, args.merge)
    
    if args.gt:
        self_coco = args.net.split('_')[0] + '.json'
        coco_for_mAP = [self_coco] + ['test.json', '100.json', '101.json', '102.json']
        mAP(model_dir, coco_for_mAP)
