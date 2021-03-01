import argparse
import json
import os
import os.path as osp
import shutil
import cv2
import numpy as np


from preprocess import extract_imgs_from_labelme
from preprocess import preprocess
from train_utils import train_loop

from predict import predict
from inference import inference

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--net", help="unique name for network model, \"sph\" for spheres, \"stm\" for stm etc")
    parser.add_argument("-t", "--thresh", default='0.3', help="prediction's threshold probability")
    parser.add_argument("-e", "--epochs", default='100', help="number of training epochs")

    args = parser.parse_args()
    args.thresh = float(args.thresh)
    args.epochs = int(args.epochs)

    #specify here the actual folder with data
    # labelme annotations file
    model_dir = osp.join('./data', args.net)
    
    path_to_img = extract_imgs_from_labelme(model_dir)
    preprocess(model_dir)

    #specify here the actual folder with data
    src_cfg_path = '/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py'
    src_ckpt_path = './data/stock/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth'
    
    train_loop(src_cfg_path, src_ckpt_path, model_dir, args.epochs)    
    
    with open('/home/DLgram/data/floto_crop/train_coco.json', 'r') as f:
        coco = json.load(f)
    
    classes = [cat['name'] for cat in coco['categories']]
    with open('config.json', 'r') as f:
        cfg = json.load(f)
        
    cfg['net'][args.net] = dict(  model_dir = model_dir,
                                  path_to_config = osp.join(model_dir, "train_coco.py"),
                                  path_to_checkpoint = osp.join(model_dir, f"epoch_{args.epochs}.pth"),
                                  classes = classes
                               )
    
    with open('config.json', 'w') as f:
        json.dump(cfg, f)
    
    # test block
    #copy image into test_dir
    img_name = osp.basename(path_to_img)
    test_dir = osp.join(model_dir, 'test')
    
    os.makedirs(test_dir)
    dst_img_path = osp.join(test_dir, img_name)
    shutil.copyfile(path_to_img, dst_img_path)
    
    inference(dst_img_path, args.net, args.thresh)