import os.path as osp
import os
import cv2
import numpy as np
import json
import subprocess
import shutil

from mmcv import Config
from mmdet.apis import init_detector
from mmdet.apis import inference_detector

import PIL.Image

CFG_NAME = 'train_coco.py'
mAP_name = 'mAP.txt'
TEST_DIR = '/home/PycharmProjects/data/STM7395767'
THRESH = 0.3 # the same as in pycocotools.cocoeval

def mAP(model_dir, coco_for_mAP, test_dir=TEST_DIR):
    
    mAP_dir = osp.join(model_dir, 'mAP')
    if not osp.isdir(mAP_dir):
        os.makedirs(mAP_dir)
    
    path_to_ckpt = osp.join(model_dir, 'latest.pth')    
    path_to_cfg = osp.join(model_dir, CFG_NAME)
    
    
    # build model
    model = init_detector(path_to_cfg, path_to_ckpt)
    
    summary = ''
    output = ''
#    file_names = []
#    imgs = []
#    gt_cnts_imgs = []
    for coco_name in coco_for_mAP:
        # modify config for mAP calculations
        print(f'calculating mAP for {coco_name}   ', end='')
        cfg = Config.fromfile(path_to_cfg)
        cfg.data.test.data_root = test_dir
        cfg.data.test.ann_file = coco_name
        cfg.dump(path_to_cfg)
        
        cmd_str = f'python3 /mmdetection/tools/test.py {path_to_cfg} {path_to_ckpt} --eval segm'
        
        # calculate mAP using pycocotools util
        p = subprocess.run(cmd_str.split(), capture_output=True)
        
        # parse util output for summary
        for line in p.stdout.decode().split('\n'):
            if 'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ]' in line:
                mAP = line[-5:]
                print(mAP)
                info = f'{coco_name};  {line[-5:]} \n'
                summary += info
                output += p.stdout.decode()
                
             
        # visualization block
        path_to_coco = osp.join(TEST_DIR, coco_name)
        with open(path_to_coco, 'r') as f:
            coco = json.load(f)
        
        classes = [cat['name'] for cat in coco['categories']]
        model.CLASSES = classes
        
        for img_ann in coco['images']:
            img_id = img_ann['id']
            file_name = img_ann['file_name']
                           
            gt_cnts = [ann['segmentation'] for ann in coco['annotations'] if ann['image_id']==img_id]
            gt_cnts = [np.asarray(cnt).astype(np.int32).reshape((-1,1,2)) for cnt in gt_cnts]

            path_to_img = osp.join(test_dir, file_name)
            img = cv2.imread(path_to_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           
            bboxes, segms = inference_detector(model, img) 
                            
            # follow by only the first category in the list
            bbs = bboxes[0]
            sgms = segms[0]                
            inds = np.where(bbs[:,4]>THRESH)[0]
            dt_cnts = []
            for ind in inds:
                cnts, _ = cv2.findContours(sgms[ind].astype(np.uint8), 1, 2)
                if len(cnts):
                    dt_cnts.append(cnts[0])
                    
            img = cv2.drawContours(img, gt_cnts, -1, (255,0,0), 1)
            img = cv2.drawContours(img, dt_cnts, -1, (0,255,0), 1)
#            display(PIL.Image.fromarray(img))
            dst_path = osp.join(mAP_dir, file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(dst_path,img)
            
    path_to_mAP = osp.join(mAP_dir, mAP_name)            
    with open(path_to_mAP, 'w') as f:
        f.write('annotation file; mAP\n')
        f.write(summary)
        f.write('full output\n')
        f.write('\n')
        f.write(output)    
        
    path_to_zip = osp.join(model_dir, 'mAP')
    shutil.make_archive(path_to_zip, 'zip', mAP_dir) 
