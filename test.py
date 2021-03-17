import os.path as osp
import json
import cv2
import numpy as np
from train_utils import train_loop

def test_coco(path_to_coco):

    with open(path, 'r') as f:
        coco = json.load(f)

    anns = coco['annotations']
    imgs = coco['images']
    cats = coco['categories']
    for i, ann in enumerate(anns):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        img_name = [img['file_name'] for img in imgs if img['id']==img_id]
        img_path = osp.join(proj_dir, img_name[0])
        cnt = np.asarray(ann['segmentation']).astype(np.int32).reshape((-1,1,2))
        img = cv2.imread(img_path)
        try:
            cv2.drawContours(img, [cnt], -1, (0,0,0), -1)
        except:
            print(ann)

model_dir = osp.join('../data', 'tile_3')
# specify here the actual folder with data
src_cfg_path = '/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco.py'
src_ckpt_path = '../data/stock/cascade_mask_rcnn_x101_64x4d_fpn_1x_coco_20200203-9a2db89d.pth'

train_loop(src_cfg_path, src_ckpt_path, model_dir, 11)

#proj_dir = '/home/PycharmProjects/data/tile_3'
#file_name = r'train_coco.json'
#path = osp.join(proj_dir, file_name)
#print(path)
#test_coco(path)

#cmd = 'python3 train.py -n tile_3 -t 0.1 -e 11'
#PIPE = subprocess.PIPE
#p = subprocess.run(cmd.split(), stdout=PIPE, stderr=PIPE)
#print(p.stdout, p.stderr)