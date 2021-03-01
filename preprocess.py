import labelme2coco
import os
import os.path as osp
import json
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

import base64
import io
import PIL.Image

RAW_COCO_NAME = 'raw_train_coco.json'
TRAIN_COCO_NAME = 'train_coco.json'

def extract_imgs_from_labelme(proj_dir):
    for file in [f for f in os.listdir(proj_dir) if f.endswith('json')]:
        json_name = file.split('.')
        json_ext = json_name[-1]
        json_base_name = ''.join(json_name[:-1])
        path_to_json = osp.join(proj_dir, file)
        with open(path_to_json, 'r') as f:
            annots = json.load(f)
        img_b64 = annots['imageData']
        img_data = base64.b64decode(img_b64) 
        f = io.BytesIO()
        f.write(img_data)
        img_pil = PIL.Image.open(f)
        img_name = annots['imagePath'].split('.')
        img_ext = img_name[-1]
        if img_ext in ('bmp', 'jpg', 'jpeg', 'png', 'JPG', 'JPEG'):
            img_name = json_base_name + '.' + img_ext
        else:
            img_name = json_base_name + '.' + 'jpg'
        print(img_name)
        annots['imagePath'] = img_name
#        print(img_base_name, img_ext, img_name, json_base_name)
        path_to_img = osp.join(proj_dir, img_name)
        img_pil.save(path_to_img)  
        
        with open(path_to_json, 'w') as f:
            json.dump(annots, f)
            
        return path_to_img
            
def lblme_to_coco(proj_dir):
    # set path for coco json to be saved
    raw_json_path = osp.join(proj_dir, RAW_COCO_NAME)
    if osp.isfile(raw_json_path):
        os.remove(raw_json_path)

    # convert labelme annotations to coco
    labelme2coco.convert(proj_dir, raw_json_path)

    with open(raw_json_path, 'r') as f:
        raw_train_coco = json.load(f)
    
    # fix path to image
    raw_train_coco['images'][0]['file_name'] = osp.basename(raw_train_coco['images'][0]['file_name'])
    
    # fix segmentation format 
    crop_ids = [cat['id'] for cat in raw_train_coco['categories'] if cat['name']=='crop']
    assert len(crop_ids)==1, 'zero or more than one categories with named "crop"'
    crop_id = crop_ids[0]

    for ann in raw_train_coco['annotations']:
        if ann['category_id']==crop_id:
            l,t,w,h = ann['bbox']
            r, b = l+w, t+h
            ann['segmentation'][0] = [l,t,r,t,r,b,l,b]
            print(f'crop at {l, t, r, b}')

    with open(raw_json_path, 'w') as f:
        json.dump(raw_train_coco, f)
        
def get_full_edge_cnts(cnts, crop_corners):
    """clip contours to crop"""
    t, l, b, r = crop_corners
    w, h = r-l, b-t
    clipped_cnts = [np.asarray(cnt.reshape((-1,2)) - np.asarray([l,t])) for cnt in cnts]
    clipped_cnts = [np.clip(cnt, [0,0], [w,h]).reshape((-1,1,2)) for cnt in clipped_cnts]
    full_cnts = []
    edge_cnts = []
    for с_cnt, cnt in zip(clipped_cnts, cnts):
        if cv2.contourArea(с_cnt) > 0:
            if (cv2.contourArea(cnt)-cv2.contourArea(с_cnt))**2<cv2.contourArea(cnt):
                full_cnts.append(с_cnt)
            else:
                edge_cnts.append(с_cnt)
                
    return len(full_cnts + edge_cnts), full_cnts, edge_cnts

def init_coco():
    coco = {}
    coco['info'] = {"description": "", 
                    "url": "", 
                    "version": "", 
                    "year": 2020, 
                    "contributor": "okunev_alexey", 
                    "date_created": str(datetime.now())
                   }, 
    coco['licenses'] = {"id": 1, 
                        "name": "Attribution-NonCommercial-ShareAlike License", 
                        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"}, 
    coco['categories'] = [{"id": 1, "name": "full", "supercategory": ""},
                          {"id": 2, "name": "edge", "supercategory": ""}
                         ]
    coco['images'] = []
    coco['annotations'] = []
    return coco

def append_img_to_coco(coco, name, crop_corners, full_cnts, edge_cnts):
    # append image name and parameters
    t, l, b, r = crop_corners
    new_img_id = len(coco['images']) + 1
    new_img = {"id": new_img_id, 
               "file_name": name, 
               "width": r-l, 
               "height": b-t, 
               "date_captured": "", 
               "license": 1, 
               "coco_url": "", 
               "flickr_url": ""}
    coco['images'].append(new_img)
    
    # append annotations
    for cnt in full_cnts:
        c_l, c_t = np.min(cnt, axis = 0).ravel()
        c_r, c_b = np.max(cnt, axis = 0).ravel()
        new_ann_id = len(coco['annotations']) + 1
        new_ann = {"id": new_ann_id, 
                   "image_id": new_img_id, 
                   "category_id": 1, 
                   "iscrowd": 0, 
                   "bbox": [int(c_l), int(c_t), int(c_r-c_l), int(c_b-c_t)], 
                   "area": cv2.contourArea(cnt), 
                   "segmentation": [cnt.astype(np.int32).ravel().tolist()],
                   "width": r-l, 
                   "height": b-t}
        coco['annotations'].append(new_ann)

    for cnt in edge_cnts:
        c_l, c_t = np.min(cnt, axis = 0).ravel()
        c_r, c_b = np.max(cnt, axis = 0).ravel()
        new_ann_id = len(coco['annotations']) + 1
        new_ann = {"id": new_ann_id, 
                   "image_id": new_img_id, 
                   "category_id": 2, 
                   "iscrowd": 0, 
                   "bbox": [int(c_l), int(c_t), int(c_r-c_l), int(c_b-c_t)], 
                   "area": cv2.contourArea(cnt), 
                   "segmentation": [cnt.astype(np.int32).ravel().tolist()],
                   "width": r-l, 
                   "height": b-t}
        coco['annotations'].append(new_ann)
    return coco

def preprocess(proj_dir):
    num_full_cnts = 0
    num_edge_cnts = 0
    #convert labelme to coco 
    lblme_to_coco(proj_dir)
    
    # load raw coco, converted from labelme
    raw_json_path = osp.join(proj_dir, RAW_COCO_NAME)
    with open(raw_json_path, 'r') as f:
        raw_train_coco = json.load(f)

    # load full image, use it's name without extension as a prefix
    path_to_src_img = osp.join(proj_dir, raw_train_coco['images'][0]['file_name'])
    img = cv2.imread(path_to_src_img)
    img_name = osp.basename(path_to_src_img)
    src_img_name = img_name.split('.')[0]

    # extract crops coordinates
    crop_ids = [cat['id'] for cat in raw_train_coco['categories'] if cat['name']=='crop']
    assert len(crop_ids)==1, 'zero or more than one categories with named "crop"'

    crop_id = crop_ids[0]
    crops = [ann['bbox'] for ann in raw_train_coco['annotations'] if ann['category_id']==crop_id]
    crops = [[int(t), int(l), int(t+h), int(l+w)] for l,t,w,h in crops]
    if len(crops)==1:
        crops *= 5

    # extract particle's contours FOR SINGLE CLASS ONLY
    cnts = [ann['segmentation'] for ann in raw_train_coco['annotations'] if ann['category_id']!=crop_id]
    cnts = [np.asarray(cnt).reshape((-1,2)).astype(np.int32) for cnt in cnts]

    # make training dataset
    coco = init_coco()
    for crop_corners in crops:
        t, l, b, r = crop_corners
        crop = img[t:b, l:r, :].copy()
        name = f'{src_img_name}_{t}_{l}_{b}_{r}.bmp'
        num_cnts, full_cnts, edge_cnts = get_full_edge_cnts(cnts, crop_corners)
        num_full_cnts += len(full_cnts)
        num_edge_cnts += len(edge_cnts)
        print(f'{num_cnts} found in {name}')
        if num_cnts:
            print(f'saving {name}')
            dst_img_path = osp.join(proj_dir, name)
            cv2.imwrite(dst_img_path, crop)
            append_img_to_coco(coco, name, crop_corners, full_cnts, edge_cnts)

    # save train coco
    train_coco_path = osp.join(proj_dir, TRAIN_COCO_NAME)
    with open(train_coco_path, 'w') as f:
        json.dump(coco, f)
        
    print(f'{num_full_cnts} full contours, {num_edge_cnts} edge contours in total')

def plot_contours(crop, full_cnts, edge_cnts):
    img = crop.copy()
    cv2.drawContours(img, full_cnts, -1, (0, 255, 0), 2)
    cv2.drawContours(img, edge_cnts, -1, (0, 0, 255), 2)
    return img

def plot_coco(coco, proj_dir):
    # leave only those images that are not duplicated
    # required for single crop case
    uniq_images = []
    uniq_image_names = []
    for img_ann in coco['images']:
        if img_ann['file_name'] not in uniq_image_names:
            uniq_images.append(img_ann)
            uniq_image_names.append(img_ann['file_name'])
            
    for img_ann in uniq_images:
        image_id = img_ann['id']
        path_to_img = osp.join(proj_dir, img_ann['file_name'])
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        cnts_1 = [ann['segmentation'] for ann in coco['annotations'] if ann['category_id']==1 and ann['image_id']==image_id]
        cnts_1 = [np.asarray(cnt).reshape((-1,1,2)).astype(np.int32) for cnt in cnts_1]
        cnts_2 = [ann['segmentation'] for ann in coco['annotations'] if ann['category_id']==2 and ann['image_id']==image_id]
        cnts_2 = [np.asarray(cnt).reshape((-1,1,2)).astype(np.int32) for cnt in cnts_2]
        
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(plot_contours(img, cnts_1, cnts_2))
        plt.show()
        
def show(proj_dir):    
    raw_json_path = osp.join(proj_dir, RAW_COCO_NAME)
    with open(raw_json_path, 'r') as f:
        raw_train_coco = json.load(f)
    plot_coco(raw_train_coco, proj_dir)
    
    train_coco_path = osp.join(proj_dir, TRAIN_COCO_NAME)
    with open(train_coco_path, 'r') as f:
        train_coco = json.load(f)
    plot_coco(train_coco, proj_dir)
    
if __name__ == "__main__":
    preprocess(proj_dir)
#    show(proj_dir)
    