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


CROP_LABELS = ['Crop', 'crop', 'CROP']
TRAIN_COCO_NAME = 'train_coco.json'
IMG_DIR_PREFIX = 'images'

def extract_imgs_from_labelme(model_dir, file):
    # splitname for name and extension
    json_name = file.split('.')
    json_ext = json_name[-1]
    json_base_name = ''.join(json_name[:-1])
    # remove _labelme mark
    json_base_name = json_base_name[:-8]
        
    path_to_json = osp.join(model_dir, file)
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
    path_to_img = osp.join(model_dir, img_name)
    try:
        img_pil.save(path_to_img)  
    except:
        img_pil.convert('RGB').save(path_to_img)
        
    with open(path_to_json, 'w') as f:
        json.dump(annots, f)
            
    return path_to_img

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
    coco['categories'] = []
    coco['images'] = []
    coco['annotations'] = []
    return coco

def get_points(shape):
    if shape['shape_type']=='polygon':
        points = shape['points']
    elif shape['shape_type']=='circle':
        center, point = np.asarray(shape['points'])
        rad = np.linalg.norm(point - center)
        
        points = []
        for i in range(16):
            angle = i/16 * 2 * np.pi
            projs = np.asarray([np.cos(angle), np.sin(angle)])
            point = center + rad * projs
            points.append(point)
    elif shape['shape_type']=='rectangle':
        lt, rb = shape['points']
        l, t = lt
        r, b = rb
        points = [[l,t], [r,t], [r,b], [l,b]]
        
    return np.asarray(points).reshape((-1,2))
        
def get_crop_ltwhs(imgs_points):
    imgs_whs = []
    for img_points in imgs_points:
        img_whs = []
        for cnt in img_points:
            min_x, min_y = np.min(cnt, axis=0)
            max_x, max_y = np.max(cnt, axis=0)
            max_x, max_y = max_x + 1, max_y + 1
            w, h = round(max_x - min_x, 0), round(max_y - min_y, 0)
            img_whs.append([min_x, min_y, w, h])
        imgs_whs.append(np.asarray(img_whs))   
    return imgs_whs

def find_max_wh(imgs_whs):
    max_imgs_wh = [np.max(img_whs, axis=0)[2:] for img_whs in imgs_whs]
    max_imgs_wh += np.asarray([64,64])
    max_imgs_wh = np.asarray(max_imgs_wh)
    print(max_imgs_wh)
    max_wh = np.max(max_imgs_wh, axis=0)
    return max_wh

def append_img_to_coco(coco, img_name, crop_tlbr, cnts, labels):
    # append image name and parameters
    t, l, b, r = crop_tlbr.ravel().astype(np.int32).tolist()
    crop_w, crop_h = r-l, b-t
    img_id = len(coco['images']) + 1
    new_img = {"id": img_id, 
               "file_name": img_name, 
               "width": crop_w, 
               "height": crop_h, 
               "date_captured": "", 
               "license": 1, 
               "coco_url": "", 
               "flickr_url": ""}
    coco['images'].append(new_img)
    
    # append category
    cats = coco['categories']
    coco_labels = [cat['name'] for cat in cats]
    
    # append annotation
    crop_lt = np.asarray([l, t])
    for cnt, label in zip(cnts, labels):
        # exclude crop contours
        if label in CROP_LABELS:
            continue
                
        # add new category if label was not accounted before
        if label in coco_labels:
            cat_id = coco_labels.index(label) + 1
        else:
            coco_labels.append(label)
            cat_id = len(cats) + 1
            new_cat = {"id": cat_id, 
                       "name": label, 
                       "supercategory": ""
                      }
            coco['categories'].append(new_cat)
        
        # clip contour to crop
        cnt = cnt.copy()
        cnt -= crop_lt
        cnt = np.clip(cnt, (0,0), (crop_h, crop_w)).astype(np.int32)
        # exclude external contours
        if not cv2.contourArea(cnt.reshape((-1,1,2))):
            continue
            
        # calculate bbox
        bb_l, bb_t = np.min(cnt, axis=0).tolist()
        bb_r, bb_b = np.max(cnt, axis=0).tolist()
        bb_w, bb_h = bb_r - bb_l + 1, bb_b - bb_t + 1
        
        ann_id = len(coco['annotations']) + 1
        new_ann = { "id": ann_id, 
                    "image_id": img_id, 
                    "category_id": cat_id, 
                    "iscrowd": 0, 
                    "bbox": [bb_l, bb_t, bb_w, bb_h], 
                    "area": cv2.contourArea(cnt.reshape((-1,1,2))), 
                    "segmentation": [cnt.ravel().tolist()],
                    "width": int(r-l), 
                    "height": int(b-t)}
        coco['annotations'].append(new_ann)

    return coco

def objects2coco(coco, imgs, src_img_names, imgs_points, imgs_cats, model_dir, ratio, augment):
    """
    objects based training.  
    images for train coco are cropped around annonotated objects, one image per object
    some data augmentation realized with position of object in image
    """
    imgs_crop_ltwhs = get_crop_ltwhs(imgs_points)
    # search for the largest shape size
    max_wh = find_max_wh(imgs_crop_ltwhs) * ratio
    max_w, max_h = max_wh.astype(np.int32)
    
    # crop images to image folder
    img_dir = osp.join(model_dir, 'images')
    if not osp.isdir(img_dir):
        os.makedirs(img_dir)
    
    num_imgs = len(src_img_names)
    for i in range(num_imgs):
        img = imgs[i]  
        src_img_name = src_img_names[i]
        img_cats = imgs_cats[i]
        img_points = imgs_points[i]
        img_crop_ltwhs = imgs_crop_ltwhs[i]
        num_cnts = img_points.shape[0]
        if num_imgs==1 and num_cnts==1 and augment==1:
            augment = 2
            
        for j in range(num_cnts):
            cnt = img_points[j] 
            bbox = img_crop_ltwhs[j].copy()
            label = img_cats[j]
            # augmentation
            for k in range(augment):
                # crop proportional to bounding box size
                dwh = bbox[2:] * (ratio - 1) * np.random.rand(2) 
                l, t = bbox[:2] - dwh
                r, b = bbox[:2] - dwh + bbox[2:] * ratio
                
                # crop equal to maximal crop size
                dwh = (max_wh - bbox[2:]) * np.random.rand(2)
                l, t = bbox[:2] - dwh
                r, b = l + max_w, t + max_h
                
#                print(f'bbox {bbox}, crop {max_wh}, l={l:.1f}, t={t:.1f}, r={r:.1f}, b={b:.1f}')
                crop_tlbr = np.asarray([t,l,b,r]).astype(np.int32)
                crop_tlbr = np.clip(crop_tlbr.reshape((-1,2)), [0,0], img.shape[:2])
                t, l, b, r = crop_tlbr.ravel()
                bmp_ind = np.random.randint(2**20)
                new_img_name = f'{src_img_name}_{bmp_ind}.bmp'
                new_img_path = osp.join(img_dir, new_img_name)
                crop_w, crop_h = int(r-l), int(b-t)
                
                cv2.imwrite(new_img_path, img[t:t+crop_h, l:l+crop_w, :])
                
                coco  = append_img_to_coco(coco, new_img_name, crop_tlbr, [cnt], [label])
                
    return coco

def crops2coco(coco, imgs, src_img_names, imgs_points, imgs_cats, model_dir, ratio, augment):
    """
    crop based training
    crops extracted from user annotations. 
    some data augmentation realized with position of the crop in image
    if crop size more then n*64 x m*64 (n, m integers) - position of n*64 x m*64 crop chosen randomly to fit into user crop
    All object inside crop are valid.
    """
    
    # get crop sizes as minimal value of crops in all images
    crops_whs = []
    for img_points, img_cats in zip(imgs_points, imgs_cats):
        crop_whs = []
        for cnt, label in zip(img_points, img_cats):
            if label in CROP_LABELS:
                min_x, min_y = np.min(cnt, axis=0)
                max_x, max_y = np.max(cnt, axis=0)
                max_x, max_y = max_x + 1, max_y + 1
                w, h = round(max_x - min_x, 0), round(max_y - min_y, 0)
                crop_whs.append([min_x, min_y, w, h])
                
        crops_whs.append(np.asarray(crop_whs)) 
        
    min_crop_wh = [np.min(crop_whs, axis=0)[2:] for crop_whs in crops_whs]
    min_crop_wh = np.asarray(min_crop_wh)
    # bring crop sizes to 64 multiplier
    fix_w, fix_h = (np.min(min_crop_wh, axis=0)//64 * 64).astype(np.int32)
    fix_w, fix_h = max(fix_w, 64), max(fix_h, 64)

    # crop images to image folder
    img_dir = osp.join(model_dir, 'images')
    if not osp.isdir(img_dir):
        os.makedirs(img_dir)

    num_imgs = len(src_img_names)
    for i in range(num_imgs):
        img = imgs[i]  
        src_img_name = src_img_names[i]
        img_cats = imgs_cats[i]
        img_points = imgs_points[i]
        crop_whs = crops_whs[i]
        num_crops = len(crop_whs)
        if num_imgs==1 and num_crops==1 and augment==1:
            augment = 2
            
        # augment by cloning crops
        crop_whs = [crop for crop in crop_whs]
        crop_whs *= augment
        for crop in crop_whs:
            # make random crop of fix size within specified area
            min_x, min_y, w, h = crop.astype(np.int32).tolist()
            l = min_x + np.random.randint(w - fix_w)
            t = min_y + np.random.randint(h - fix_h)
            crop_tlbr = np.asarray([t,l,t+fix_h,l+fix_w])
            
            # name to save croppped image                       
            bmp_ind = np.random.randint(2**20)
            new_img_name = f'{src_img_name}_{bmp_ind}.bmp'
            new_img_path = osp.join(img_dir, new_img_name)
            cv2.imwrite(new_img_path, img[t:t+fix_h, l:l+fix_w, :])
        
            coco  = append_img_to_coco(coco, new_img_name, crop_tlbr, img_points, img_cats)
                
    return coco
    
def preprocess(model_dir, ratio, augment):
    labelme_files = [f for f in os.listdir(model_dir) if f.endswith('_labelme.json')]
    num_imgs = len(labelme_files)
    coco = init_coco()
    path_to_src_imgs = []
    imgs = []
    src_img_names = []
    imgs_points = []
    imgs_cats = []
    crop_label_flag = False
    
    for file in labelme_files:
        # open source image
        path_to_src_img = extract_imgs_from_labelme(model_dir, file)
        path_to_src_imgs.append(path_to_src_img)
        imgs.append(cv2.imread(path_to_src_img))
        img_name = osp.basename(path_to_src_img)
        src_img_name = img_name.split('.')[:-1]
        src_img_name = ''.join(src_img_name)
        src_img_names.append(src_img_name)
        
        # open labelme annotations 
        path_to_json = osp.join(model_dir, file)
        with open(path_to_json, 'r') as f:
            annots = json.load(f)
        ## make training dataset
        shapes = annots['shapes']
        # extract contours
#        img_points = [np.asarray(shape['points']).reshape((-1,2)) for shape in shapes]
        img_points = [get_points(shape) for shape in shapes]
        print(img_points)
        imgs_points.append(np.asarray(img_points))
        
        # extract labels
        labels = [shape['label'] for shape in shapes]
        imgs_cats.append(labels)
        for label in labels:
            if label in CROP_LABELS:
                crop_label_flag = True

    # populate COCO annotations
    coco['images'] = []
    coco['categories'] = []
    coco['annotations'] = []

    if crop_label_flag:
        coco = crops2coco(coco, imgs, src_img_names, imgs_points, imgs_cats, model_dir, ratio, augment)
    else:
        coco = objects2coco(coco, imgs, src_img_names, imgs_points, imgs_cats, model_dir, ratio, augment)

    train_coco_name = 'train_coco.json'
    train_coco_path = osp.join(model_dir, train_coco_name)
    with open(train_coco_path, 'w') as f:
        json.dump(coco, f)
    
    return path_to_src_imgs

def plot_contours(crop, cnts):
    img = crop.copy()
    color = np.random.randint(0, 256, 3).tolist()
    color = tuple(color)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
    return img

def plot_coco(coco, proj_dir):
    # leave only those images that are not duplicated
    # required for single crop case
    img_dir = osp.join(proj_dir, IMG_DIR_PREFIX)
    uniq_images = []
    uniq_image_names = []
    for img_ann in coco['images']:
        if img_ann['file_name'] not in uniq_image_names:
            uniq_images.append(img_ann)
            uniq_image_names.append(img_ann['file_name'])
            
    for img_ann in uniq_images:
        image_id = img_ann['id']
        path_to_img = osp.join(img_dir, img_ann['file_name'])
        img = cv2.imread(path_to_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cnts = [ann['segmentation'] for ann in coco['annotations'] if ann['image_id']==image_id]
        cnts = [np.asarray(cnt).reshape((-1,1,2)).astype(np.int32) for cnt in cnts]
        
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(plot_contours(img, cnts))
        plt.show()
        
def show(proj_dir):    
    train_coco_path = osp.join(proj_dir, TRAIN_COCO_NAME)
    with open(train_coco_path, 'r') as f:
        train_coco = json.load(f)
    plot_coco(train_coco, proj_dir)

if __name__ == "__main__":
    preprocess(model_dir, ratio, augment)
    