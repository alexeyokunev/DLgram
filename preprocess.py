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

def get_crop_ltwhs(imgs_points):
    imgs_whs = []
    for img_points in imgs_points:
        img_whs = []
        for cnt in img_points:
            min_x, min_y = np.min(cnt, axis=0)
            max_x, max_y = np.max(cnt, axis=0)
            w, h = round(max_x - min_x, 0), round(max_y - min_y, 0)
            img_whs.append([min_x, min_y, w, h])
        imgs_whs.append(np.asarray(img_whs))   
    return imgs_whs

def find_max_wh(imgs_whs):
    max_imgs_wh = [np.max(img_whs, axis=0)[2:] for img_whs in imgs_whs]
    max_imgs_wh = np.asarray(max_imgs_wh)
    print(max_imgs_wh)
    max_wh = np.max(max_imgs_wh, axis=0)
    return max_wh

def append_img_to_coco(coco, img_name, crop_tlbr, cnt, bbox, label):
    # append image name and parameters
    t, l, b, r = crop_tlbr.ravel()
    img_id = len(coco['images']) + 1
    new_img = {"id": img_id, 
               "file_name": img_name, 
               "width": int(r-l), 
               "height": int(b-t), 
               "date_captured": "", 
               "license": 1, 
               "coco_url": "", 
               "flickr_url": ""}
    coco['images'].append(new_img)
    
    # append category
    cats = coco['categories']
    labels = [cat['name'] for cat in cats]
    if label in labels:
        cat_id = labels.index(label) + 1
    else:
        cat_id = len(cats) + 1
        new_cat = {"id": cat_id, 
                   "name": label, 
                   "supercategory": ""
                  }
        coco['categories'].append(new_cat)
    
    # append annotation
    crop_lt = np.asarray([l, t])
    cnt = cnt.copy()
    cnt -= crop_lt
    cnt = cnt.astype(np.int32)
    bbox = bbox.copy()
    bbox[:2] -= crop_lt
    ann_id = len(coco['annotations']) + 1
    new_ann = { "id": ann_id, 
                "image_id": img_id, 
                "category_id": cat_id, 
                "iscrowd": 0, 
                "bbox": bbox.astype(np.int32).tolist(), 
                "area": cv2.contourArea(cnt.reshape((-1,1,2))), 
                "segmentation": [cnt.ravel().tolist()],
                "width": int(r-l), 
                "height": int(b-t)}
    coco['annotations'].append(new_ann)

    return coco

#model_dir = osp.join('../data', net)
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
        img_points = [np.asarray(shape['points']).reshape((-1,2)) for shape in shapes]
        imgs_points.append(np.asarray(img_points))
        
        # extract labels
        labels = [shape['label'] for shape in shapes]
        imgs_cats.append(labels)
        if 'crop' in labels:
            crop_label_flag = True
        
    imgs_crop_ltwhs = get_crop_ltwhs(imgs_points)
    # search for the largest shape size
    max_wh = find_max_wh(imgs_crop_ltwhs) * ratio
    max_w, max_h = max_wh.astype(np.int32)
    
    # crop images to image folder
    img_dir = osp.join(model_dir, 'images')
    if not osp.isdir(img_dir):
        os.makedirs(img_dir)
    
    # populate COCO annotations
    coco['images'] = []
    coco['categories'] = []
    coco['annotations'] = []
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
            
            # object at the center of crop
#            center = (bbox[:2] + bbox[2:]/2) 
#            l, t = center - max_wh/2    

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
                
                print(f'bbox {bbox}, crop {max_wh}, l={l:1,f}, t={t:1,f}, r={r:1,f}, b={b:1,f}')
                crop_tlbr = np.asarray([t,l,b,r]).astype(np.int32)
                crop_tlbr = np.clip(crop_tlbr.reshape((-1,2)), [0,0], img.shape[:2])
                t, l, b, r = crop_tlbr.ravel()
                bmp_ind = np.random.randint(2**20)
                new_img_name = f'{src_img_name}_{bmp_ind}.bmp'
                new_img_path = osp.join(img_dir, new_img_name)
                crop_w, crop_h = int(r-l), int(b-t)
                

                cv2.imwrite(new_img_path, img[t:t+crop_h, l:l+crop_w, :])
                
                # add padding
#                offset_h, offset_w = int((max_h - crop_h)/2), int((max_w - crop_w)/2)
#                padded = np.zeros((max_h+1, max_w+1, 3))
#                padded[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, :] = img[t:t+crop_h, l:l+crop_w, :]
#                bbox[:2] += np.asarray([offset_w, offset_h])
#                cnt += np.asarray([offset_w, offset_h]) 
#                cv2.imwrite(new_img_path, padded)
                
        
                coco  = append_img_to_coco(coco, new_img_name, crop_tlbr, cnt, bbox, label)

    train_coco_name = 'train_coco.json'
    train_coco_path = osp.join(model_dir, train_coco_name)
    with open(train_coco_path, 'w') as f:
        json.dump(coco, f)
    
    return path_to_src_imgs
        
if __name__ == "__main__":
    preprocess(model_dir, ratio, augment)
    