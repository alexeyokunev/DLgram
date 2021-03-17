import cv2
import PIL.Image
import numpy as np
import json
import os.path as osp
import pickle

import base64
import io

from mmcv import Config
from mmdet.apis import init_detector
from mmdet.apis import inference_detector

def infer(path_to_cfg, path_to_ckpt, clses, img):
    
    cfg = Config.fromfile(path_to_cfg)
    
    # set image size for inference
    img_h, img_w = img.shape[:2]
    img_h, img_w = int(img_h), int(img_w)

    cfg.data.test.pipeline[1]['img_scale'] = img_w, img_h
    print(f'image size {img_w}x{img_h} set for inference')
    
    print('building model')
    model = init_detector(cfg, path_to_ckpt)
    model.CLASSES = clses
    print('model initialized')
    
    return inference_detector(model, img) 

def img_arr_to_b64(img_arr):
    """taken from https://raw.githubusercontent.com/wkentaro/labelme/master/labelme/utils/image.py"""
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format="PNG")
    img_bin = f.getvalue()
    if hasattr(base64, "encodebytes"):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:,4])
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    return pick


def get_shapes(bboxes, segms, clses, thresh):
    print(f'num of objects above threshold {thresh}')
    shapes = []
    for cls, bbs, sgms in zip(clses, bboxes, segms):
        sgms = np.asarray(sgms)
        inds = np.where(bbs[:,4]>thresh)[0]
        print(f'class {cls}, bboxes {bbs[inds].shape}, masks {len(sgms[inds])}')
        for ind in inds:
            shape = {}
            cnts, _ = cv2.findContours(sgms[ind].astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key = lambda cnt: cv2.contourArea(cnt), reverse=True) 
            if len(cnts):
                cnt = np.asarray(cnts[0]).reshape((-1,2))
                # remove artifacts to make contour hull
                #cnt = cv2.convexHull(cnt)
                # remove excessive points
                epsilon = 0.01*cv2.arcLength(cnt,True)
                cnt = cv2.approxPolyDP(cnt,epsilon,True)
                
                shape = dict(label = cls,
                             line_color = 'null',
                             fill_color = 'null',
                             points = [p.ravel().tolist() for p in cnt]
                            )
                shapes.append(shape)
    stat = {}
    for shape in shapes:
        key = shape['label']
        if key in stat.keys():
            stat[key] += 1
        else:
            stat[key] = 1
    print(stat)
    return shapes

def mmdet2labelme(bboxes, segms, clses, img, thresh):
    res_json = {}
    res_json['version'] = '4.5.6'
    res_json['flags'] = {}
    res_json['imagePath'] = ''
    res_json['imageHeight'] = img.shape[0]
    res_json['imageWidth'] = img.shape[1]
    res_json['imageData'] = img_arr_to_b64(img).decode('utf-8')
    res_json['shapes'] = get_shapes(bboxes, segms, clses, thresh)  
    return res_json


def predict(path_to_cfg, path_to_ckpt, clses, img, thresh, merge):
    
    result = infer(path_to_cfg, path_to_ckpt, clses, img)
    bboxes, segms = result
    
    if merge:
        bboxes = np.vstack(bboxes)
        merged_segms = []
        for segm in segms:
            merged_segms += segm
        idx = non_max_suppression_fast(bboxes, overlapThresh=0.3)
        bboxes = [bboxes[idx]]
        segms = [segm for i, segm in enumerate(merged_segms) if i in idx]
        segms = [segms]
        print(f'num of merged bboxes {bboxes[0].shape[0]} segms {len(segms[0])}')
        clses = ['merged']
        
    result = bboxes, segms
#    with open('../data/tmp/res.pickle', 'wb') as f:
#        pickle.dump(result, f)
    
    return mmdet2labelme(bboxes, segms, clses, img, thresh)

if __name__ == '__main__':
    predict(path_to_cfg, path_to_ckpt, clses, img, thresh, merge)    

