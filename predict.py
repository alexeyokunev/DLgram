import cv2
import PIL.Image
import numpy as np
import json
import os.path as osp

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


def get_shapes(result, clses, thresh):
    bboxes, segms = result
    shapes = []
    for cls, bbs, sgms in zip(clses, bboxes, segms):
        sgms = np.asarray(sgms)
        inds = np.where(bbs[:,4]>thresh)[0]
        print(cls, bbs[inds].shape, len(sgms[inds]))
        for ind in inds:
            shape = {}
            cnts, _ = cv2.findContours(sgms[ind].astype(np.uint8),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key = lambda cnt: cv2.contourArea(cnt), reverse=True) 
            if len(cnts):
                cnt = np.asarray(cnts[0]).reshape((-1,2))
                shape = dict(label = cls,
                             line_color = 'null',
                             fill_color = 'null',
                             points = [p.ravel().tolist() for p in cnt]
                            )
                shapes.append(shape)
    return shapes

def mmdet2labelme(result, clses, img, thresh):
    res_json = {}
    res_json['version'] = '4.5.6'
    res_json['flags'] = {}
    res_json['imagePath'] = ''
    res_json['imageHeight'] = img.shape[0]
    res_json['imageWidth'] = img.shape[1]
    res_json['imageData'] = img_arr_to_b64(img).decode('utf-8')
    res_json['shapes'] = get_shapes(result, clses, thresh)  
    return res_json

def predict(path_to_cfg, path_to_ckpt, clses, img, thresh):
    
    result = infer(path_to_cfg, path_to_ckpt, clses, img)
    return mmdet2labelme(result, clses, img, thresh)

if __name__ == '__main__':
    predict(path_to_cfg, path_to_ckpt, img, thresh)    

