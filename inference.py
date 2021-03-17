import argparse
import json
import os.path as osp
import cv2
import numpy as np

from predict import predict

def calc_stat(res_json):
    stat = {}
    for shape in res_json['shapes']:
        key = shape['label']
        if key in stat.keys():
            stat[key] += 1
        else:
            stat[key] = 1
    return stat

def plot_preds(img, shapes):
    res_img = img.copy()
    label_colors = {}
    for shape in shapes:
        lbl = shape['label']
        if lbl not in label_colors.keys():
            label_colors[lbl] = tuple([int(np.random.randint(256)) for _ in range(3)])
        cnt = np.asarray(shape['points'])
        cnts = [cnt.reshape((-1,1,2))]
        cv2.drawContours(res_img, cnts, -1, label_colors[lbl], res_img.shape[0]//200)
    return res_img

def inference(path, net, thresh, merge):
    test_dir = osp.dirname(path)
    # read net parameters
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    
    nets = cfg['net']
    
    path_to_cfg = nets[net]['path_to_config']
    path_to_ckpt = nets[net]['path_to_checkpoint']    
    classes = nets[net]['classes']
    # read test image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # inference
    res_json = predict(path_to_cfg, path_to_ckpt, classes, img, thresh, merge)
    # save annotations
    img_name = osp.basename(path).split('.')
    img_base_name = img_name[:-1]
    img_base_name = ''.join(img_base_name)
    img_name_ext = img_name[-1]
    dst_json_name = img_base_name + '.json'
    dst_json_path = osp.join(test_dir, dst_json_name)
    with open(dst_json_path, 'w') as f:
        json.dump(res_json, f)
    # save statistics
    stat = calc_stat(res_json)
    dst_stat_name = img_base_name + '.csv'
    dst_stat_path = osp.join(test_dir, dst_stat_name)
    with open(dst_stat_path, 'w') as f:
        json.dump(stat, f)
    # plot masks
    img = cv2.imread(path)
    res_img = plot_preds(img, res_json['shapes'])
    res_img_name = img_base_name + '_inf.' + img_name_ext
    res_img_path = osp.join(test_dir, res_img_name)
    cv2.imwrite(res_img_path, res_img) 
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="image name")
    parser.add_argument("-n", "--net", default='stock', help="unique name for network model, \"sph\" for spheres, \"stm\" for stm etc")
    parser.add_argument("-t", "--thresh", default=0.3, help="prediction's threshold probability")
    parser.add_argument("-e", "--epochs", default=100, help="prediction's threshold probability")
    parser.add_argument("-r", "--ratio", default='1.5', help="ration of crop size to bounding box size")
    parser.add_argument("-a", "--augment", default='1', help="number of augmented crops")
    parser.add_argument("-m", "--merge", default='0', help="merge classes")


    args = parser.parse_args()
    args.thresh = float(args.thresh)
    args.epochs = int(args.epochs)
    args.ratio = float(args.ratio)
    args.augment = int(args.augment)
    args.merge = int(args.merge)
    
    inference(args.path, args.net, args.thresh, args.merge)