import json
import matplotlib.pyplot as plt
import numpy as np
from labelme import utils
import uuid
import PIL.Image
import PIL.ImageDraw
import math
import os
import glob
import pandas as pd
import cv2
import argparse


def shape_to_mask(
        img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
    draw.rectangle(xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shape, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []

    points = shape["points"]
    label = shape["label"]
    group_id = shape.get("group_id")
    if group_id is None:
        group_id = uuid.uuid1()
    shape_type = shape.get("shape_type", None)
    cls_name = label
    instance = (cls_name, group_id)
    if instance not in instances:
        instances.append(instance)
    ins_id = instances.index(instance) + 1
    cls_id = label_name_to_value[cls_name]
    mask = shape_to_mask(img_shape[:2], points, shape_type)
    cls[mask] = cls_id
    ins[mask] = ins_id
    return cls, ins

dirnames = os.listdir(r'E:\bone_wsi\100x\100x_label\data')
def main(args):
    data_root = args.data_root
    dirnames = os.listdir(data_root)
    for dirname in dirnames:
        print(dirname)
        filenames=glob.glob(os.path.join(data_root,dirname,'*.json'))
        for filename in filenames:
            filepath = filename
            ax = filepath.split('\\')[-1]
            bx = ax.split('.')[0]
            data = json.load(open(filepath, 'rb'))
            img = utils.img_b64_to_arr(data['imageData'])
            a = 0
            for shape in data['shapes']:
                label_name_to_value = {"_background_": 0} #将背景排除
                mask = []
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
                lbl, lbl_names = shapes_to_label(img.shape, shape, label_name_to_value)
                mask.append((lbl).astype(np.uint8))
                mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])
                mask2 = np.ones_like(img)
                #mask2 = np.ones([400,400,3])
                for i in range(mask2.shape[2]):
                    mask2[:, :, i] = mask.squeeze()
                pic = img * mask2

                points = shape["points"]
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask = PIL.Image.fromarray(mask)
                draw = PIL.ImageDraw.Draw(mask)
                xy = [tuple(point) for point in points]
                assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
                x0 = int(xy[0][0])
                x1 = int(xy[1][0])
                y0 = int(xy[0][1])
                y1 = int(xy[1][1])
                print(x0,x1,y0,y1)
                x_min = min(x0,x1)
                x_max = max(x0,x1)
                y_min = min(y0,y1)
                y_max = max(y0,y1)
                if x_min < 0:
                    x_min = 0
                if x_max > img.shape[1]:
                    x_max = img.shape[1]
                if y_min < 0:
                    y_min = 0
                if y_max > img.shape[0]:
                    y_max = img.shape[0]
                pic = pic[y_min:y_max, x_min:x_max, :]
                new_im = PIL.Image.fromarray(pic)
                name = os.path.join(args.save_root,label_name,bx + shape["label"] + '_' + str(a) + '.tif')
                name_path = os.path.join(args.save_root,label_name)
                isExists = os.path.exists(name_path)
                if not isExists:
                    os.makedirs(name_path)
                new_im.save(name, quality=100,subsampling=0)
                a += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='')
    parser.add_argument('--save_root', default='')
    opt = parser.parse_args()
    main(opt)