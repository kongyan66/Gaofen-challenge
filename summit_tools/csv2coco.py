# -*- coding: utf-8 -*-
'''
@time: 2019/01/11 11:28
created for AI competion:http://sw.chreos.org/challenge/dataset/4
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
from IPython import embed

np.random.seed(41)

#0为背景
# wordname_37 = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other-airplane', 'Passenger Ship',
#                'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other-ship',
#                'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor', 'other-vehicle', 'Basketball Court',
#                'Tennis Court', 'Football Field', 'Baseball Field', 'Intersection', 'Roundabout', 'Bridge']
wordname_7 = ['A220','A330','A320/321','Boeing737-800','Boeing787','ARJ21','other']
classname_to_id = {name: i+1  for i, name in enumerate(wordname_7)}

class Csv2CoCo:

    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi,label,key)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'Klawens created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        img = cv2.imread(self.image_dir +'/'+str(path) + '.tif')
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = path
        image['file_name'] = str(path) + '.tif'
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape,label, path):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = path
        annotation['category_id'] = int(classname_to_id[str(label)])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a
   

if __name__ == '__main__':
    csv_file = "/home/z/code/s2anet/data/summit_test/test.csv"
    image_dir = "/home/z/code/s2anet/data/summit_test/images"
    saved_coco_path = "/home/z/code/s2anet/data/summit_test"
    # 整合csv格式标注文件
    total_csv_annotations = {}   # 保存每一个实例的标注信息（列表）
    annotations = pd.read_csv(csv_file,header=None).values

    for annotation in annotations:

        #key = annotation[0].split(os.sep)[-1]
        # key保存图片名列表
        key = annotation[0]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
        else:
            total_csv_annotations[key] = value
    # 按照键值划分数据
    total_keys = list(total_csv_annotations.keys())


    # # 把训练集转化为COCO的json格式
    l2c_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(total_keys)
    l2c_train.save_coco_json(train_instance, '%s/train.json'%saved_coco_path)
