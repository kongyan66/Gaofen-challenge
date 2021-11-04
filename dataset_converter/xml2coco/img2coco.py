'''
通过图片路径生成一个coco格式的空标签文件，用于测试并保存测试结果
'''
import os, csv
from tqdm import tqdm
import pandas as pd
import numpy as np
from csv2coco_train import Csv2CoCo
image_dir = "data/submit_test/images"

# newline 若文件不存在测创建文件
f = open('test.csv','w',newline='')

csv_writer = csv.writer(f)
images = os.listdir(image_dir) 
pbar = tqdm(images)
for img in pbar:
    pbar.set_description("beging to create test-image json annotation")
    name = os.path.splitext(img)[0]
    csv_writer.writerow([name,1,1,1,1,'A220'])
f.close()

total_csv_annotations = {}   # 保存每一个实例的标注信息（列表）
annotations = pd.read_csv('test.csv',header=None).values

for annotation in annotations:

    # key = annotation[0].split(os.sep)[-1]
    # key保存图片名列表
    key = annotation[0]
    value = np.array([annotation[1:]])
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
    else:
        total_csv_annotations[key] = value
# 按照键值划分数据
total_keys = list(total_csv_annotations.keys())

l2c_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
train_instance = l2c_train.to_coco(total_keys)
l2c_train.save_coco_json(train_instance, 'test.json')
