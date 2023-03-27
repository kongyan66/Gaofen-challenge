import os, csv
from tqdm import tqdm
import pandas as pd
import numpy as np
from .csv2coco import Csv2CoCo


def create_test_json(image_dir):
    # fiet-stage: create the csv file by test-images
    JSON_PATH = '/workspace'
    CSV_DIR = os.path.join(JSON_PATH,'test.csv')
    JSON_DIR = os.path.join(JSON_PATH, 'test.json')
    IMAGES_DIR = image_dir

    f = open(CSV_DIR,'w',newline='')
    csv_writer = csv.writer(f)
    images = os.listdir(IMAGES_DIR)
    pbar = tqdm(images)
    for img in pbar:
        if img.split('.')[1] not in ['csv', 'json']:
            pbar.set_description("beging to create test-image json annotation")
            name = os.path.splitext(img)[0]
            csv_writer.writerow([name,1,1,1,1,'A220'])
    f.close()
    # second-stage: create the json file by csv file
    total_csv_annotations = {}   
    annotations = pd.read_csv(CSV_DIR, header=None).values

    for annotation in annotations:

        # key = annotation[0].split(os.sep)[-1]
        key = annotation[0]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key], value), axis=0)
        else:
            total_csv_annotations[key] = value
       
    total_keys = list(total_csv_annotations.keys())
    l2c_train = Csv2CoCo(image_dir=IMAGES_DIR,total_annos=total_csv_annotations)
    train_instance = l2c_train.to_coco(total_keys)
    l2c_train.save_coco_json(train_instance, JSON_DIR)
if __name__=='__main__':
    create_test_json()