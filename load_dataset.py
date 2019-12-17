import json
import os
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd


if __name__ == '__main__':
    os.makedirs('data/train_images')
    os.makedirs('data/test_images')

    with open('dataset_train.json', 'r') as f:
        train_dataset = json.load(f)

    local_train_dataset = []
    image_dir = 'data/train_images/'

    for idx, note in enumerate(train_dataset):
        image_name = str(idx) + '.jpg'
        local_train_dataset.append({
            'Id': note['image'],
            'class': note['class'],
            'image_name': image_name
        })
        data = urlopen(note['image']).read()
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite(image_dir + image_name, image)

    df = pd.DataFrame(local_train_dataset)
    df.to_csv('data/train_dataset.csv', index=False)

    df = pd.read_csv('dataset_test_ids.csv')
    image_dir = 'data/test_images/'
    image_names = []
    for idx, row in df.iterrows():
        image_name = str(idx) + '.jpg'
        image_names.append(image_name)
        data = urlopen(row['Id']).read()
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite(image_dir + image_name, image)

    df['image_name'] = image_names
    df.to_csv('data/test_dataset.csv')
