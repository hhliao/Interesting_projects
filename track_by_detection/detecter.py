# -*- coding : utf-8 -*-
# @Time     : 2023/8/2 - 10:53
# @Author   : rainbowliao
# @FileName : detecter.py
#

import torch
from ultralytics import YOLO


class_names = {3: 'car', 4: 'van', 5: 'truck', 8: 'bus'}


class Detecter:
    def __init__(self, model_path, conf=0.5, device='gpu'):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.conf = conf
        self.device = 'gpu' if device == 'gpu' and torch.cuda.is_available() else 'cpu'

    def predict(self, image):
        results = self.model.predict(image, stream=False, conf=self.conf)
        # print(f'predict = {results}')
        bboxes = results[0].boxes.xyxy.detach().cpu().numpy()
        clses = results[0].boxes.cls.detach().cpu().numpy()
        confidences = results[0].boxes.conf.detach().cpu().numpy()

        dets = []
        for bbox, cls, conf in zip(bboxes, clses, confidences):
            if cls not in class_names:
                continue
            dets.append({'bbox': list(bbox), 'score': conf, 'class': class_names[cls], 'class_id': cls})
        return dets


if __name__ == '__main__':
    import cv2
    import numpy as np
    import json

    model_path = './yolov8_weights/800_van2car_best.pt'
    image_file = './saved_imgs/1.jpg'
    det = Detecter(model_path)

    with open('./yolov8_weights/area.json', 'r', encoding='utf-8') as fd:
        area_json = json.load(fd)
    area = np.array(area_json['shapes'][0]['points'])
    x_min = int(min(area[:, 0]))
    x_max = int(max(area[:, 0]))
    y_min = int(min(area[:, 1]))
    y_max = int(max(area[:, 1]))
    # print(x_min)

    image = cv2.imread(image_file)
    croped_img = image[y_min: y_max, x_min: x_max, :]

    out = det.predict(croped_img)
    print(out)
