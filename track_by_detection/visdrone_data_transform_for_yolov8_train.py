# -*- coding : utf-8 -*-
# @Time     : 2023/7/30 - 11:02
# @Author   : rainbowliao
# @FileName : visdrone_data_transform.py
#
# visdrone data format
# <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

import os
import sys
import glob
import cv2


label_map = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor',
}


def transform_data(data_file, img_file):
    with open(data_file, "r", encoding="utf-8") as fd:
        lines = fd.readlines()

    image = cv2.imread(img_file)
    height, width, _ = image.shape

    results = []
    for line in lines:
        element = line.strip().split(',')
        bbox_left = float(element[0])
        bbox_top = float(element[1])
        bbox_width = float(element[2])
        bbox_height = float(element[3])
        object_category = int(element[5]) - 1

        if object_category not in label_map.keys():
            continue

        center_x = (bbox_left + bbox_width / 2) / width
        center_y = (bbox_top + bbox_height / 2) / height
        b_width = bbox_width / width
        b_height = bbox_height / height
        results.append({
            'label': object_category,
            'x': center_x,
            'y': center_y,
            'width': b_width,
            'height': b_height
        })

    return results


def main():
    in_path = sys.argv[1]

    img_path = os.path.join(in_path, 'images')
    ana_path = os.path.join(in_path, 'annotations')

    svd_path = os.path.join(in_path, 'labels')
    if not os.path.exists(svd_path):
        os.makedirs(svd_path)

    txt_files = sorted(glob.glob(os.path.join(ana_path, '*.txt')))
    for txt_file in txt_files:
        print(txt_file)
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        img_file = os.path.join(img_path, base_name + '.jpg')
        if not os.path.exists(img_file):
            print(f'{img_file} is not found')

        result = transform_data(txt_file, img_file)

        saved_file = os.path.join(svd_path, os.path.basename(txt_file))
        with open(saved_file, 'w', encoding='utf-8') as fw:
            for line in result:
                print(line)
                write_line = str(line['label']) + ' ' + str(line['x']) + ' ' + str(line['y']) + ' ' + \
                         str(line['width']) + ' ' + str(line['height']) + '\n'
                fw.write(write_line)
    print('Done...')


if __name__ == '__main__':
    main()

