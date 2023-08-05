# -*- coding : utf-8 -*-
# @Time     : 2023/8/4 - 00:20
# @Author   : rainbowliao
# @FileName : kalman_tracker_by_center.py
#
import os
import cv2
import numpy as np
import json
import argparse
from motrackers import CentroidKF_Tracker
from motrackers.utils import draw_tracks
from detecter import Detecter
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, LineString


def get_track_area_and_line(track_area_file):
    # init tracking region
    with open(track_area_file, 'r', encoding='utf-8') as fd:
        area_json = json.load(fd)

    shapes = area_json['shapes']
    if len(shapes) != 2:
        raise ValueError(f"{track_area_file} should be in two shapes")

    mask = []
    line = []
    for shape in shapes:
        if shape['shape_type'] == 'line':
            start_pt, end_pt = shape['points'][0],shape['points'][1]
            line.append(start_pt)
            line.append(end_pt)
        else:
            mask = np.array(shape['points']).astype(int)
    return {'mask': mask, 'line': line}


def is_cross_line(line, bbox):
    x, y, w, h = bbox
    pt1 = (x, y)
    pt2 = (x + w, y)
    pt3 = (x + w, y + h)
    pt4 = (x, y + h)
    rectange = Polygon([pt1, pt2, pt3, pt4])
    path = LineString(line)
    return path.intersects(rectange)


def main(args):
    # initialization
    video_path = args.video_path
    track_area_file = args.track_area
    sigma_l = args.sigma_l
    sigma_h = args.sigma_h
    t_min = args.t_min
    output_path = args.output_path

    if output_path is None:
        output_path = './output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize object detection model and tracker
    detecter = Detecter(args.det_model, conf=args.det_conf, device='gpu')
    tracker = CentroidKF_Tracker(max_lost=t_min)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"{video_path} is not open...")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    # initialize track area and line
    track_area_config = get_track_area_and_line(track_area_file)
    mask = np.zeros((height, width), dtype=np.uint8)
    no_mask = np.ones((height, width), dtype=np.uint8)
    mask = Image.fromarray(mask)
    no_mask = Image.fromarray(no_mask)
    draw = ImageDraw.Draw(mask)
    draw_no = ImageDraw.Draw(no_mask)
    xy = [tuple(point) for point in track_area_config['mask']]
    draw.polygon(xy=xy, outline=1, fill=1)
    draw_no.polygon(xy=xy, outline=1, fill=0)
    mask = np.asarray(mask)
    no_mask = np.asarray(no_mask)

    out_file = os.path.join(output_path, 'tracking_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # cv2.VideoWriter_fourcc('M','P','4','V')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    frame_id = 1
    count_results = {'up': 0, 'down': 0, 'img_w': width, 'img_h': height, 'line': track_area_config['line']}

    pre_tracks = {}
    while True:
        ok, frame_img = capture.read()
        print(f'grabing frame {frame_id}')
        if not ok:
            break

        image = cv2.bitwise_and(frame_img, frame_img, mask=mask)

        detections = detecter.predict(image)

        bboxes, confidences, class_ids = [], [], []
        for det in detections:
            bbox = det['bbox']
            conf = det['score']
            class_id = det['class_id']
            x_min = min(bbox[0], bbox[2])
            x_max = max(bbox[0], bbox[2])
            y_min = min(bbox[1], bbox[3])
            y_max = max(bbox[1], bbox[3])
            w = x_max - x_min
            h = y_max - y_min

            bboxes.append([x_min, y_min, w, h])
            confidences.append(conf)
            class_ids.append(class_id)

        bboxes = np.array(bboxes).astype('int')
        confidences = np.array(confidences)
        class_ids = np.array(class_ids).astype('int')

        # update tracker
        tracks = tracker.update(bboxes, confidences, class_ids)
        # todo:
        # 1. 记录每个id的历史位置与当前位置，用于更新上下
        # 2. 后续其他的内容
        for track in tracks:
            frame_num, id, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            c_bbox = [bb_left, bb_top, bb_width, bb_height]
            if frame_num == 1:
                # 第一帧不参与计数，因为无法判断方向
                pre_tracks[id] = {'xywh': c_bbox, 'is_counted': False}
                continue
            # print(f"bb_left = {bb_left}, bb_top = {bb_top}, bb_width = {bb_width}, bb_height = {bb_height}")

            if is_cross_line(track_area_config['line'], c_bbox):
                # 与计数线有交汇了，
                if id not in pre_tracks:
                    # 如果目标第一次出现，则不参与计数
                    pre_tracks[id] = {'xywh': c_bbox, 'is_counted': False}
                    continue
                elif pre_tracks[id]['is_counted']:
                    pre_tracks[id]['xywh'] = c_bbox
                    # 如果已经被计数了，直接跳过
                    continue
                else:
                    pre_c_bbox = pre_tracks[id]['xywh']
                    pre_y = pre_c_bbox[1] + pre_c_bbox[3] // 2
                    c_y = c_bbox[1] + c_bbox[3] // 2
                    if c_y < pre_y:
                        count_results['up'] += 1
                    else:
                        count_results['down'] += 1
                    pre_tracks[id]['is_counted'] = True
                    pre_tracks[id]['xywh'] = c_bbox

        # dram result
        vis_image = image.copy()
        for bbox in bboxes:
            x_min, y_min, w, h = bbox
            cv2.rectangle(vis_image, (x_min, y_min), (x_min+w, y_min+h), (0, 0, 255), 3)

        vis_image = draw_tracks(vis_image, tracks)

        counting_text = 'up:' + str(count_results['up']) + ' down:' + str(count_results['down'])
        pt_count = (int(0.5 * count_results['img_w']), int(0.8 * count_results['img_h']))
        cv2.putText(vis_image, counting_text, pt_count, cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 3)

        line_start, line_end = count_results['line']
        line_start = (int(line_start[0]), int(line_start[1]))
        line_end = (int(line_end[0]), int(line_end[1]))
        cv2.line(vis_image, line_start, line_end, (0, 255, 0), thickness=5)

        frame_img = vis_image * mask[..., None] + frame_img * no_mask[..., None]
        writer.write(frame_img)
        write_file = os.path.join(output_path, str(frame_id) + '_vis.jpg')
        cv2.imwrite(write_file, frame_img)

        frame_id += 1

    capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IOU Tracker script")
    parser.add_argument('-video', '--video_path', type=str, default='', required=True,
                        help="video with format '/path/to/video.mp4' ")
    parser.add_argument('-area', '--track_area', type=str, required=True,
                        help='iou threshold for best match')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help="output path to store the tracking results ")
    parser.add_argument('-sl', '--sigma_l', type=float, default=0,
                        help="low detection threshold")
    parser.add_argument('-sh', '--sigma_h', type=float, default=0.5,
                        help="high detection threshold")
    parser.add_argument('-tm', '--t_min', type=float, default=10,
                        help="minimum track length")
    parser.add_argument('-dt_model', '--det_model', type=str,
                        help="detection model path")
    parser.add_argument('-det_conf', '--det_conf', type=float, default=0.5,
                        help="detection confidence")

    args = parser.parse_args()
    main(args)