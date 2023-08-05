# -*- coding : utf-8 -*-
# @Time     : 2023/8/2 - 15:08
# @Author   : rainbowliao
# @FileName : iou_tracker.py
#
import os
import sys
import cv2
import json
import argparse
from time import time
import numpy as np
from util import iou
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


def show_tracker(image, trackers, count_results):
    color = {'car': (0, 0, 255), 'van': (0, 0, 255), 'truck': (255, 0, 255), 'bus': (255,0,0) }
    vis_image = image.copy()
    for track in trackers:
        bbox = track['bboxes'][-1]
        x_min = min(bbox[0], bbox[2])
        x_max = max(bbox[0], bbox[2])
        y_min = min(bbox[1], bbox[3])
        y_max = max(bbox[1], bbox[3])

        pt1 = (int(x_min), int(y_min))
        pt2 = (int(x_max), int(y_min))
        pt3 = (int(x_max), int(y_max))
        pt4 = (int(x_min), int(y_max))

        center = (pt1[0], (pt1[1] + pt3[1]) // 2)
        track_id = track['track_id']
        class_name = track['class_name']
        cv2.rectangle(vis_image, pt1, pt3, color[class_name], thickness=3)
        cv2.putText(vis_image, str(track_id) +':' + class_name, center,
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)

        counting_text = 'up:' + str(count_results['up']) + ' down:' + str(count_results['down'])
        pt_count = (int(0.5*count_results['img_w']), int(0.8*count_results['img_h']))
        cv2.putText(vis_image, counting_text, pt_count, cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 0), 3)

        line_start, line_end = count_results['line']
        line_start = (int(line_start[0]), int(line_start[1]))
        line_end = (int(line_end[0]), int(line_end[1]))
        cv2.line(vis_image, line_start, line_end, (0, 255, 0), thickness=5)
    return vis_image


def is_cross_line(line, tracker):
    bbox = tracker['bboxes'][-1]
    x_min = min(bbox[0], bbox[2])
    x_max = max(bbox[0], bbox[2])
    y_min = min(bbox[1], bbox[3])
    y_max = max(bbox[1], bbox[3])

    pt1 = (int(x_min), int(y_min))
    pt2 = (int(x_max), int(y_min))
    pt3 = (int(x_max), int(y_max))
    pt4 = (int(x_min), int(y_max))
    rectange = Polygon([pt1, pt2, pt3, pt4])
    path = LineString(line)
    return path.intersects(rectange)


def main(args):
    # initialize
    sigma_l = args.sigma_l
    sigma_h = args.sigma_h
    sigma_iou = args.sigma_iou
    t_min = args.t_min
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # initialize object detection model
    detecter = Detecter(args.det_model, conf=args.det_conf, device='gpu')

    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        print(f"{args.video_path} is not open...")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    # initialize track area and line
    track_area_config = get_track_area_and_line(args.track_area)
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
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("no_mask.png", no_mask)


    out_file = os.path.join(output_path, 'tracking_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'MPEG') #cv2.VideoWriter_fourcc('M','P','4','V')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    tracks_active = []
    tracks_finished = []
    frame_id = 1
    track_idx = 1

    count_results = {'up': 0, 'down': 0, 'img_w': width, 'img_h': height, 'line': track_area_config['line']}
    while True:
        ok, frame = capture.read()
        print(f'grabing frame {frame_id}')
        if not ok:
            # 视频读取完毕，终止所有跟踪器
            tracks_finished += [track for track in tracks_active
                                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]
            break

        # 提取待检测区域
        image = cv2.bitwise_and(frame, frame, mask=mask)

        # 物体检测
        detection = detecter.predict(image)
        dets = [det for det in detection if det['score'] >= sigma_l]

        updated_tracks = []
        for track in tracks_active:
            if len(dets) > 0:
                # 根据最大IOU更新跟踪器
                best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x['bbox']))
                if iou(track['bboxes'][-1], best_match['bbox']) > sigma_iou:
                    track['bboxes'].append(best_match['bbox'])
                    track['max_score'] = max(track['max_score'], best_match['score'])
                    track['frame_ids'].append(frame_id)

                    updated_tracks.append(track)
                    del dets[dets.index(best_match)]

            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                if track['max_score'] > sigma_h and len(track['bboxes']) > t_min:
                    tracks_finished.append(track)

        # 如有未分配的目标，创建新的跟踪器
        new_tracks = []
        for det in dets:
            new_track = {
                'bboxes': [det['bbox']], # 跟踪目标的矩形框
                'max_score': det['score'], # 跟踪目标的最大score
                'start_frame': frame_id,  # 目标出现的 帧id
                'frame_ids': [frame_id],  # 目标出现的所有帧id
                'track_id': track_idx,    # 跟踪标号
                'class_name': det['class'], # 类别
                'is_counted': False       # 是否已经计数
            }
            track_idx += 1
            new_tracks.append(new_track)

        tracks_active = updated_tracks + new_tracks

        # 判断物体是否过线，根据跟踪起始位置y坐标判断方向
        for tracker in tracks_active:
            if is_cross_line(track_area_config['line'], tracker) and not tracker['is_counted']:
                tracker['is_counted'] = True
                bbox_end = tracker['bboxes'][-1]
                bbox_start = tracker['bboxes'][0]
                center_end_y = (bbox_end[1] + bbox_end[3]) / 2
                center_start_y = (bbox_start[1] + bbox_start[3]) / 2
                if center_end_y > center_start_y:
                    count_results['down'] += 1
                else:
                    count_results['up'] += 1

        # 在图片中显示跟踪结果
        vis_image = show_tracker(image, tracks_active, count_results)
        # 将未参与检测的图片区域拼接回来
        frame = vis_image * mask[..., None] + frame * no_mask[..., None]
        writer.write(frame)
        # 写入图片
        # write_file = os.path.join(output_path, str(frame_id) + '_vis.jpg')
        # cv2.imwrite(write_file, frame)

        frame_id += 1

    capture.release()
    writer.release()


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
    parser.add_argument('-si', '--sigma_iou', type=float, default=0.5,
                        help="intersection-over-union threshold")
    parser.add_argument('-tm', '--t_min', type=float, default=2,
                        help="minimum track length")
    parser.add_argument('-dt_model', '--det_model', type=str,
                        help="detection model path")
    parser.add_argument('-det_conf', '--det_conf', type=float, default=0.5,
                        help="detection confidence")

    args = parser.parse_args()
    main(args)
