#/bin/bash
set -e

# python iou_tracker.py \
python kalman_tracker_by_center.py \
  -video ./video/test_video.mp4 \
  -area ./yolov8_weights/1.json \
  -o ./output2 \
  -dt_model ./yolov8_weights/800_van2car_best.pt \
  -det_conf 0.5