#!/bin/bash

mkdir -p models/onnx
. .venv/bin/activate
python -m tf2onnx.convert --opset 16 \
    --tflite .venv/lib/python3.10/site-packages/mediapipe/modules/face_detection/face_detection_short_range.tflite \
    --output models/onnx/face_detection_short_range.onnx
# python -m tf2onnx.convert --opset 16 \
#     --tflite .venv/lib/python3.10/site-packages/mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite \
#     --output models/onnx/face_detection_full_range_sparse.onnx
python -m tf2onnx.convert --opset 16 \
    --tflite .venv/lib/python3.10/site-packages/mediapipe/modules/face_landmark/face_landmark.tflite \
    --output models/onnx/face_landmark.onnx
