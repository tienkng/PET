#!/bin/bash

python ./inference4onnx.py \
    --onnx_model /home/tiennv/FPT/yolov9/PET/weight/model.onnx \
    --img_folder /home/tiennv/FPT/yolov9/APGCC/black_guy/test1_time/images \
    --vis_dir /home/tiennv/FPT/yolov9/PET/onnx_img_inference_output \
    --input_height 512 \
    --input_width 1024 \
    --device cuda