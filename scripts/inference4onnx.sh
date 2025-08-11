#!/bin/bash

python ./inference4onnx.py \
    --onnx_model /home/tiennv/FPT/yolov9/counting_people/PET/PET/weight/model_simp_clean.onnx \
    --img_folder /home/tiennv/FPT/yolov9/PET/data4render/images \
    --vis_dir /home/tiennv/FPT/yolov9/counting_people/PET/PET/onnx_img_inference_output \
    --input_height 512 \
    --input_width 1024 \
    --device cuda
