#!/bin/bash
DATA_PATH="/remote-home/share/KITTI/KITTI_fengyi"
DEPTH_ENCODER='unet'

python train.py --data_path=${DATA_PATH} --log_dir=./checkpoints --depth_encoder=${DEPTH_ENCODER} --num_epochs=40 --batch_size=16 --png