#!/bin/bash

gpu_id=${1}
logdir=./train_log/coco_resnet18

python main/detection/faster_rcnn/train_faster_rcnn.py --logdir=${logdir} --gpus=${gpu_id} \
            --config \
                GENERAL.WORKER_NUM=4 \
                BACKBONE.NAME='resnet18_v1b' \
                DATASET.TYPE="coco" \
                TRAIN.SAVE_INTERVAL=9999 \
                TRAIN.EVAL_INTERVAL=20
