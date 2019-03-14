# -*- coding: utf-8 -*-

from utils.attr_dict import AttrDict


config = AttrDict()
_C = config  # short alias to avoid coding

_C.GENERAL.LOG_INTERVAL = 50
_C.GENERAL.WORKER_NUM = 8
_C.GENERAL.FP16 = True
_C.GENERAL.FP16_RESCALE_FACTOR = 128

_C.BACKBONE.NAME = 'resnet50_v1b'

_C.TRAIN.MODE_MIXUP = False
_C.TRAIN.START_EPOCH = 1
# LR_SCHEDULE means equivalent steps when the total batch size is 8.
# When the total bs!=8, the actual iterations to decrease learning rate, and
# the base learning rate are computed from BASE_LR and LR_SCHEDULE.
# Therefore, there is *no need* to modify the config if you only change the number of GPUs.
_C.TRAIN.LR_SCHEDULE = [120000, 160000, 180000]  # "1x" schedule in detectron
# _C.TRAIN.LR_SCHEDULE = [240000, 320000, 360000]      # "2x" schedule in detectron
# Longer schedules for from-scratch training (https://arxiv.org/abs/1811.08883):
# _C.TRAIN.LR_SCHEDULE = [960000, 1040000, 1080000]    # "6x" schedule in detectron
# _C.TRAIN.LR_SCHEDULE = [1500000, 1580000, 1620000]   # "9x" schedule in detectron
_C.TRAIN.NO_MIXUP_SCHEDULE = 160000
_C.TRAIN.BASE_LR = 0.00125
_C.TRAIN.LR_DECAY_FACTOR = 0.1
_C.TRAIN.LR_WARMUP = 8000
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.STEPS_PER_EPOCH = 500  # None to use the default epoch.
_C.TRAIN.SAVE_INTERVAL = 9999  # Only keep best mAP params.
_C.TRAIN.EVAL_INTERVAL = 20
_C.TRAIN.RANDOM_SEED = 233

_C.DATASET.TYPE = 'coco'  # 'coco' or 'voc'

_C.freeze()


def update_configs(num_gpus):
    _C.freeze(False)

    _C.AUTO.NUM_GPUS = num_gpus
    if _C.DATASET.TYPE == 'voc':
        if _C.TRAIN.STEPS_PER_EPOCH:
            raise NotImplementedError()
        _C.AUTO.END_EPOCH = 20
        _C.AUTO.LR_DECAY_EPOCH = [14, 20]
        _C.TRAIN.BASE_LR = 0.001
        _C.TRAIN.WEIGHT_DECAY = 5e-4
        _C.AUTO.NO_MIXUP_EPOCH = 5
    elif _C.DATASET.TYPE == 'coco':
        if _C.TRAIN.STEPS_PER_EPOCH:
            _C.AUTO.END_EPOCH = _C.TRAIN.LR_SCHEDULE[-1] * 8 \
                    // num_gpus // _C.TRAIN.STEPS_PER_EPOCH
            _C.AUTO.LR_DECAY_EPOCH = [lr * 8 // num_gpus // _C.TRAIN.STEPS_PER_EPOCH \
                    for lr in _C.TRAIN.LR_SCHEDULE[:-1]]
            _C.AUTO.NO_MIXUP_EPOCH = _C.TRAIN.NO_MIXUP_SCHEDULE * 8 \
                // num_gpus // _C.TRAIN.STEPS_PER_EPOCH
        else:
            _C.AUTO.END_EPOCH = 26
            _C.AUTO.LR_DECAY_EPOCH = [17, 23]
            _C.AUTO.NO_MIXUP_EPOCH = 5
        if num_gpus == 1:
            _C.TRAIN.LR_WARMUP = -1
        else:
            _C.TRAIN.BASE_LR *= num_gpus
            _C.TRAIN.LR_WARMUP /= num_gpus

    _C.freeze()
