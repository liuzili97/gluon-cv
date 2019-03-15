"""Train Faster-RCNN end to end."""
import argparse
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import time
import pprint
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet import profiler

from utils import logger
from utils.common import sec_to_time
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy
from configs.faster_rcnn_config import update_configs, config as cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN networks e2e.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--logdir', type=str, default='train_log',
                        help='Saving parameter dir')
    parser.add_argument('--load', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--profile', action='store_true', help="Add mx profile.")
    parser.add_argument('--config', nargs='+',
                        help="A list of KEY=VALUE to overwrite those defined in config.py")
    args = parser.parse_args()

    if args.config:
        cfg.update_args(args.config)
    num_gpus = len(args.gpus.split(','))
    update_configs(num_gpus)

    return args


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        # pred_label = mx.nd.argmax(rpn_cls_logits, axis=1, keepdims=True)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, os.path.join(args.logdir, 'eval'), cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if cfg.TRAIN.MODE_MIXUP:
        from gluoncv.data.mixup import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    dtype = 'float16' if cfg.GENERAL.FP16 else 'float32'
    train_dataset = train_dataset.transform(FasterRCNNDefaultTrainTransform(net.short, net.max_size, net, dtype=dtype))
    val_dataset = val_dataset.transform(FasterRCNNDefaultValTransform(net.short, net.max_size, dtype=dtype))
    train_loader = mx.gluon.data.DataLoader(
        train_dataset, batch_size, True,
        batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers, thread_pool=True)
    val_loader = mx.gluon.data.DataLoader(
        val_dataset, batch_size, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers, thread_pool=True)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_save_prefix = os.path.join(prefix, 'best')
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
                    epoch, current_map, best_map, '{:s}.params'.format(best_save_prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}.params'.format(best_save_prefix))
        with open(best_save_prefix + '_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        save_param_filename = os.path.join(
            '{:s}'.format(prefix), '{:05d}_{:.4f}.params'.format(epoch, current_map))
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, save_param_filename))
        net.save_parameters(save_param_filename)


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    tbar = tqdm(val_data)
    tbar.set_description_str("[ EVAL  ]")
    for batch in tbar:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    rescale_factor = float(cfg.GENERAL.FP16_RESCALE_FACTOR) if cfg.GENERAL.FP16 else None
    trainer = gluon.Trainer(
        net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
        'sgd', {'learning_rate': cfg.TRAIN.BASE_LR, 'wd': cfg.TRAIN.WEIGHT_DECAY,
                'momentum': cfg.TRAIN.MOMENTUM, 'clip_gradient': 5,
                'multi_precision': cfg.GENERAL.FP16,
                'rescale_grad': 1.0 / cfg.GENERAL.FP16_RESCALE_FACTOR if cfg.GENERAL.FP16 else 1.0})

    # lr decay policy
    lr_steps = cfg.AUTO.LR_DECAY_EPOCH
    lr_warmup = float(cfg.TRAIN.LR_WARMUP)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False, weight=rescale_factor)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1/9., weight=rescale_factor)  # i.e. smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(weight=rescale_factor)
    rcnn_box_loss = mx.gluon.loss.HuberLoss(weight=rescale_factor)  # i.e. smoothl1
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'),]
    metrics2 = [RPNAccMetric(), RPNL1LossMetric(), RCNNAccMetric(), RCNNL1LossMetric()]

    logger.info("Trainable parameters: ------------------------------------------\n" + \
            pprint.pformat(net.collect_train_params().keys(), indent=1, width=100, compact=True))
    logger.info('LR Schedule [Epochs {} - {}].'.format(
        cfg.AUTO.LR_DECAY_EPOCH,
        [cfg.TRAIN.BASE_LR * cfg.TRAIN.LR_DECAY_FACTOR ** i for i in range(len(cfg.AUTO.LR_DECAY_EPOCH))]))
    logger.info('Start training from [Epoch {}] to [Epoch {}].'.format(
        cfg.TRAIN.START_EPOCH, cfg.AUTO.END_EPOCH))

    best_map = [0]
    steps_per_epoch = cfg.TRAIN.STEPS_PER_EPOCH if cfg.TRAIN.STEPS_PER_EPOCH else len(train_data)
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.AUTO.END_EPOCH + 1):
        mix_ratio = 1.0
        if cfg.TRAIN.MODE_MIXUP:
            # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
            train_data._dataset.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= (cfg.AUTO.END_EPOCH + 1) - cfg.AUTO.NO_MIXUP_EPOCH:
                train_data._dataset.set_mixup(None)
                mix_ratio = 1.0
        if lr_steps and epoch >= lr_steps[0]:
            while lr_steps and epoch >= lr_steps[0]:
                new_lr = trainer.learning_rate * cfg.TRAIN.LR_DECAY_FACTOR
                lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()

        tic = time.time()
        btic = time.time()
        if epoch == cfg.TRAIN.START_EPOCH or (epoch - 1) % cfg.TRAIN.EVAL_INTERVAL == 0:
            net.hybridize(static_alloc=True)
        base_lr = trainer.learning_rate
        tbar = tqdm(train_data, total=steps_per_epoch)
        tbar.set_description_str("[ TRAIN ]")
        for i, batch in enumerate(tbar):
            i += 1
            total_iter = (epoch - 1) * steps_per_epoch + i
            if total_iter <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(total_iter / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if total_iter % cfg.GENERAL.LOG_INTERVAL == 0:
                        tqdm.write('[Warm Up] Set learning rate to {}'.format(new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)  # Split data to 1 batch each device.
            batch_size = len(batch[0])

            losses = []
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            if args.profile and i == 10:
                profiler.set_config(profile_all=True, aggregate_stats=True,
                                    filename='profile_output.json')
                profiler.set_state('run')
            with autograd.record():
                for data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks in zip(*batch):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    cls_pred, box_pred, roi, samples, matches, rpn_score, rpn_box, anchors = net(data, gt_box)

                    # losses of rpn
                    if cfg.GENERAL.FP16:
                        rpn_score = rpn_score.astype('float32')
                        rpn_box = rpn_box.astype('float32')
                        rpn_cls_targets = rpn_cls_targets.astype('float32')
                        rpn_box_targets = rpn_box_targets.astype('float32')
                        rpn_box_masks = rpn_box_masks.astype('float32')
                    rpn_score = rpn_score.squeeze(axis=-1)
                    num_rpn_pos = (rpn_cls_targets >= 0).sum()
                    rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets, rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
                    rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets, rpn_box_masks) * rpn_box.size / num_rpn_pos
                    # rpn overall loss, use sum rather than average
                    rpn_loss = rpn_loss1 + rpn_loss2

                    # generate targets for rcnn
                    cls_targets, box_targets, box_masks = net.target_generator(roi, samples, matches, gt_label, gt_box)
                    # losses of rcnn
                    if cfg.GENERAL.FP16:
                        cls_pred = cls_pred.astype('float32')
                        box_pred = box_pred.astype('float32')
                        cls_targets = cls_targets.astype('float32')
                        box_targets = box_targets.astype('float32')
                        box_masks = box_masks.astype('float32')
                    num_rcnn_pos = (cls_targets >= 0).sum()
                    rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets, cls_targets >= 0) * cls_targets.size / cls_targets.shape[0] / num_rcnn_pos
                    rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / box_pred.shape[0] / num_rcnn_pos
                    rcnn_loss = rcnn_loss1 + rcnn_loss2
                    # overall losses
                    losses.append(rpn_loss.sum() * mix_ratio + rcnn_loss.sum() * mix_ratio)
                    metric_losses[0].append(rpn_loss1.sum() * mix_ratio)
                    metric_losses[1].append(rpn_loss2.sum() * mix_ratio)
                    metric_losses[2].append(rcnn_loss1.sum() * mix_ratio)
                    metric_losses[3].append(rcnn_loss2.sum() * mix_ratio)
                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets>=0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[cls_targets], [cls_pred]])
                    add_losses[3].append([[box_targets, box_masks], [box_pred]])
                autograd.backward(losses)

                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
            trainer.step(batch_size)
            if args.profile:
                mx.nd.waitall()
                profiler.set_state('stop')

            # update metrics
            if cfg.GENERAL.LOG_INTERVAL and total_iter % cfg.GENERAL.LOG_INTERVAL == 0:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                total_speed = cfg.GENERAL.LOG_INTERVAL * batch_size / (time.time() - btic)
                speed = total_speed / batch_size  # batch size rely on the gpu num.
                epoch_time_left = (steps_per_epoch - i + 1) / speed
                total_time_left = ((cfg.AUTO.END_EPOCH - epoch) * steps_per_epoch  - i + 1) / speed
                epoch_tl_h, epoch_tl_m, epoch_tl_s = sec_to_time(epoch_time_left)
                total_tl_h, total_tl_m, _ = sec_to_time(total_time_left)
                tqdm.write('[Epoch {}][Batch {}], {:.3f}/{:0>2}h{:0>2}m{:0>2}s/{:0>2}h{:0>2}m, {}'.format(
                    epoch, total_iter, total_speed, epoch_tl_h, epoch_tl_m, epoch_tl_s, total_tl_h, total_tl_m, msg))
                btic = time.time()

            if cfg.TRAIN.STEPS_PER_EPOCH and i >= cfg.TRAIN.STEPS_PER_EPOCH:
                break
        tbar.close()
        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}s, {}'.format(
            epoch, (time.time() - tic), msg))
        if epoch % cfg.TRAIN.EVAL_INTERVAL == 0:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, cfg.TRAIN.SAVE_INTERVAL, args.logdir)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(cfg.TRAIN.RANDOM_SEED)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('faster_rcnn', cfg.BACKBONE.NAME, cfg.DATASET.TYPE))
    time_str = time.strftime("%m%d_%H%M")
    args.logdir = os.path.join(args.logdir, "{}_{}".format(net_name, time_str))

    # set up logger
    logger.set_logger_dir(args.logdir, 'd')
    logger.info("Config: ------------------------------------------\n" + \
            pprint.pformat(cfg.to_dict(), indent=1, width=100, compact=True))

    net = get_model(net_name, pretrained_base=True,
                    dtype='float16' if cfg.GENERAL.FP16 else 'float32')
    if cfg.GENERAL.FP16:
        net.cast('float16')
    if args.load.strip():
        net.load_parameters(args.load.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(cfg.DATASET.TYPE, args)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.batch_size, cfg.GENERAL.WORKER_NUM)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
