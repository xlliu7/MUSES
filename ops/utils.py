import torch
import numpy as np
import time
import logging
from ruamel import yaml
import pdb

def get_logger(args, logfile):
    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # return logger

def get_configs(dataset, cfg_path):
    data = yaml.load(open(cfg_path, 'r'), Loader=yaml.RoundTripLoader)
    
    return data[dataset]

def get_and_save_args(parser):
    args = parser.parse_args()
    dataset = args.dataset
    cfg_path = args.cfg
    print('Using config ' + cfg_path)
    opt_dict = yaml.load(open(cfg_path, 'r'), Loader=yaml.RoundTripLoader)
    # pdb.set_trace()
    default_config = opt_dict[dataset]
    current_config = vars(args)
    for k, v in current_config.items():
        if k in default_config:
            if (v != default_config[k]) and (v is not None):
                print(f"Updating:  {k}: {default_config[k]} (default) ----> {v}")
                default_config[k] = v
    yaml.dump(default_config, open('./current_configs.yaml', 'w'), indent=4, Dumper=yaml.RoundTripDumper)
    return default_config


def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print(len(grad_in), len(grad_out))
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))
        print((grad_in[1].size()))
        print((grad_in[2].size()))

        # print((grad_out[0]))
        # print((grad_in[0]))

    return hook


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0
    else:
        return float(inter[1] - inter[0]) / float(union[1] - union[0])


def bbox_overlaps(proposals, gts):
  '''the numpy equivalent to the cython version of bbox_overlaps
  :param proposals(Nx2 array)
  :param gts(Kx2 array)
  :return n_prop * n_gt (NxK) array'''
  props_ext = proposals[:, :, None]
  gts_ext = gts[None, :, :]
  i_x0 = np.maximum(props_ext[:, 0, :], gts_ext[:, :, 0])
  i_x1 = np.minimum(props_ext[:, 1, :], gts_ext[:, :, 1])
  inters = np.clip(i_x1 - i_x0, 0, None)
  u_x0 = np.minimum(props_ext[:, 0, :], gts_ext[:, :, 0])
  u_x1 = np.maximum(props_ext[:, 1, :], gts_ext[:, :, 1])
  union = u_x1 - u_x0
  union[union==0] = 1
  overlaps = inters / union
  return overlaps


class ConfusionMatrix(object):
    def __init__(self, K, class_names=None, epsilon=1e-6):  # K is number of classes
        self.num_classes = K
        self.class_names = class_names if class_names else [str(x) for x in range(self.num_classes)]
        self.outdated = False
        self.reset()

    def reset(self):
        # declare a table matrix and zero it
        # one row for each class, column is predicted class
        self.cm = np.zeros([self.num_classes, self.num_classes], dtype='int64')
        # self.valids
        self.valids = np.zeros([self.num_classes], dtype='int64')
        # mean average precision, i.e., mean class accuracy
        self.mean_class_acc = 0
        # tp+fp = sum of each colume, how many samples are recognized as class k
        # tp+fn = sum of each row, how many samples are truely to class k
        self.tp_fp, self.tp_fn = [
            np.zeros([self.num_classes], dtype='int64') for x in range(2)]

        self.precs, self.recs = [
            np.zeros([self.num_classes], dtype='float64') for x in range(2)]
        self.acc = 0
        self.num_seen = 0
        if self.num_classes > 2:
            self.combined_prec = 0
            self.combined_rec = 0
            self.combined_acc = 0
        # self.

    def update(self, preds, targets):
        """
        preds are predicted classes
        """
        # convert cudalong tensor to long tensor
        # preds:  bz x 1
        for m in range(len(preds)):
            self.cm[targets[m]][preds[m]] += 1
        self.num_seen += len(preds)
        self.outdated = True

    def compute(self):
        # total = 0
        # print('confusion matrix field updated!')
        tp = np.diag(self.cm)
        self.tp_fp = self.cm.sum(0)  # sum up different rows, pred num of each class
        self.tp_fn = self.cm.sum(1)  # sum up different cols,  true num of each class
        self.precs = tp / (self.tp_fp + (self.tp_fp == 0))
        self.recs = tp / (self.tp_fn + (self.tp_fn == 0))
        # for i in range(len(self.precs)):
        #     if np.isnan(self.precs[i])
        self.acc = tp.sum() / (self.num_seen * 1.0)
        # self.mean_class_acc = self.valids.mean()
        if self.num_classes > 2:
            # combine all positive classes and recompute metrics
            new_tp = self.cm[1:, 1:].sum()
            new_tp_fp = self.tp_fp[1:].sum()
            new_tp_fn = self.tp_fn[1:].sum()
            self.combined_prec = new_tp / (new_tp_fp + (new_tp_fp == 0))
            self.combined_rec = new_tp / (new_tp_fn + (new_tp_fn == 0))
            self.combined_acc = new_tp * 1.0 / self.num_seen
        self.outdated = False

    def __str__(self):
        disp = 'Class\tTrueNum\tPredNum\tPrec\tRecall'
        if self.outdated:
            self.compute()
        for i in range(self.num_classes):
            disp += '\n{}\t{}\t{}\t{:.3f}\t{:.3f}'.format(self.class_names[i], self.tp_fn[i], self.tp_fp[i], self.precs[i], self.recs[i])
        disp += '\n' + '=' * 15
        disp += '\n{}\tAcc:\t{:.3f}\t{:.3f}\t{:.3f}'.format('mean', self.acc, self.precs.mean(), self.recs.mean())
        if self.num_classes > 2:
            disp += '\n' + '=' * 5 + " combine all positive classes" + '=' * 5
            disp += '\nPositive\tAcc:\t{}\t{}\t{}'.format(self.combined_acc, self.combined_prec, self.combined_rec)
        return disp



def temporal_nms(bboxes, thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = bboxes[:, 0]
    t2 = bboxes[:, 1]
    scores = bboxes[:, 2]

    durations = t2 - t1
    order = scores.argsort()[::-1]  # proposal得分由高到低的indices

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1] # iou小于阈值的剩余proposals

    return bboxes[keep, :]
