import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from terminaltables import *

from dataset import VideoDataSet
from ops.utils import temporal_nms

sys.path.append('./anet_toolkit/Evaluation')
import os
import pdb
import pickle

from anet_toolkit.Evaluation.eval_detection import \
    compute_average_precision_detection

from ops.utils import get_configs, softmax


# options
parser = argparse.ArgumentParser(
    description="Evaluate detection performance metrics")
parser.add_argument('dataset', type=str, choices=['thumos14', 'muses'])
parser.add_argument('detection_pickles', type=str, nargs='+')
parser.add_argument('--nms_threshold', type=float, default=0.4)
parser.add_argument('--no_regression', default=False, action="store_true")
parser.add_argument('-j', '--ap_workers', type=int, default=16)
parser.add_argument('--top_k', type=int, default=None)
parser.add_argument('--cls_scores', type=str, nargs='+')
parser.add_argument('--reg_scores', type=str, default=None)
parser.add_argument('--cls_top_k', type=int, default=1)
parser.add_argument('--cfg', default='data/dataset_cfg.yml')
parser.add_argument('--score_weights', type=float, default=None, nargs='+')
parser.add_argument('--min_length', type=float, default=None, help='minimum duration of proposals in second')
parser.add_argument('--one_iou', action='store_true')
parser.add_argument('--no_comp', action='store_true')


args = parser.parse_args()

configs = get_configs(args.dataset, args.cfg)
dataset_configs = configs['dataset_configs']
model_configs = configs["model_configs"]
num_class = model_configs['num_class']

nms_threshold = args.nms_threshold if args.nms_threshold else configs['evaluation']['nms_threshold']
top_k = args.top_k if args.top_k else configs['evaluation']['top_k']

print('---'*10)
print(time.strftime('%Y-%m-%d %H:%M:%S'))
print("initiating evaluation of detection results {}".format(args.detection_pickles))
print('top_k={}'.format(top_k))
sys.stdout.flush()

score_pickle_list = []
for pc in args.detection_pickles:
    score_pickle_list.append(pickle.load(open(pc, 'rb')))

if args.score_weights:
    weights = np.array(args.score_weights) / sum(args.score_weights)
else:
    weights = [1.0/len(score_pickle_list) for _ in score_pickle_list]


def merge_scores(vid):
    def merge_part(arrs, index, weights):
        if arrs[0][index] is not None:
            return np.sum([a[index] * w for a, w in zip(arrs, weights)], axis=0)
        else:
            return None

    arrays = [pc[vid] for pc in score_pickle_list]
    act_weights = weights
    comp_weights = weights
    reg_weights = weights
    rel_props = score_pickle_list[0][vid][0]

    return rel_props, \
           merge_part(arrays, 1, act_weights), \
           merge_part(arrays, 2, comp_weights), \
           merge_part(arrays, 3, reg_weights)

print('Merge detection scores from {} sources...'.format(len(score_pickle_list)))
detection_scores = {k: merge_scores(k) for k in score_pickle_list[0]}
print('Done.')


if 'deploy_prop_file' in dataset_configs:
    prop_file = dataset_configs['deploy_prop_file']
else:
    prop_file = dataset_configs['test_prop_file']
if 'deploy_online_slice' in dataset_configs:
    online_slice = dataset_configs['deploy_online_slice']
else:
    online_slice = dataset_configs.get('online_slice', False)


dataset = VideoDataSet(dataset_configs,
                    prop_file=prop_file,
                    ft_path=dataset_configs['train_ft_path'],
                    test_mode=True)
from functools import reduce

gt_lens = np.array(reduce(lambda x,y: x+y, [[(x.end_frame-x.start_frame)/6 for x in v.gt] for v in dataset.video_list]))
# pdb.set_trace()
dataset_detections = [dict() for i in range(num_class)]



def merge_all_vid_scores(pickle_list):
    def merge_op(arrs, index, weights):
        if arrs[0][index] is not None:
            return np.sum([a[index] * w for a, w in zip(arrs, weights)], axis=0)
        else:
            return None
    out_score_dict = {}
    for vid in pickle_list[0]:
        arrays = [pc[vid] for pc in pickle_list]
        act_weights = weights
        comp_weights = weights
        reg_weights = weights
        rel_props = pickle_list[0][vid][0]

        out_score_dict[vid] = [rel_props, \
           merge_op(arrays, 1, act_weights), \
           merge_op(arrays, 2, comp_weights), \
           merge_op(arrays, 3, reg_weights)]
    
    return out_score_dict

if args.cls_scores:
    print('Using classifier scores from {}'.format(args.cls_scores))
    cls_score_pickle_list = []
    for pc in args.cls_scores:
        cls_score_pickle_list.append(pickle.load(open(pc, 'rb')))
    
    cls_score_dict = merge_all_vid_scores(cls_score_pickle_list)

    # cls_score_pc = pickle.load(open(args.cls_scores, 'rb'), encoding='bytes')
    # cls_score_dict = cls_score_pc
    # cls_score_dict = {os.path.splitext(os.path.basename(k.decode('utf-8')))[0]:v for k, v in cls_score_pc.items()}
else:
    cls_score_dict = None

if args.reg_scores:
    print('Using regression scores from {}'.format(args.reg_scores))
    reg_score_dict = pickle.load(open(args.reg_scores, 'rb'))
else:
    reg_score_dict = None


# generate detection results
def gen_detection_results(video_id, score_tp):
    if len(score_tp[0].shape) == 3:
        rel_prop = np.squeeze(score_tp[0], 0)
    else:
        rel_prop = score_tp[0]

    # standardize regression scores
    reg_scores = score_tp[3]
    if reg_scores is None:
        reg_scores = np.zeros((len(rel_prop), num_class, 2), dtype=np.float32)
    reg_scores = reg_scores.reshape((-1, num_class, 2))

    if cls_score_dict is None:
        combined_scores = softmax(score_tp[1][:, :])
        combined_scores = combined_scores[:,1:]
    else:
        combined_scores = softmax(cls_score_dict[video_id][1])[:, 1:]
        if combined_scores.shape[1] < score_tp[2].shape[1]:
            combined_scores = np.concatenate(
                (combined_scores, np.zeros([len(combined_scores), score_tp[2].shape[1]-combined_scores.shape[1]])), axis=1)
        elif combined_scores.shape[1] > score_tp[2].shape[1]:
            combined_scores = combined_scores[:, :score_tp[2].shape[1]]
    if not args.no_comp:
        combined_scores = combined_scores * np.exp(score_tp[2])
    keep_idx = np.argsort(combined_scores.ravel())[-top_k:]

    # pdb.set_trace()
    delete_short = args.min_length is not None
    if delete_short:
        print('delete short proposals')
        duration = dataset.video_dict[video_id].num_frames / 6
        prop_duration = duration * (rel_prop[:,1] - rel_prop[:, 0])
        non_short_prop_idx = np.where(prop_duration <= args.min_length)[0]
        keep_idx = [x for x in keep_idx if x // num_class in non_short_prop_idx]
    
    # keep_prop_num = len({x//num_class for x in keep_idx})

    for k in keep_idx:
        cls = k % num_class
        prop_idx = k // num_class
        if video_id not in dataset_detections[cls]:
            dataset_detections[cls][video_id] = np.array([
                [rel_prop[prop_idx, 0], rel_prop[prop_idx, 1], combined_scores[prop_idx, cls],
                reg_scores[prop_idx, cls, 0], reg_scores[prop_idx, cls, 1]]
            ])
        else:
            dataset_detections[cls][video_id] = np.vstack(
                [dataset_detections[cls][video_id],
                [rel_prop[prop_idx, 0], rel_prop[prop_idx, 1], combined_scores[prop_idx, cls],
                reg_scores[prop_idx, cls, 0], reg_scores[prop_idx, cls, 1]]])
    
    return len(keep_idx)


print("Preprocessing detections...")
orig_num_list = []
keep_num_list = []
def mean(x):
    return sum(x)/len(x)

for k, v in detection_scores.items():
    orig_num = len(v[0])
    keep_num = gen_detection_results(k, v)
    orig_num_list.append(orig_num)
    keep_num_list.append(keep_num)
print('Done. {} videos, avg prop num {:.0f} => {:.0f}'.format(len(detection_scores), mean(orig_num_list), mean(keep_num_list)))

# perform NMS
print("Performing nms with thr {} ...".format(nms_threshold))
for cls in range(num_class):
    dataset_detections[cls] = {
        k: temporal_nms(v, nms_threshold) for k,v in dataset_detections[cls].items()
    }
print("NMS Done.")

def perform_regression(detections):
    t0 = detections[:, 0]
    t1 = detections[:, 1]
    center = (t0 + t1) / 2
    duration = (t1 - t0)

    new_center = center + duration * detections[:, 3]
    new_duration = duration * np.exp(detections[:, 4])

    new_detections = np.concatenate((
        np.clip(new_center - new_duration / 2, 0, 1)[:, None], np.clip(new_center + new_duration / 2, 0, 1)[:, None], detections[:, 2:]
    ), axis=1)
    return new_detections

# perform regression
if not args.no_regression:
    print("Performing location regression...")
    for cls in range(num_class):
        dataset_detections[cls] = {
            k: perform_regression(v) for k, v in dataset_detections[cls].items()
        }
    print("Regression Done.")
else:
    print("Skip regresssion as requested by --no_regression")


# ravel test detections
def ravel_detections(detection_db, cls):
    detection_list = []
    for vid, dets in detection_db[cls].items():
        detection_list.extend([[vid, cls] + x[:3] for x in dets.tolist()])
    df = pd.DataFrame(detection_list, columns=["video-id", "cls","t-start", "t-end", "score"])
    return df


plain_detections = [ravel_detections(dataset_detections, cls) for cls in range(num_class)]

# get gt
gt_list = []

all_gt = dataset.get_all_gt()

all_gt = pd.DataFrame(all_gt, columns=["video-id", "cls","t-start", "t-end"])
gt_by_cls = []
for cls in range(num_class):
    gt_by_cls.append(all_gt[all_gt.cls == cls].reset_index(drop=True).drop('cls', 1))
    print(cls, len(gt_by_cls[cls]))
# pdb.set_trace()

pickle.dump(gt_by_cls, open('gt_dump.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(plain_detections, open('pred_dump.pc', 'wb'), pickle.HIGHEST_PROTOCOL)
print("Calling mean AP calculator from toolkit with {} workers...".format(args.ap_workers))

if args.one_iou:
    iou_range = [0.5]
else:
    if args.dataset == 'thumos14':
        iou_range = np.arange(0.1, 1.0, 0.1)
    elif args.dataset == 'muses':
        iou_range = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    else:
        iou_range = np.arange(0.5, 1.0, 0.05)
        # raise ValueError("unknown dataset {}".format(args.dataset))

ap_values = np.zeros((num_class, len(iou_range)))


def eval_ap(iou, iou_idx, cls, gt, predition):
    ap = compute_average_precision_detection(gt, predition, iou)
    sys.stdout.flush()
    return cls, iou_idx, ap


def callback(rst):
    sys.stdout.flush()
    ap_values[rst[0], rst[1]] = rst[2][0]

pool = Pool(args.ap_workers)
jobs = []
for iou_idx, min_overlap in enumerate(iou_range):
    for cls in range(num_class):
        if len(gt_by_cls[cls]) == 0:
            continue
        jobs.append(pool.apply_async(eval_ap, args=([min_overlap], iou_idx, cls, gt_by_cls[cls], plain_detections[cls],),callback=callback))
pool.close()
pool.join()
print("Evaluation done.\n\n")
map_iou = ap_values.mean(axis=0)
per_cls_map = ap_values.mean(axis=1)
#
# for 
display_title = "Detection Performance on {}".format(args.dataset)

display_data = [["IoU thresh"], ["mAP"]]

for i in range(len(iou_range)):
    display_data[0].append("{:.02f}".format(iou_range[i]))
    display_data[1].append("{:.04f}".format(map_iou[i]))

display_data[0].append('Average')
display_data[1].append("{:.04f}".format(map_iou.mean()))
table = AsciiTable(display_data, display_title)
table.justify_columns[-1] = 'right'
table.inner_footing_row_border = True
print(table.table)
# first_line = '\t'.join(['iou'], ['{:.02f}'])

print('Per-class average AP over all iou thresholds')

for i,x in enumerate(per_cls_map):
    print('%.4f' % x, end='\t')
    
print(time.strftime('%Y-%m-%d %H:%M:%S') + ' Done')

