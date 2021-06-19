import logging
import math
import os
import os.path
import os.path as osp
import pdb
import pickle
import time

import numpy as np
import torch
import torch.utils.data as data
from numpy.random import randint
from tqdm import tqdm

from ops.io import load_proposal_file
from ops.utils import temporal_iou


class ActionInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None,
                 best_iou=None, overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps

        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self

        self.loc_reg = None
        self.size_reg = None

    def compute_regression_targets(self, gt_list, fg_thresh):
        if self.best_iou < fg_thresh:
            # background proposals do not need this
            return
        # find the groundtruth instance with the highest IOU
        ious = [temporal_iou((self.start_frame, self.end_frame), (gt.start_frame, gt.end_frame)) for gt in gt_list]
        best_gt_id = np.argmax(ious)
        best_gt = gt_list[best_gt_id]
        prop_center = (self.start_frame + self.end_frame) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame) / 2
        prop_size = self.end_frame - self.start_frame + 1
        gt_size = best_gt.end_frame - best_gt.start_frame + 1

        # get regression target:
        # (1). center shift propotional to the proposal duration
        # (2). logarithm of the groundtruth duration over proposal duraiton

        self.loc_reg = (gt_center - prop_center) / prop_size
        try:
            self.size_reg = math.log(gt_size / prop_size)
        except:
            print(gt_size, prop_size, self.start_frame, self.end_frame)
            raise

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1

    @property
    def regression_targets(self):
        return [self.loc_reg, self.size_reg] if self.loc_reg is not None else [0, 0]


class VideoRecord:
    def __init__(self, prop_record, num_classes=None):
        self._data = prop_record

        frame_count = int(self._data[1])

        # build instance record
        self.gt = [
            ActionInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            ActionInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]),
                        best_iou=float(x[1]), overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
        ]
        if num_classes is not None:
            self.proposals = list(filter(lambda x: x.label <= num_classes, self.proposals))

        self.proposals = list(filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def id(self):
        return self._data[0].strip("\n").split("/")[-1]
    @property
    def num_frames(self):
        return int(self._data[1])

    def get_fg(self, fg_thresh, with_gt=True):
        fg = [p for p in self.proposals if p.best_iou > fg_thresh]
        if with_gt:
            fg.extend(self.gt)

        for x in fg:
            x.compute_regression_targets(self.gt, fg_thresh)
        return fg

    def get_negatives(self, incomplete_iou_thresh, bg_iou_thresh,
                      bg_coverage_thresh=0.01, incomplete_overlap_thresh=0.7):

        tag = [0] * len(self.proposals)

        incomplete_props = []
        background_props = []

        for i in range(len(tag)):
            if self.proposals[i].best_iou < incomplete_iou_thresh \
                    and self.proposals[i].overlap_self > incomplete_overlap_thresh:
                tag[i] = 1 # incomplete
                incomplete_props.append(self.proposals[i])

        for i in range(len(tag)):
            if tag[i] == 0 and \
                self.proposals[i].best_iou < bg_iou_thresh and \
                            self.proposals[i].coverage > bg_coverage_thresh:
                background_props.append(self.proposals[i])
        return incomplete_props, background_props


class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset_configs, prop_file, ft_path, exclude_empty=True,
                 epoch_multiplier=1, test_mode=False, gt_as_fg=True, reg_stats=None):
        
        self.ft_path = ft_path
        self.prop_file = prop_file
        self.ft_loader = dataset_configs.get('ft_loader', None)

        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier
        self.gt_as_fg = gt_as_fg
        self.test_mode = test_mode

        self.fg_ratio = dataset_configs['fg_ratio']
        self.incomplete_ratio = dataset_configs['incomplete_ratio']
        self.bg_ratio = dataset_configs['bg_ratio']
        self.prop_per_video = dataset_configs['prop_per_video']
        self.num_classes = dataset_configs['num_class']
        self.ft_file_ext = dataset_configs['ft_file_ext']

        self.fg_iou_thresh = dataset_configs['fg_iou_thresh']
        self.bg_iou_thresh = dataset_configs['bg_iou_thresh']
       
        self.bg_coverage_thresh = dataset_configs['bg_coverage_thresh']
        self.incomplete_iou_thresh = dataset_configs['incomplete_iou_thresh']
        self.incomplete_overlap_thresh = dataset_configs['incomplete_overlap_thresh']

        self.starting_ratio = dataset_configs['starting_ratio']
        self.ending_ratio = dataset_configs['ending_ratio']

        denum = self.fg_ratio + self.bg_ratio + self.incomplete_ratio
        self.fg_per_video = int(self.prop_per_video * (self.fg_ratio / denum))
        self.bg_per_video = int(self.prop_per_video * (self.bg_ratio / denum))
        self.incomplete_per_video = self.prop_per_video - self.fg_per_video - self.bg_per_video
         
        parse_time = time.time()
        logging.info('Parsing proposal file')
        self._parse_prop_file(stats=reg_stats)
        print("File parsed. Time:{:.2f}".format(time.time() - parse_time))

        if not self.test_mode:
            """pre-compute iou and distance among proposals"""
            self._prepare_prop_dict()
        
    def _parse_prop_file(self, stats=None):
        print('loading prop_file ' + self.prop_file)
        prop_info = load_proposal_file(self.prop_file)
    
        self.video_list = [VideoRecord(p, self.num_classes) for p in prop_info]
        
        print('max number of proposal in one video is %d' % max([len(v.proposals) for v in self.video_list]))
        print('create video list')  # empty proposal problem starts
        if self.exclude_empty and not self.test_mode:
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        self.video_dict = {v.id: v for v in self.video_list}
        
        if not self.test_mode:
            # construct three pools:
            # 1. Foreground
            # 2. Background
            # 3. Incomplete

            self.fg_pool = []
            self.bg_pool = []
            self.incomp_pool = []

            for v in self.video_list:
                self.fg_pool.extend([(v.id, prop) for prop in v.get_fg(self.fg_iou_thresh, self.gt_as_fg)])

                incomp, bg = v.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                            self.bg_coverage_thresh, self.incomplete_overlap_thresh)

                self.incomp_pool.extend([(v.id, prop) for prop in incomp])
                self.bg_pool.extend([(v.id, prop) for prop in bg])
            if stats is None:
                self._compute_regresssion_stats()
            else:
                self.stats = stats
    
    def _video_centric_sampling(self, video):
        '''In each video, sample three kinds of proposals: positive(aka fg)/incomplete/negative(aka bg)'''
        fg, incomp, bg = self.prop_dict[video.id][0], self.prop_dict[video.id][1], self.prop_dict[video.id][2]

        out_props = []

        # 8 props per video
        for i in range(self.fg_per_video):
            props = self._sample_proposals(0, video.id, fg, 1)
            out_props.extend(props)  # sample foreground

        for i in range(self.incomplete_per_video):
            if len(incomp) == 0:
                props = self._sample_proposals(0, video.id, fg, 1)
            else:
                props = self._sample_proposals(1, video.id, incomp, 1)
            out_props.extend(props)  # sample incomp

        for i in range(self.bg_per_video):
            if len(bg) == 0:
                props = self._sample_proposals(0, video.id, fg, 1)
            else:
                props = self._sample_proposals(2, video.id, bg, 1)
            out_props.extend(props)  # sample bg


        return out_props

    def _sample_indices(self, prop, frame_cnt):
        '''get the coordinates of the proposal and the extended proposal'''
        start_frame = prop.start_frame + 1
        end_frame = prop.end_frame

        duration = end_frame - start_frame + 1
        assert duration != 0, (prop.start_frame, prop.end_frame, prop.best_iou)

        # extend proposal
        valid_starting = max(1, start_frame - int(duration * self.starting_ratio))
        valid_ending = min(frame_cnt, end_frame + int(duration * self.ending_ratio))

        # get starting
        act_s_e = (start_frame, end_frame)
        comp_s_e = (valid_starting, valid_ending)

        offsets = np.concatenate((act_s_e, comp_s_e))
        return offsets

    def _load_prop_data(self, prop):

        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames

        # get the coordinates of the proposal and the extended proposal 
        prop_indices = self._sample_indices(prop[0][1], frame_cnt)

        # get label
        if prop[1] == 0:
            label = prop[0][1].label
        elif prop[1] == 1:
            label = prop[0][1].label  # incomplete
        elif prop[1] == 2:
            label = 0  # background
        else:
            raise ValueError()

        # get regression target
        if prop[1] == 0:
            reg_targets = prop[0][1].regression_targets
            reg_targets = (reg_targets[0] - self.stats[0][0]) / self.stats[1][0], \
                          (reg_targets[1] - self.stats[0][1]) / self.stats[1][1]
        else:
            reg_targets = (0.0, 0.0)

        return prop_indices, label, reg_targets, prop[1]

    def get_training_data(self, index):
        video = self.video_list[index]
        props = self._video_centric_sampling(video)
    
        out_prop_ind = []
        out_prop_type = []
        out_prop_labels = []
        out_prop_reg_targets = []
        # gt_instances = [[x.start_frame, x.end_frame, x.label]  for x in video.gt]

        for idx, p in enumerate(props):
            prop_indices, prop_label, reg_targets, prop_type = self._load_prop_data(p)

            out_prop_ind.append(prop_indices)
            out_prop_labels.append(prop_label)
            out_prop_reg_targets.append(reg_targets)
            out_prop_type.append(prop_type)

        out_prop_labels = torch.from_numpy(np.array(out_prop_labels))
        out_prop_reg_targets = torch.from_numpy(np.array(out_prop_reg_targets, dtype=np.float32))
        out_prop_type = torch.from_numpy(np.array(out_prop_type))

        #load prop fts
 
        video_id = video.id.split('/')[-1]

        ft_full_path = osp.join(self.ft_path, video_id + self.ft_file_ext)
        if self.ft_file_ext == '':
            ft_tensor = torch.load(ft_full_path).float()
        else:
            if self.ft_file_ext == '.npy':
                ft_arr = np.load(ft_full_path)
            else:
                with open(ft_full_path, 'rb') as f:
                    ft_arr = pickle.load(f, encoding='bytes')
            ft_tensor = torch.from_numpy(ft_arr)
                
        slice_tensor = ft_tensor.transpose(1, 0)
    
        return slice_tensor, np.array(out_prop_ind), out_prop_type, out_prop_labels, out_prop_reg_targets, index

    def _compute_regresssion_stats(self):

        targets = []
        for video in self.video_list:
            fg = video.get_fg(self.fg_iou_thresh, False)
            # print(len(fg))
            # always zero?
            for p in fg:
                targets.append(list(p.regression_targets))
        # targrts might be empty
        self.stats = np.array((np.mean(targets, axis=0), np.std(targets, axis=0)))

    def get_test_data(self, video):
        '''only return proposal in scaled coordinates and abs coordinates'''
        props = video.proposals
        video_id = video.id
        frame_cnt = video.num_frames
 
        # process proposals to subsampled sequences
        rel_prop_list = []
        proposal_tick_list = []

        
        ft_full_path = osp.join(self.ft_path, video_id + self.ft_file_ext)
        if self.ft_file_ext == '':
            ft_tensor = torch.load(ft_full_path).float()
        else:
            if self.ft_file_ext == '.npy':
                ft_arr = np.load(ft_full_path)
            else:
                with open(ft_full_path, 'rb') as f:
                    ft_arr = pickle.load(f, encoding='bytes')
            ft_tensor = torch.from_numpy(ft_arr)
        
        for proposal in props:

            rel_prop = proposal.start_frame / frame_cnt, proposal.end_frame / frame_cnt
            rel_duration = rel_prop[1] - rel_prop[0]
            rel_starting_duration = rel_duration * self.starting_ratio
            rel_ending_duration = rel_duration * self.ending_ratio
            rel_starting = rel_prop[0] - rel_starting_duration
            rel_ending = rel_prop[1] + rel_ending_duration

            real_rel_starting = max(0.0, rel_starting)
            real_rel_ending = min(1.0, rel_ending)


            proposal_ticks =  int(rel_prop[0] * frame_cnt), int(rel_prop[1] * frame_cnt), \
                              int(real_rel_starting * frame_cnt), int(real_rel_ending * frame_cnt)

            rel_prop_list.append(rel_prop)
            proposal_tick_list.append(proposal_ticks)
  
        ft_tensor = ft_tensor.transpose(1,0)

        return ft_tensor, torch.from_numpy(np.array(proposal_tick_list)), torch.from_numpy(np.array(rel_prop_list)), video_id, video.num_frames


    def _prepare_prop_dict(self):
        self.prop_dict = {}
        pbar = tqdm(total=len(self.video_list))
        for cnt, video in enumerate(self.video_list):
            pbar.update(1)
            fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg)
            incomp, bg = video.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                             self.bg_coverage_thresh, self.incomplete_overlap_thresh)
            self.prop_dict[video.id] = [fg, incomp, bg]
                        
        pbar.close()


    def _sample_proposals(self, proposal_type, video_id, type_pool, requested_num):
        # sample requested number of proposals
        idx = np.random.choice(len(type_pool), requested_num)
        center_prop = type_pool[idx[0]]

        props = [((video_id, center_prop), proposal_type)]
        return props

    def get_all_gt(self):
        gt_list = []
        for video in self.video_list:
            vid = video.id
            gt_list.extend([[vid, x.label - 1, x.start_frame / video.num_frames,
                             x.end_frame / video.num_frames] for x in video.gt])
        return gt_list
    
    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index])
        else:
            return self.get_training_data(real_index)

    def __len__(self):
        return int(len(self.video_list) * self.epoch_multiplier)

    

