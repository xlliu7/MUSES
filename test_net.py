import argparse
import os
import torch
import numpy as np
from dataset import VideoDataSet
from models import TwoStageDetector
from torch import multiprocessing
from ops.utils import get_configs
from tqdm import tqdm
import random
import os.path as osp
import pdb


parser = argparse.ArgumentParser(
    description="MUSES Testing Tool")
parser.add_argument('dataset', type=str, choices=['thumos14', 'muses'])
parser.add_argument('weights', type=str)
parser.add_argument('save_scores', type=str)
parser.add_argument('--cfg')
parser.add_argument('--save_raw_scores', type=str, default=None)
parser.add_argument('--no_regression', action="store_true", default=False)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()

configs = get_configs(args.dataset, args.cfg)
dataset_configs = configs['dataset_configs']
model_configs = configs["model_configs"]

num_class = model_configs['num_class']

gpu_list = args.gpus if args.gpus is not None else [0]
score_dir = osp.dirname(args.save_scores)
if not osp.exists(score_dir):
    print('creating score directory %s' % score_dir)
    os.makedirs(score_dir)


def runner_func(dataset, state_dict, stats, gpu_id, index_queue, result_queue):

    torch.cuda.set_device(gpu_id)
    net = TwoStageDetector(model_configs, test_mode=True, roi_size=dataset_configs['roi_pool_size'])
    net.load_state_dict(state_dict)
    # net.prepare_test_fc()
    net.eval()
    net.cuda()

    while True:
        index = index_queue.get()
        
        video_ft, prop_ticks, rel_props, video_id, n_frames = dataset[index]

        # calculate scores
        n_out = prop_ticks.size(0)
        act_scores = torch.zeros((n_out, num_class + 1)).cuda()
        comp_scores = torch.zeros((n_out, num_class)).cuda()

        if not args.no_regression:
            reg_scores = torch.zeros((n_out, num_class * 2)).cuda()
        else:
            reg_scores = None

        with torch.no_grad():
            act_scores, comp_scores, reg_scores = net(
            video_ft.unsqueeze(0).cuda(),
            prop_ticks.unsqueeze(0), None, None, None)

        if reg_scores is not None:
            reg_scores = reg_scores.view(-1, num_class, 2)
            reg_scores[:, :, 0] = reg_scores[:, :, 0] * stats[1, 0] + stats[0, 0]
            reg_scores[:, :, 1] = reg_scores[:, :, 1] * stats[1, 1] + stats[0, 1]

        # perform stpp on scores
        result_queue.put((dataset.video_list[index].id, rel_props.numpy(), act_scores.cpu().numpy(),
                          comp_scores.cpu().numpy(), reg_scores.cpu().numpy(), 0))


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')  # this is crucial to using multiprocessing processes with PyTorch


    # This net is used to provides setup settings. It is not used for testing.

    checkpoint = torch.load(args.weights)
    # pdb.set_trace()
    print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['best_loss']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}

    stats = checkpoint['reg_stats'].numpy()

    prop_file = dataset_configs['test_prop_file']
    print('using prop_file ' + prop_file)
    
    dataset = VideoDataSet(dataset_configs,
                          prop_file=prop_file,
                          ft_path=dataset_configs['test_ft_path'],
                          test_mode=True)
    print('Dataset Initilized')


    index_queue = ctx.Queue()
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=runner_func,
                           args=(dataset, base_dict, stats, gpu_list[i % len(gpu_list)],
                                 index_queue, result_queue))
               for i in range(args.workers)]


    for w in workers:
        w.daemon = True
        w.start()

    max_num = args.max_num if args.max_num > 0 else len(dataset)
    print('{} videos to process'.format(max_num))
    for i in range(max_num):
        index_queue.put(i)

    out_dict = {}
    pbar = tqdm(total=max_num)
    for i in range(max_num):
        pbar.update(1)
        rst = result_queue.get()
        out_dict[rst[0]] = rst[1:]
    pbar.close()

    if args.save_scores is not None:
        save_dict = {k: v[:-1] for k, v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_scores, 'wb'), pickle.HIGHEST_PROTOCOL)

    if args.save_raw_scores is not None:
        save_dict = {k: v[-1] for k, v in out_dict.items()}
        import pickle

        pickle.dump(save_dict, open(args.save_raw_scores, 'wb'), pickle.HIGHEST_PROTOCOL)
