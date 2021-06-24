# -*- coding: utf-8 -*-
import logging
import os
import pdb
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
from opts import parser
# import torchvision
# from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm

from dataset import VideoDataSet
from models import TwoStageDetector
from ops.model_ops import ClassWiseRegressionLoss, CompletenessLoss
from ops.utils import get_and_save_args, get_logger

np.seterr('raise')
SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_loss = 100
cudnn.benchmark = False
pin_memory = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def main():

    global args, best_loss, writer

    configs = get_and_save_args(parser)
    parser.set_defaults(**configs)
    dataset_configs = configs["dataset_configs"]
    model_configs = configs["model_configs"]
    args = parser.parse_args()
    if 'batch_size' in model_configs:
        args.batch_size = model_configs['batch_size']
    if 'iter_size' in model_configs:
        args.iter_size = model_configs['iter_size']

    model = TwoStageDetector(
        model_configs, roi_size=dataset_configs['roi_pool_size'])
    cnt = 0
    for p in model.parameters():
        cnt += p.data.numel()
    print(cnt)

    """copy codes and creat dir for saving models and logs"""
    if not os.path.isdir(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(args.snapshot_pref, date+'_train.log')
    get_logger(args, logfile)
    logging.info(' '.join(sys.argv))
    logging.info('\ncreating folder: ' + args.snapshot_pref)

    if not args.evaluate:
        pass
        # writer = SummaryWriter(args.snapshot_pref)
        # make a copy of the entire project folder, which can cost huge space
        # recorder = Recorder(args.snapshot_pref, ["models", "__pycache__"])
        # recorder.writeopt(args)

    logging.info('\nruntime args\n\n{}\n\nconfig\n\n{}'.format(
        args, dataset_configs))
    logging.info(str(model))
    logging.info(str(cnt))
    if 'lr' in model_configs:
        args.lr = model_configs['lr']
        logging.info('Using learning rate {}'.format(args.lr))

    """construct model"""

    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info(("=> loaded checkpoint '{}' (epoch {})"
                          .format(args.evaluate, checkpoint['epoch'])))
        else:
            logging.info(
                ("=> no checkpoint found at '{}'".format(args.resume)))

    """construct dataset"""

    train_dataset = VideoDataSet(dataset_configs,
                                 prop_file=dataset_configs['train_prop_file'],
                                 ft_path=dataset_configs['train_ft_path'],
                                 epoch_multiplier=dataset_configs['training_epoch_multiplier'],
                                 test_mode=False)
    kwargs = {}
    kwargs['shuffle'] = True

    loss_kwargs = {}
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True, **kwargs)  # in training we drop the last incomplete minibatch

    # val_loader = None
    val_loader = torch.utils.data.DataLoader(
        VideoDataSet(dataset_configs,
                     prop_file=dataset_configs['test_prop_file'],
                     ft_path=dataset_configs['test_ft_path'],
                     epoch_multiplier=dataset_configs['testing_epoch_multiplier'],
                     reg_stats=train_loader.dataset.stats,
                     test_mode=False),
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True)
    logging.info('Dataloaders constructed')

    """loss and optimizer"""
    activity_criterion = torch.nn.CrossEntropyLoss(**loss_kwargs).cuda()
    completeness_criterion = CompletenessLoss().cuda()
    regression_criterion = ClassWiseRegressionLoss().cuda()

    # for group in policies:
    #     logging.info(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #         group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, model, activity_criterion,
                 completeness_criterion, regression_criterion, 0, -1)
        return

    print('Start training loop')

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        train(train_loader, model, activity_criterion,
              completeness_criterion, regression_criterion, optimizer, epoch)

        # evaluate on validation set
        latest_ckpt_path = args.snapshot_pref + \
            '_'.join((args.dataset, 'latest', 'checkpoint.pth.tar'))
        ckpt = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': 1000,
            'reg_stats': torch.from_numpy(train_loader.dataset.stats)}

        torch.save(ckpt, latest_ckpt_path)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            loss = validate(val_loader, model, activity_criterion, completeness_criterion,
                            regression_criterion, (epoch + 1) * len(train_loader), epoch)
            # remember best validation loss and save checkpoint
            # loss = np.exp(-epoch/100)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            ckpt['best_loss'] = best_loss
            save_checkpoint(ckpt, is_best, epoch,
                            filename='checkpoint.pth.tar')

    # writer.close()


def get_item(input):
    if isinstance(input, (int, float)):
        return input
    else:
        return input.item()


def train(train_loader, model, act_criterion, comp_criterion, regression_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    comp_losses = AverageMeter()
    reg_losses = AverageMeter()
    act_accuracies = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()

    ohem_num = train_loader.dataset.fg_per_video
    comp_group_size = train_loader.dataset.fg_per_video + \
        train_loader.dataset.incomplete_per_video
    for i, (video_fts, rois, prop_type, prop_labels, prop_reg_targets, video_idx) in enumerate(train_loader):
        # print('batch %d' % i)
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = video_fts.size(0)

        activity_out, activity_target, activity_prop_type, \
            completeness_out, completeness_target, \
            regression_out, regression_labels, regression_target = model(video_fts, rois, prop_labels,
                                                                         prop_reg_targets, prop_type)

        act_loss = act_criterion(activity_out, activity_target)
        comp_loss = comp_criterion(
            completeness_out, completeness_target, ohem_num, comp_group_size)
        reg_loss = regression_criterion(
            regression_out, regression_labels, regression_target)

        loss = act_loss + comp_loss * args.comp_loss_weight + \
            reg_loss * args.reg_loss_weight

        losses.update(loss.item(), batch_size)
        act_losses.update(act_loss.item(), batch_size)
        comp_losses.update(comp_loss.item(), batch_size)
        reg_losses.update(reg_loss.item(), batch_size)

        act_acc = accuracy(activity_out, activity_target)
        act_accuracies.update(act_acc[0].item(), activity_out.size(0))

        fg_indexer = (activity_prop_type == 0)
        bg_indexer = (activity_prop_type == 2)

        try:
            fg_acc = accuracy(
                activity_out[fg_indexer, :], activity_target[fg_indexer])
            fg_accuracies.update(fg_acc[0].item(), len(fg_indexer))

            bg_acc = accuracy(
                activity_out[bg_indexer, :], activity_target[bg_indexer])
            bg_accuracies.update(bg_acc[0].item(), len(bg_indexer))
        except:
            # print('warning: failed to compute fg/bg acc')
            pass
        loss.backward()

        if (i + 1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            if args.iter_size != 1:
                for g in optimizer.param_groups:
                    for p in g['params']:
                        p.grad /= args.iter_size

            if args.clip_gradient is not None:
                total_norm = clip_grad_norm(
                    model.parameters(), args.clip_gradient)
                if total_norm > args.clip_gradient:
                    logging.info("clipping gradient: {} with coef {}".format(
                        total_norm, args.clip_gradient / total_norm))
            else:
                total_norm = 0

            optimizer.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        # writer.add_scalar('data/loss', losses.val, epoch*len(train_loader)+i+1)
        # writer.add_scalar('data/Reg_loss', reg_losses.val, epoch*len(train_loader)+i+1)
        # writer.add_scalar('data/Act_loss', act_losses.val, epoch*len(train_loader)+i+1)
        # writer.add_scalar('data/comp_loss', comp_losses.val, epoch*len(train_loader)+i+1)

        # writer.add_scalar('data/epoch', epoch, epoch*len(train_loader)+i+1)
        # writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'], epoch*len(train_loader)+i+1)
        if i % args.iter_size == 0 and i // args.iter_size % args.print_freq == 0:
            # logging.info('\n{}\n{}'.format(activity_out.argmax(1), activity_target))
            logging.info('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Act. Loss {act_losses.val:.3f} ({act_losses.avg: .3f}) \t'
                         'Comp. Loss {comp_losses.val:.3f} ({comp_losses.avg: .3f}) '
                         .format(
                             epoch, i, len(train_loader), batch_time=batch_time,
                             data_time=data_time, loss=losses, act_losses=act_losses,
                             comp_losses=comp_losses, lr=optimizer.param_groups[0]['lr'], ) +
                         '\tReg. Loss {reg_loss.val:.3f} ({reg_loss.avg:.3f})'.format(
                             reg_loss=reg_losses)
                         + '\n Act. FG {fg_acc.val:.02f} ({fg_acc.avg:.02f}) Act. BG {bg_acc.avg:.02f} ({bg_acc.avg:.02f})'
                         .format(act_acc=act_accuracies,
                                 fg_acc=fg_accuracies, bg_acc=bg_accuracies)
                         )


def validate(val_loader, model, act_criterion, comp_criterion, regression_criterion, iter, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    comp_losses = AverageMeter()
    reg_losses = AverageMeter()
    act_accuracies = AverageMeter()
    fg_accuracies = AverageMeter()
    bg_accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    ohem_num = val_loader.dataset.fg_per_video
    comp_group_size = val_loader.dataset.fg_per_video + \
        val_loader.dataset.incomplete_per_video

    for i, (video_fts, rois, prop_type, prop_labels, prop_reg_targets, video_idx) in enumerate(val_loader):
        # measure data loading time
        batch_size = video_fts.size(0)

        activity_out, activity_target, activity_prop_type, \
            completeness_out, completeness_target, \
            regression_out, regression_labels, regression_target = model(video_fts, rois, prop_labels,
                                                                         prop_reg_targets, prop_type)

        act_loss = act_criterion(activity_out, activity_target)
        comp_loss = comp_criterion(
            completeness_out, completeness_target, ohem_num, comp_group_size)
        try:
            reg_loss = regression_criterion(
                regression_out, regression_labels, regression_target)
            reg_loss_value = reg_loss.item()
        except Exception as e:
            logging.info(str(e))
            reg_loss = 0
            reg_loss_value = 0
        loss = act_loss + comp_loss * args.comp_loss_weight + \
            reg_loss * args.reg_loss_weight

        losses.update(loss.item(), batch_size)
        act_losses.update(act_loss.item(), batch_size)
        comp_losses.update(comp_loss.item(), batch_size)
        reg_losses.update(reg_loss_value, batch_size)

        act_acc = accuracy(activity_out, activity_target)
        act_accuracies.update(act_acc[0].item(), activity_out.size(0))

        fg_indexer = (activity_prop_type == 0)
        bg_indexer = (activity_prop_type == 2)

        # if len(fg_indexer) > 0:
        try:
            fg_acc = accuracy(
                activity_out[fg_indexer, :], activity_target[fg_indexer])
            fg_accuracies.update(fg_acc[0].item(), len(fg_indexer))

            bg_acc = accuracy(
                activity_out[bg_indexer, :], activity_target[bg_indexer])
            bg_accuracies.update(bg_acc[0].item(), len(bg_indexer))

        except Exception as e:
            pass

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.iter_size == 0 and (i // args.iter_size) % args.print_freq == 0:
            # logging.info('\n{}\n{}'.format(
            #     activity_out.argmax(1), activity_target))
            logging.info('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Act. Loss {act_loss.val:.3f} ({act_loss.avg:.3f})\t'
                         'Comp. Loss {comp_loss.val:.3f} ({comp_loss.avg:.3f})\t'
                         'Act. Accuracy {act_acc.val:.02f} ({act_acc.avg:.2f}) FG {fg_acc.val:.02f} BG {bg_acc.val:.02f}'.format(
                             i, len(val_loader), batch_time=batch_time, loss=losses,
                             act_loss=act_losses, comp_loss=comp_losses, act_acc=act_accuracies,
                             fg_acc=fg_accuracies, bg_acc=bg_accuracies) +
                         '\tReg. Loss {reg_loss.val:.3f} ({reg_loss.avg:.3f})'.format(
                             reg_loss=reg_losses))

    logging.info('Testing Results: Loss {loss.avg:.5f} \t '
                 'Activity Loss {act_loss.avg:.3f} \t '
                 'Completeness Loss {comp_loss.avg:.3f}\n'
                 'Act Accuracy {act_acc.avg:.02f} FG Acc. {fg_acc.avg:.02f} BG Acc. {bg_acc.avg:.02f}'
                 .format(act_loss=act_losses, comp_loss=comp_losses, loss=losses, act_acc=act_accuracies,
                         fg_acc=fg_accuracies, bg_acc=bg_accuracies)
                 + '\t Regression Loss {reg_loss.avg:.3f}'.format(reg_loss=reg_losses))

    # if iter > 0:
    #     writer.add_scalar('val/loss', losses.val, iter)
    #     writer.add_scalar('val/Reg_loss', reg_losses.val, iter)
    #     writer.add_scalar('val/Act_loss', act_losses.val, iter)
    #     writer.add_scalar('val/comp_loss', comp_losses.val, iter)
    #     writer.add_scalar('val/act_acc', act_accuracies.val, iter)
    #     writer.add_scalar('val/fg_acc', fg_accuracies.val, iter)
    #     writer.add_scalar('val/bg_acc', bg_accuracies.val, iter)
    return losses.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    filename = args.snapshot_pref + \
        '_'.join((args.dataset, 'epoch', str(epoch), filename))
    torch.save(state, filename)
    if is_best:
        best_name = args.snapshot_pref + \
            '_'.join((args.dataset, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
