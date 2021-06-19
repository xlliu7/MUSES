# -*- coding: utf-8 -*-
import torch

from models import TwoStageDetector


if __name__ == '__main__':
    model_configs = dict(
        num_class=25,
        feat_dim=1024,
        act_net_dims=[2048, 384],
        comp_net_dims=[4096, 384],
        dropout=0.8,
        roi_scale=0.125
    )

    model = TwoStageDetector(model_configs, 1024)
    model.cuda()

    input = torch.rand([1, 1024, 256])

    # generate 8 proposals with the same timestamps
    rois = torch.FloatTensor([[[40, 80, 20, 100]]]).repeat(1, 8, 1)
    # prop_type: indicate the types of these proposals, 0 for positive, 1 for incomplete, 2 for negative
    prop_type = torch.LongTensor([[0, 1, 1, 1, 1, 1, 1, 2]])
    outputs = model(input.cuda(), rois.cuda(), None, None, prop_type.cuda())

    print('Test passed')
