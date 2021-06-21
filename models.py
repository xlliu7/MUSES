import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _single

import numpy as np
import torch.nn.functional as F

from ops.roi_pool import RoIPool
from ops.dcn import deform_conv


def make_mlp(dims, drop_last_relu=False):
    assert len(dims) > 0
    layers = []
    num_layers = len(dims) - 1

    for i in range(num_layers):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i != num_layers - 1:
            layers.append(nn.ReLU())
    if not drop_last_relu:
        layers.append(nn.ReLU())
        
    return layers


class RoIHead(nn.Module):
    def __init__(self, model_configs, test_mode=False, **kwargs):
        super(RoIHead, self).__init__()
        self.num_class = model_configs['num_class']
        
        self.dropout = model_configs['dropout']
        self.test_mode = test_mode
        self.roi_size = kwargs.get('roi_size', 4)
        self.act_net_dims = model_configs['act_net_dims']
        self.comp_net_dims = model_configs['comp_net_dims']
        self.use_dropout = model_configs.get('use_dropout', True)

        self.act_feat_dim = self.act_net_dims[0]
        self.comp_feat_dim = self.comp_net_dims[0]

        self._prepare()

        # for action classification
        self.Act_MLP = nn.Sequential(*make_mlp(self.act_net_dims))
        # for boundary regression and completeness classification. Please refer to SSN (Temporal action detection with structured segment networks) for details of completeness classification.
        self.Comp_MLP = nn.Sequential(*make_mlp(self.comp_net_dims))
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def _prepare(self):
        act_fc_dim = self.act_net_dims[-1]
        loc_fc_dim = self.comp_net_dims[-1]
        self.activity_fc = nn.Linear(act_fc_dim, self.num_class + 1)
        self.completeness_fc = nn.Linear(loc_fc_dim, self.num_class)
        self.regressor_fc = nn.Linear(loc_fc_dim, 2 * self.num_class)

        nn.init.normal_(self.activity_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.activity_fc.bias.data, 0)
        nn.init.normal_(self.completeness_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.completeness_fc.bias.data, 0)
        nn.init.normal_(self.regressor_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.regressor_fc.bias.data, 0)


    def forward(self, input, *args, **kwargs):
        completeness_fts = input
        shape = completeness_fts.shape   # [n, 512, 8]
        batch_size, channels, length = shape
        activity_fts = completeness_fts[:,:,length//4:3*length//4]  # Note that the features include extended part. We only take features inside the proposal for action/event classification  

        activity_fts = activity_fts.contiguous().view(batch_size, -1)
        completeness_fts = completeness_fts.view(batch_size, -1)

        out_act_fts = self.Act_MLP(activity_fts)
        comp_fts = self.Comp_MLP(completeness_fts)
        
        if self.use_dropout:
            act_fts = self.dropout_layer(out_act_fts)
        else:
            act_fts = out_act_fts

        raw_act_fc = self.activity_fc(act_fts)
        
        raw_comp_fc = self.completeness_fc(comp_fts)
        raw_regress_fc = self.regressor_fc(comp_fts)

        if not self.test_mode:
            raw_regress_fc = raw_regress_fc.view(-1, self.completeness_fc.out_features, 2).contiguous()
        else:
            raw_regress_fc = raw_regress_fc.view(-1, self.completeness_fc.out_features*2).contiguous()
        return raw_act_fc, raw_comp_fc, raw_regress_fc
        

class TALayer(nn.Module):
    '''(Single-scale) Temporal Aggregation Layer. For efficiency and convenience, we do not really apply the reshape operation and 2D convolution. Instead, we directly sample the points on the 1D feature sequence according to the kernel size and the width of 2D feature map. We implement this with deformable convolution with fixed offsets.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size_2d,
                 unit_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        '''
        in_channels:  number of channels of the input feature，
        out_channels: the number of channels of the output feature,
        kernel_size_2d: kernel size of 2D convolution
        unit_size: the width of 2D feature map
        '''
        super(TALayer, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)
        assert unit_size >= kernel_size_2d[1]
        self.unit_size = unit_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_2d = kernel_size_2d
        equiv_kernel = kernel_size_2d[0]*kernel_size_2d[1]
        # the kernel size of 2D deformable convolution
        self.kernel_size = [1, equiv_kernel]
        self.stride = _pair(stride)
        self.padding = [0, (equiv_kernel-1)//2]
        self.dilation = _pair(dilation)
        
        self.with_bias = bias

        self.groups = groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.rand([out_channels]))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / float(np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)

        self.reset_parameters()
    
        self.base_offset = self.get_base_offset(kernel_size_2d, unit_size).cuda()

    def get_base_offset(self, kernel_size, unit_size):
        num_group, group_size = kernel_size
        per_group_offset = []
        center_group_idx = (num_group - 1) // 2
        for i in range(num_group):
            per_group_offset.append((unit_size-group_size) * (i-center_group_idx))
        x_offset = [per_group_offset[i//group_size] for i in range(num_group*group_size)]
        y_offset = [0 for i in range(num_group*group_size)]
        yx_offset = torch.FloatTensor([y_offset, x_offset]).transpose(0,1).reshape([1, len(y_offset)*2, 1, 1])
        return yx_offset

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
       
        shape = x.shape
        x = x.reshape(shape[0], shape[1], 1, shape[2])
        offset_replicator = torch.ones([x.shape[0],1,x.shape[2], x.shape[3]], device=x.device, dtype=x.dtype)
        offset = self.base_offset * offset_replicator

        input_pad = (
            x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant',
                           0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding,
                          self.dilation, self.groups, 1)
        if self.with_bias:
            out = out + self.bias.reshape([1, -1, 1, 1])
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()

        out = out.squeeze(2)
        return out

    def __repr__(self):
        return 'TALayer({}, {}, {}, unit_size={})'.format(self.in_channels, self.out_channels, self.kernel_size_2d, self.unit_size)



class MSTALayer(nn.Module):
    '''Multi-scale Temporal Aggregation layer'''
    def __init__(self, input_channels, out_channels, kernel_sizes, unit_sizes, fusion_type='add'):
        super(MSTALayer, self).__init__()
        assert len(unit_sizes) == len(kernel_sizes), 'unit_sizes and kernel_sizes should have the same length'
        self.fusion_type = fusion_type
        if self.fusion_type != 'concat':
            per_branch_out_chn = out_channels
        else:
            assert out_channels % len(unit_sizes) == 0, 'out_channels must be divisible by number of branches'
            per_branch_out_chn = out_channels // len(unit_sizes)
        branches = [TALayer(input_channels, per_branch_out_chn, kernel_sizes[i], unit_sizes[i]) for i in range(len(unit_sizes))]
        self.branches = nn.ModuleList(branches)
    
    def forward(self, x):
        branch_outputs = [l(x) for l in self.branches]
        if self.fusion_type == 'add':
            return sum(branch_outputs)
        elif self.fusion_type == 'concat':
            return torch.cat(branch_outputs, dim=1)
        elif self.fusion_type == 'max':
            return torch.cat([x.unsqueeze(0) for x in branch_outputs], dim=0).max(0)[0]
        else:
            raise NotImplementedError


class BaseNet(nn.Module):
    '''Multi-scale Temporal Aggregation (MSTA) Subnet, composed of sequential MSTA layer'''
    def __init__(self, kernels, input_dim, dims=[384, 512], fusion_type='add'):
        '''kernels: a list of tuples [(kh1, kw1, W1), (kh2, kw2, W12), ...] that describes different branches of a MSTA layer. Each tuple describes the configuration of a single-scale Temporal Aggregation layer. In each layer, we first cut the input feature into units of length W and arrange them to a 2D feature map with a width of W. (kh, kw) is the kernel size of 2D convolution applied on the 2D feature map.
        
        input_dim: the dimension of the input feature

        dims: the dimension of each MSTA layer

        fusion_type: the way we fuse parallel single-scale temporal aggregation layer. Default: 'sum'
        '''
        super(BaseNet, self).__init__()
        # the width of 2D feature map in each TA branch
        self.unit_sizes = [x[-1] for x in kernels]
        self.kernel_sizes = [x[:2] for x in kernels]
        layers = []
        self.dims = dims
        self.fusion_type = fusion_type

        for i in range(len(self.dims)):
            in_channels = input_dim if i == 0 else self.dims[i-1]
            out_channels = self.dims[i]
            layers += [MSTALayer(in_channels, out_channels, self.kernel_sizes, self.unit_sizes, fusion_type=fusion_type), nn.ReLU()]
        self.layers = nn.Sequential(*layers)


    def forward(self, X):
        '''input: (N,C,T)'''
        return self.layers(X)



class TwoStageDetector(nn.Module):
    def __init__(self, model_configs, test_mode=False, roi_size=4, **kwargs):
        super(TwoStageDetector, self).__init__()
        
        self.num_class = model_configs['num_class']
        self.roi_size = roi_size
        self.test_mode = test_mode
        self.dropout = model_configs['dropout']
        self.feat_dim = model_configs['feat_dim']

        self.roi_scale = model_configs.get('roi_scale', 0.125)
        
        self.backbone_dims = model_configs.get('backbone_dims', [384, 512])
        self.residual = model_configs.get('residual', False)
        
        self.build_backbone()

        if self.roi_size != 4:
            print('warning, roi_size !=4')
        kwargs['roi_size'] = roi_size
        self.roi_extractor = RoIPool(self.roi_size*2, self.roi_scale)
        self.roi_head = RoIHead(model_configs, test_mode=test_mode, **kwargs)

    def build_backbone(self):   
        self.backbone = BaseNet(
            [[1,3,3],[3,3,3],[3,3,6], [3,3,9]], self.feat_dim, dims=self.backbone_dims)


    def get_optim_policies(self):

        normal_weight = []
        normal_bias = []
        bn_params = []
        
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, TALayer)):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                elif len(normal_bias) > 2:
                    print('more than 2 params')
            
            elif isinstance(m, nn.BatchNorm1d):
                bn_params.extend(list(m.parameters()))
          
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn_params, 'lr_mult': 1, 'name': 'bn_params', 'decay_mult': 1}
        ]

    def add_batch_ind(self, rois):
        # rois: tensor of shape (batch_size, video_size, 4). video_size is the number of proposals per video
        rois_np = rois.cpu().numpy()
        batch_size, video_size = rois.shape[:2]
        batch_ind = np.arange(batch_size).reshape([batch_size, 1, 1]).repeat(video_size, axis=1)
        # batch_ind = torch.from_numpy(batch_ind).cuda()
        
        rois_np = rois_np[:,:,2:]   # get extended roi
        rois_np[:,:,1] = np.maximum(rois_np[:,:,1], rois_np[:,:,0]+1) # in case right is smaller than left
        rois_with_batch_ind = np.concatenate((batch_ind, rois_np), axis=-1).reshape([batch_size*video_size, -1])
        return torch.from_numpy(rois_with_batch_ind.astype('float32')).cuda()

    def extract_features(self, input, *args):
        return self.backbone(input)

    def forward(self, input, rois, target, reg_target, prop_type):
        
        base_ft = self.backbone(input)
        if self.residual:
            base_ft += input

        rois_with_batch_ind = self.add_batch_ind(rois)
        roi_features = self.roi_extractor(base_ft, rois_with_batch_ind)

        batch_size = input.shape[0]

        raw_act_fc, raw_comp_fc, raw_regress_fc = self.roi_head(roi_features, gt_classes=target)

        # the following part is similar to P-GCN

        if not self.test_mode:
        
            raw_comp_fc = raw_comp_fc.view(batch_size, -1, raw_comp_fc.size()[-1])[:, :-1, :].contiguous()
            raw_comp_fc = raw_comp_fc.view(-1, raw_comp_fc.size()[-1])

            comp_target = target[:, :-1].contiguous().view(-1).data

            # keep the target proposal
            type_data = prop_type.view(-1).data
            target = target.view(-1)

            act_indexer = (type_data == 0) + (type_data == 2)

            reg_target = reg_target.view(-1, 2)
            reg_indexer = (type_data == 0)

            out = raw_act_fc[act_indexer, :], target[act_indexer], type_data[act_indexer], \
                raw_comp_fc, comp_target, \
                raw_regress_fc[reg_indexer, :, :], target[reg_indexer], reg_target[reg_indexer, :]

            return out
        else:
            return raw_act_fc, raw_comp_fc, raw_regress_fc

if __name__ == '__main__':
    model_configs = dict(
        num_class=20,
        feat_dim=1024,
        act_net_dims=[2048, 384],
        comp_net_dims=[4096, 384],
        dropout=0.8,
        roi_scale=0.125
    )

    model = TwoStageDetector(model_configs, 1024)
