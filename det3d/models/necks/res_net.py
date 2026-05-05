import numpy as np
import torch
from torch import nn

from det3d.models.utils import Sequential
from ..registry import NECKS
from ..utils import build_norm_layer

class BasicBlock(nn.Module):
    """Standard ResNet Basic Block adapted for det3d."""
    def __init__(
        self, 
        inplanes, 
        planes, 
        stride=1, 
        norm_cfg=None):
        super(BasicBlock, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
            
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, planes)[1],
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

@NECKS.register_module
class ResNetNeck(nn.Module):
    def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters, num_input_features, norm_cfg=None, logger=None, **kwargs):
        super(ResNetNeck, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            # Build ResNet stage
            stage_blocks = []
            stage_blocks.append(BasicBlock(in_filters[i], self._num_filters[i], stride=self._layer_strides[i], norm_cfg=self._norm_cfg))
            for _ in range(1, layer_num):
                stage_blocks.append(BasicBlock(self._num_filters[i], self._num_filters[i], stride=1, norm_cfg=self._norm_cfg))
            
            blocks.append(Sequential(*stage_blocks))
            num_out_filters = self._num_filters[i]

            # Build Upsampling (FPN) stage - identical to your RPN
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False),
                        build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx])[1],
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = Sequential(
                        nn.Conv2d(num_out_filters, self._num_upsample_filters[i - self._upsample_start_idx], stride, stride=stride, bias=False),
                        build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx])[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
                
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        
        if logger:
            logger.info("Finish ResNetNeck Initialization")

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x