'''
reference
1. seblock: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
2. sev2: https://github.com/mahendran-narayanan/SENetV2-Aggregated-dense-layer-for-channelwise-and-global-representations/blob/main/se_v2.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from collections import OrderedDict
import os
from pathlib import Path
from contextlib import redirect_stdout

class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels=64, reduction=16, out_channels=64, name_base = ''):
        super().__init__()
        self._pool = nn.Sequential(OrderedDict([((name_base+'_se_pool'), nn.AdaptiveAvgPool2d(1))]))
        self._fc_module = nn.Sequential()
        self._fc_module.add_module((name_base+'_se_fc1'), nn.Linear(in_channels, in_channels // reduction))
        self._fc_module.add_module((name_base+'_se_relu'), nn.ReLU(inplace=True))
        self._fc_module.add_module((name_base+'_se_fc2'), nn.Linear(in_channels // reduction, out_channels))
        self._fc_module.add_module((name_base+'_se_sigmoid'), nn.Sigmoid())
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self._pool(x).view(b, c)
        y = self._fc_module(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == "__main__":
    folder_root = './info'
    model_name = 'SEBlock'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = squeeze_excitation_block(in_channels=64, reduction=16, out_channels=64).to(device)
    file_path = Path(os.path.join(folder_root, (model_name+'.txt')))
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            summary(model, (64, 176, 40))
    summary(model, (64, 176, 40))