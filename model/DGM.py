import math
import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.utils import _pair

class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, extra_offset_mask=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.with_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(self.in_channels * 2, 3 * self.kernel_size[0] * self.kernel_size[1], kernel_size=self.kernel_size, stride=stride, padding=padding, bias=True)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):
        feat_degradation = torch.cat([input_feat, inter], dim=1)

        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(input_feat.contiguous(), offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)

class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            self._activate_fn(activation='gelu'),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            self._activate_fn(activation='gelu'),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        '''
        :param x: degradation representation: B * C
        :param inter: degradation intermediate representation map: B * C * H * W
        '''
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)

        return x * gamma + beta

    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == 'leakyrelu'):
            return nn.LeakyReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        elif (activation == "mish"):
            return torch.nn.Mish()
        else:
            return nn.Sigmoid()
    

class DGM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(DGM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.dcn = DCN_layer(self.channels_in, self.channels_out, kernel_size,
                             padding=(kernel_size - 1) // 2, bias=False)
        self.sft = SFT_layer(self.channels_in, self.channels_out)

    def forward(self, x, inter):
        '''
        :param x: feature map: B * C * H * W
        :inter: degradation map: B * C * H * W
        '''
        dcn_out = self.dcn(x, inter)
        sft_out = self.sft(x, inter)
        out = dcn_out + sft_out
        out = x + out

        return out

class DGB(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1):
        super(DGB, self).__init__()

        # self.da_conv1 = DGM(n_feat, n_feat, kernel_size)
        # self.da_conv2 = DGM(n_feat, n_feat, kernel_size)
        self.dgm1 = DGM(in_channels, out_channels, kernel_size)
        self.dgm2 = DGM(out_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same')
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same')

        self.activation = self._activate_fn(activation='gelu')

    def forward(self, x, degradation_representation):
        '''
        :param x: feature map: B * C * H * W
        :param wet_quality_embeddings: degradation representation: B * C * H * W
        '''
        out = self.activation(self.dgm1(x, degradation_representation))
        out = self.activation(self.conv1(out))
        out = self.activation(self.dgm2(out, degradation_representation))
        out = self.conv2(out) + x

        return out
    
    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == 'leakyrelu'):
            return nn.LeakyReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        elif (activation == "mish"):
            return torch.nn.Mish()
        else:
            return nn.Sigmoid()
        
class DGB_Module(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, activation='relu', name_base=''):
        super(DGB_Module, self).__init__()

        self.dgb = DGB(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.conv_block = nn.Sequential()
        self.conv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        self.conv_block.add_module((name_base+"_conv2"), nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same'))
        self.conv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))

    def forward(self, x, degradation_representation):
        dgb_feat = self.dgb(x, degradation_representation)
        fusion_feat = self.conv_block(dgb_feat)
        return fusion_feat

    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == 'leakyrelu'):
            return nn.LeakyReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        elif (activation == "mish"):
            return torch.nn.Mish()
        else:
            return nn.Sigmoid()
