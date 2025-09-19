import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import os
from pathlib import Path
from contextlib import redirect_stdout

from model.DenseBlock import dense_block, dense_block_1, residual_dense_block
from model.SEBlock import squeeze_excitation_block

# from DenseBlock import dense_block, dense_block_1, residual_dense_block
# from SEBlock import squeeze_excitation_block

class DenseUNet(nn.Module):
    def __init__(self, model_name="denseunet", conf=None, device=None):
        super().__init__()
        self.model_name = model_name
        self.channels = 64
        if conf == None:
            self.denseblock_num = 3
            self.activation = 'relu'
            self.init_weights = False
        else:
            self.denseblock_num = conf["Model"]["denseblock_num"]
            self.activation = conf["Model"]["activation"]
            self.init_weights = conf["Model"]["init_weight"]
        
        # encoder
        self.layer1_e_conv_block = self._conv_block(in_channels=1, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')
        self.layer1_e_dense_block = dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense')
        self.layer1_down_sample = self._downsample_block(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer1_e')

        self.layer2_e_conv_block = self._conv_block(in_channels=self.channels, out_channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e')
        self.layer2_e_dense_block = dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense')
        self.layer2_down_sample = self._downsample_block(in_channels=2*self.channels, out_channels=2*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer2_e')

        self.layer3_e_conv_block = self._conv_block(in_channels=2*self.channels, out_channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e')
        self.layer3_e_dense_block = dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense')
        self.layer3_down_sample = self._downsample_block(in_channels=4*self.channels, out_channels=4*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer3_e')

        self.layer4_e_conv_block = nn.Conv2d(in_channels = 4*self.channels, out_channels = 8*self.channels, kernel_size = 3, stride = 1, padding = 1)
        self.layer4_e_activation = self._activate_fn(activation=self.activation)

        # decoder
        self.layer4_d_dconv_block = nn.ConvTranspose2d(in_channels = 8*self.channels, out_channels = 8*self.channels, kernel_size = 3, stride = 1, padding = (1, 1))
        self.layer4_d_activation = self._activate_fn(activation=self.activation)
        self.layer4_up_sample = self._upsample_block(in_channels=8*self.channels, out_channels=4*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer4_d')

        self.layer3_d_dense_block = dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense')
        self.layer3_d_dconv_block = self._dconv_block(in_channels=4*self.channels, out_channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d')
        self.layer3_up_sample = self._upsample_block(in_channels=4*self.channels, out_channels=2*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer3_d')
    
        self.layer2_d_dense_block = dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense')
        self.layer2_d_dconv_block = self._dconv_block(in_channels=2*self.channels, out_channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d')
        self.layer2_up_sample = self._upsample_block(in_channels=2*self.channels, out_channels=self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer2_d')

        self.layer1_d_dense_block = dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense')
        self.layer1_d_dconv_block = self._dconv_block(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d')
        
        self.layer1_d_conv_block = nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=1, stride=1)
        self.layer1_d_activation = self._activate_fn(activation="sigmoid")
        
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(x)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)
        layer4_e_conv_block = self.layer4_e_activation(layer4_e_conv_block)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_d_dconv_block = self.layer4_d_activation(layer4_d_dconv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        layer1_d_conv_block = self.layer1_d_conv_block(layer1_d_dconv_block)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block

    def _conv_block(self, in_channels=64, out_channels=64, kernel_size=4, stride=1, activation='relu', name_base=''):
        conv_block = nn.Sequential()
        conv_block.add_module((name_base+"_conv1"), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same'))
        conv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        conv_block.add_module((name_base+"_conv2"), nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same'))
        conv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))
        return conv_block
    
    def _downsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base=''):
        downsample_block = nn.Sequential()
        downsample_block.add_module((name_base+'_downsample'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        return downsample_block
    
    def _upsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base=''):
        upsample_block = nn.Sequential()
        upsample_block.add_module((name_base+'_upsample'), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        return upsample_block
    
    def _dconv_block(self, in_channels=64, out_channels=64, kernel_size=4, stride=1, activation='relu', name_base=''):
        dconv_block = nn.Sequential()
        dconv_block.add_module((name_base+"_dconv1"), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 1))
        dconv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        dconv_block.add_module((name_base+"_dconv2"), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 1))
        dconv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))
        # dconv_block.add_module((name_base+"_dconv1"), Multiscale_Transpose_Conv(in_channels=in_channels, out_channels=out_channels, stride=stride))
        # dconv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        # dconv_block.add_module((name_base+"_dconv2"), Multiscale_Transpose_Conv(in_channels=in_channels, out_channels=out_channels, stride=stride))
        # dconv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))
        return dconv_block
           
    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        elif (activation == "mish"):
            return torch.nn.Mish()
        else:
            return nn.Sigmoid()
        
    def _initialize_weights(self):
        if self.activation == 'gelu' or self.activation == 'mish' :
            activate_fn = 'relu'
        else:
            activate_fn = self.activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activate_fn)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activate_fn)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DenseUNet_SEBlock(DenseUNet):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, conf, device)
        self.seblock_pos = seblock_pos
        self._layer1_e_se_block = squeeze_excitation_block(in_channels=self.channels, out_channels=self.channels, name_base='layer1_e')
        self._layer2_e_se_block = squeeze_excitation_block(in_channels=2*self.channels, out_channels=2*self.channels, name_base='layer2_e')
        self._layer3_e_se_block = squeeze_excitation_block(in_channels=4*self.channels, out_channels=4*self.channels, name_base='layer3_e')
        self._layer3_d_se_block = squeeze_excitation_block(in_channels=4*self.channels, out_channels=4*self.channels, name_base='layer3_d')
        self._layer2_d_se_block = squeeze_excitation_block(in_channels=2*self.channels, out_channels=2*self.channels, name_base='layer2_d')
        self._layer1_d_se_block = squeeze_excitation_block(in_channels=self.channels, out_channels=self.channels, name_base='layer1_d')
    
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(x)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer1_e_se_block = self._layer1_e_se_block(layer1_e_dense_block)
            layer1_down_sample = self.layer1_down_sample(layer1_e_se_block)
        else:
            layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer2_e_se_block = self._layer2_e_se_block(layer2_e_dense_block)
            layer2_down_sample = self.layer2_down_sample(layer2_e_se_block)
        else:
            layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer3_e_se_block = self._layer3_e_se_block(layer3_e_dense_block)
            layer3_down_sample = self.layer3_down_sample(layer3_e_se_block)
        else:
            layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)
        layer4_e_conv_block = self.layer4_e_activation(layer4_e_conv_block)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_d_dconv_block = self.layer4_d_activation(layer4_d_dconv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer3_d_se_block = self._layer3_d_se_block(layer3_d_dense_block)
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_se_block)
        else:
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer2_d_se_block = self._layer2_d_se_block(layer2_d_dense_block)
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_se_block)
        else:
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer1_d_se_block = self._layer1_d_se_block(layer1_d_dense_block)
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_se_block)
        else:
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        layer1_d_conv_block = self.layer1_d_conv_block(layer1_d_dconv_block)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block
    
class DenseUNet_SEBlock_lastlayer(DenseUNet_SEBlock):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        # encoder
        self.layer4_e_conv_block = self._conv_block(in_channels=4*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_e')

        # decoder
        self.layer4_d_dconv_block = self._dconv_block(in_channels=8*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_d')
        
        if self.init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(x)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer1_e_se_block = self._layer1_e_se_block(layer1_e_dense_block)
            layer1_down_sample = self.layer1_down_sample(layer1_e_se_block)
        else:
            layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer2_e_se_block = self._layer2_e_se_block(layer2_e_dense_block)
            layer2_down_sample = self.layer2_down_sample(layer2_e_se_block)
        else:
            layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer3_e_se_block = self._layer3_e_se_block(layer3_e_dense_block)
            layer3_down_sample = self.layer3_down_sample(layer3_e_se_block)
        else:
            layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer3_d_se_block = self._layer3_d_se_block(layer3_d_dense_block)
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_se_block)
        else:
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer2_d_se_block = self._layer2_d_se_block(layer2_d_dense_block)
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_se_block)
        else:
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer1_d_se_block = self._layer1_d_se_block(layer1_d_dense_block)
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_se_block)
        else:
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        layer1_d_conv_block = self.layer1_d_conv_block(layer1_d_dconv_block)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block

    
class DenseUNet_SEBlock_Plusv2(DenseUNet_SEBlock):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        self.BFM = self._BFM_layer(in_channels=1, out_channels=64, conv_channels=32, kernel_size=3, name_base='layer1_e')
        self.layer1_e_conv_block = self._conv_block(in_channels=64, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')
        
        if self.init_weights:
            self._initialize_weights()
        
    def _BFM_layer(self, in_channels=1, out_channels=128, conv_channels=64, kernel_size=3, dilation=[3, 2], name_base=''):
        BFM_module = nn.Sequential()
        BFM_module.add_module((name_base+'BFM_dilated1'), nn.Conv2d(in_channels = in_channels, out_channels = 2 * conv_channels, kernel_size = kernel_size, dilation = dilation[0], padding='same'))
        BFM_module.add_module((name_base+'BFM_dilated2'), nn.Conv2d(in_channels = 2 * conv_channels, out_channels = out_channels, kernel_size = kernel_size, dilation = dilation[1], padding='same'))
        return BFM_module
        
    def forward(self, x):
        # BFM
        BFM_module = self.BFM(x)
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(BFM_module)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer1_e_se_block = self._layer1_e_se_block(layer1_e_dense_block)
            layer1_down_sample = self.layer1_down_sample(layer1_e_se_block)
        else:
            layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer2_e_se_block = self._layer2_e_se_block(layer2_e_dense_block)
            layer2_down_sample = self.layer2_down_sample(layer2_e_se_block)
        else:
            layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer3_e_se_block = self._layer3_e_se_block(layer3_e_dense_block)
            layer3_down_sample = self.layer3_down_sample(layer3_e_se_block)
        else:
            layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)
        layer4_e_conv_block = self.layer4_e_activation(layer4_e_conv_block)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_d_dconv_block = self.layer4_d_activation(layer4_d_dconv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer3_d_se_block = self._layer3_d_se_block(layer3_d_dense_block)
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_se_block)
        else:
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer2_d_se_block = self._layer2_d_se_block(layer2_d_dense_block)
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_se_block)
        else:
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer1_d_se_block = self._layer1_d_se_block(layer1_d_dense_block)
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_se_block)
        else:
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        # feature fusion: BFM + layer1_d_dconv_block
        feature_fusion = torch.add(BFM_module, layer1_d_dconv_block)
        layer1_d_conv_block = self.layer1_d_conv_block(feature_fusion)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block
    
class DenseUNet_SEBlock_lastlayer_Plusv2(DenseUNet_SEBlock):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        self.BFM = self._BFM_layer(in_channels=1, out_channels=64, conv_channels=32, kernel_size=3, name_base='layer1_e')
        self.layer1_e_conv_block = self._conv_block(in_channels=64, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')
        self.layer4_e_conv_block = self._conv_block(in_channels=4*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_e')

        self.layer4_d_dconv_block = self._dconv_block(in_channels=8*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_d')
        
        if self.init_weights:
            self._initialize_weights()
        
        
    def _BFM_layer(self, in_channels=1, out_channels=128, conv_channels=64, kernel_size=3, dilation=[3, 2], name_base=''):
        BFM_module = nn.Sequential()
        BFM_module.add_module((name_base+'BFM_dilated1'), nn.Conv2d(in_channels = in_channels, out_channels = 2 * conv_channels, kernel_size = kernel_size, dilation = dilation[0], padding='same'))
        BFM_module.add_module((name_base+'BFM_dilated2'), nn.Conv2d(in_channels = 2 * conv_channels, out_channels = out_channels, kernel_size = kernel_size, dilation = dilation[1], padding='same'))
        return BFM_module
    
    def forward(self, x):
        # BFM
        BFM_module = self.BFM(x)
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(BFM_module)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer1_e_se_block = self._layer1_e_se_block(layer1_e_dense_block)
            layer1_down_sample = self.layer1_down_sample(layer1_e_se_block)
        else:
            layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer2_e_se_block = self._layer2_e_se_block(layer2_e_dense_block)
            layer2_down_sample = self.layer2_down_sample(layer2_e_se_block)
        else:
            layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer3_e_se_block = self._layer3_e_se_block(layer3_e_dense_block)
            layer3_down_sample = self.layer3_down_sample(layer3_e_se_block)
        else:
            layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer3_d_se_block = self._layer3_d_se_block(layer3_d_dense_block)
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_se_block)
        else:
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer2_d_se_block = self._layer2_d_se_block(layer2_d_dense_block)
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_se_block)
        else:
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer1_d_se_block = self._layer1_d_se_block(layer1_d_dense_block)
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_se_block)
        else:
            layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        # feature fusion: BFM + layer1_d_dconv_block
        feature_fusion = BFM_module + layer1_d_dconv_block
        layer1_d_conv_block = self.layer1_d_conv_block(feature_fusion)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block
  
  
class DenseUNet_SEBlock_lastlayer_plusv2_updown(DenseUNet_SEBlock_lastlayer_Plusv2):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
    def _downsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base='', dropout_rate = 0.2):
        downsample_block = nn.Sequential()
        downsample_block.add_module((name_base+'_downsample_conv_s2'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        downsample_block.add_module((name_base+'_downsample_'+self.activation), self._activate_fn(activation=activation))
        downsample_block.add_module((name_base+'_downsample_drop'), nn.Dropout(p=dropout_rate))
        downsample_block.add_module((name_base+'_downsample_norm'), nn.BatchNorm2d(num_features=out_channels))
        return downsample_block
    
    def _upsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base='', dropout_rate = 0.2):
        upsample_block = nn.Sequential()
        upsample_block.add_module((name_base+'_upsample_dconv_s2'), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        upsample_block.add_module((name_base+'_upsample_'+self.activation), self._activate_fn(activation=activation))
        upsample_block.add_module((name_base+'_upsample_drop'), nn.Dropout(p=dropout_rate))
        upsample_block.add_module((name_base+'_upsample_norm'), nn.BatchNorm2d(num_features=out_channels))
        return upsample_block
    
class DenseUNet_SEBlock_Plusv2_denseblockv1(DenseUNet_SEBlock_Plusv2):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        self.layer1_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense_se')

        self.layer2_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense_se')

        self.layer3_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense_se')

        self.layer3_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense_se')
    
        self.layer2_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense_se')

        self.layer1_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense_se')
        
        if self.init_weights:
            self._initialize_weights()
    
    
class DenseUNet_1(DenseUNet_SEBlock):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        self.layer1_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense_se')

        self.layer2_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense_se')

        self.layer3_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense_se')

        self.layer3_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense_se')
    
        self.layer2_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense_se')

        self.layer1_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense_se')
        
        if self.init_weights:
            self._initialize_weights()
    
class Residual_Dense_UNet(nn.Module):
    def __init__(self, model_name="denseunet", conf=None, device=None):
        super().__init__()
        self.model_name = model_name
        self.channels = 64
        if conf == None:
            self.denseblock_num = 6
            self.activation = 'relu'
        else:
            self.denseblock_num = conf["Model"]["denseblock_num"]
            self.activation = conf["Model"]["activation"]

        # encoder
        self.layer1_e_conv_block = self._conv_block(in_channels=1, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')
        self.layer1_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense')
        self.layer1_down_sample = self._downsample_block(in_channels=self.channels, out_channels=self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer1_e')

        self.layer2_e_conv_block = self._conv_block(in_channels=self.channels, out_channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e')
        self.layer2_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense')
        self.layer2_down_sample = self._downsample_block(in_channels=2*self.channels, out_channels=2*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer2_e')

        self.layer3_e_conv_block = self._conv_block(in_channels=2*self.channels, out_channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e')
        self.layer3_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense')
        self.layer3_down_sample = self._downsample_block(in_channels=4*self.channels, out_channels=4*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer3_e')

        self.layer4_e_conv_block = nn.Conv2d(in_channels = 4*self.channels, out_channels = 8*self.channels, kernel_size = 3, stride = 1, padding = 1)
        self.layer4_e_activation = self._activate_fn(activation=self.activation)

        # decoder
        self.layer4_d_dconv_block = nn.ConvTranspose2d(in_channels = 8*self.channels, out_channels = 8*self.channels, kernel_size = 3, stride = 1, padding = (1, 1))
        self.layer4_d_activation = self._activate_fn(activation=self.activation)
        self.layer4_up_sample = self._upsample_block(in_channels=8*self.channels, out_channels=4*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer4_d')

        self.layer3_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense')
        self.layer3_d_dconv_block = self._dconv_block(in_channels=4*self.channels, out_channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d')
        self.layer3_up_sample = self._upsample_block(in_channels=4*self.channels, out_channels=2*self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer3_d')
    
        self.layer2_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense')
        self.layer2_d_dconv_block = self._dconv_block(in_channels=2*self.channels, out_channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d')
        self.layer2_up_sample = self._upsample_block(in_channels=2*self.channels, out_channels=self.channels, kernel_size=2, stride=2, activation=self.activation, name_base='layer2_d')

        self.layer1_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense')
        self.layer1_d_dconv_block = self._dconv_block(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d')
        
        self.layer1_d_conv_block = nn.Conv2d(in_channels=self.channels, out_channels=1, kernel_size=1, stride=1)
        self.layer1_d_activation = self._activate_fn(activation="sigmoid")

    def forward(self, x):
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(x)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)
        layer4_e_conv_block = self.layer4_e_activation(layer4_e_conv_block)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_d_dconv_block = self.layer4_d_activation(layer4_d_dconv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        layer1_d_conv_block = self.layer1_d_conv_block(layer1_d_dconv_block)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block

    def _conv_block(self, in_channels=64, out_channels=64, kernel_size=4, stride=1, activation='relu', name_base=''):
        conv_block = nn.Sequential()
        conv_block.add_module((name_base+"_conv1"), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same'))
        conv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        conv_block.add_module((name_base+"_conv2"), nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 'same'))
        conv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))
        return conv_block
    
    def _downsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base=''):
        downsample_block = nn.Sequential()
        downsample_block.add_module((name_base+'_downsample'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        return downsample_block
    
    def _upsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base=''):
        upsample_block = nn.Sequential()
        upsample_block.add_module((name_base+'_upsample'), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        return upsample_block
    
    def _dconv_block(self, in_channels=64, out_channels=64, kernel_size=4, stride=1, activation='relu', name_base=''):
        dconv_block = nn.Sequential()
        dconv_block.add_module((name_base+"_dconv1"), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 1))
        dconv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        dconv_block.add_module((name_base+"_dconv2"), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = 1))
        dconv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))
        # dconv_block.add_module((name_base+"_dconv1"), Multiscale_Transpose_Conv(in_channels=in_channels, out_channels=out_channels, stride=stride))
        # dconv_block.add_module((name_base+"_"+ activation + "1"), self._activate_fn(activation=activation))
        # dconv_block.add_module((name_base+"_dconv2"), Multiscale_Transpose_Conv(in_channels=in_channels, out_channels=out_channels, stride=stride))
        # dconv_block.add_module((name_base+"_"+ activation + "2"), self._activate_fn(activation=activation))
        return dconv_block
           
    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        else:
            return nn.Sigmoid()
        
class Residual_Dense_UNet_v1(DenseUNet_SEBlock):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        # encoder
        self.BFM = self._BFM_layer(in_channels=1, out_channels=64, conv_channels=32, kernel_size=3, name_base='layer1_e')
        self.layer1_e_conv_block = self._conv_block(in_channels=64, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')
        self.layer1_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense', add_rate=0.5)
        self.layer2_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense', add_rate=0.5)
        self.layer3_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense', add_rate=0.5)
        self.layer4_e_conv_block = self._conv_block(in_channels=4*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_e')

        # decoder
        self.layer4_d_dconv_block = self._dconv_block(in_channels=8*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_d')
        self.layer3_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense', add_rate=0.5)
        self.layer2_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense', add_rate=0.5)
        self.layer1_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense', add_rate=0.5)

        if self.init_weights:
            self._initialize_weights()
            
    def _BFM_layer(self, in_channels=1, out_channels=128, conv_channels=64, kernel_size=3, dilation=[3, 2], name_base=''):
        BFM_module = nn.Sequential()
        BFM_module.add_module((name_base+'BFM_dilated1'), nn.Conv2d(in_channels = in_channels, out_channels = 2 * conv_channels, kernel_size = kernel_size, dilation = dilation[0], padding='same'))
        BFM_module.add_module((name_base+'BFM_dilated2'), nn.Conv2d(in_channels = 2 * conv_channels, out_channels = out_channels, kernel_size = kernel_size, dilation = dilation[1], padding='same'))
        return BFM_module
    
    def _downsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base='', dropout_rate = 0.2):
        downsample_block = nn.Sequential()
        downsample_block.add_module((name_base+'_downsample_conv_s2'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        downsample_block.add_module((name_base+'_downsample_'+self.activation), self._activate_fn(activation=activation))
        downsample_block.add_module((name_base+'_downsample_drop'), nn.Dropout(p=dropout_rate))
        downsample_block.add_module((name_base+'_downsample_norm'), nn.BatchNorm2d(num_features=out_channels))
        return downsample_block
    
    def _upsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base='', dropout_rate = 0.2):
        upsample_block = nn.Sequential()
        upsample_block.add_module((name_base+'_upsample_dconv_s2'), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        upsample_block.add_module((name_base+'_upsample_'+self.activation), self._activate_fn(activation=activation))
        upsample_block.add_module((name_base+'_upsample_drop'), nn.Dropout(p=dropout_rate))
        upsample_block.add_module((name_base+'_upsample_norm'), nn.BatchNorm2d(num_features=out_channels))
        return upsample_block
            
    def forward(self, x):
        # BFM
        BFM_module = self.BFM(x)
        # encoder
        layer1_e_conv_block = self.layer1_e_conv_block(BFM_module)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block)
        layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)

        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)

        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)

        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)

        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        layer4_up_sample = self.layer4_up_sample(layer4_d_dconv_block)

        layer3_add = torch.add(layer3_e_dense_block, layer4_up_sample)
        layer3_d_dense_block = self.layer3_d_dense_block(layer3_add)
        layer3_d_se_block = self._layer3_d_se_block(layer3_d_dense_block)
        layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_se_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        layer2_d_se_block = self._layer2_d_se_block(layer2_d_dense_block)
        layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_se_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        layer1_d_se_block = self._layer1_d_se_block(layer1_d_dense_block)
        layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_se_block)

        # feature fusion: BFM + layer1_d_dconv_block
        feature_fusion = BFM_module + layer1_d_dconv_block
        layer1_d_conv_block = self.layer1_d_conv_block(feature_fusion)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block
    
class Residual_Dense_UNet_v1_updown(Residual_Dense_UNet_v1):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        if self.init_weights:
            self._initialize_weights()
        
    def _downsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base='', dropout_rate = 0.2):
        downsample_block = nn.Sequential()
        downsample_block.add_module((name_base+'_downsample_conv_s2'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        downsample_block.add_module((name_base+'_downsample_'+self.activation), self._activate_fn(activation=activation))
        downsample_block.add_module((name_base+'_downsample_drop'), nn.Dropout(p=dropout_rate))
        downsample_block.add_module((name_base+'_downsample_norm'), nn.BatchNorm2d(num_features=out_channels))
        return downsample_block
    
    def _upsample_block(self, in_channels=64, out_channels=64, kernel_size=2, stride=2, activation='relu', name_base='', dropout_rate = 0.2):
        upsample_block = nn.Sequential()
        upsample_block.add_module((name_base+'_upsample_dconv_s2'), nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride))
        upsample_block.add_module((name_base+'_upsample_'+self.activation), self._activate_fn(activation=activation))
        upsample_block.add_module((name_base+'_upsample_drop'), nn.Dropout(p=dropout_rate))
        upsample_block.add_module((name_base+'_upsample_norm'), nn.BatchNorm2d(num_features=out_channels))
        return upsample_block
        
if __name__ == "__main__":
    folder_root = './info'
    model_name = 'DenseUNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseUNet().to(device)
    print(model)
    file_path = Path(os.path.join(folder_root, (model_name+'.txt')))
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            summary(model, (1, 176, 40))
    summary(model, (1, 176, 40))