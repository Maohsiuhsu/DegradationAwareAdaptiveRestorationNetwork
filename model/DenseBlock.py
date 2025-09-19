import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# from timm.layers import DropPath
from torchviz import make_dot

import os
from pathlib import Path
from contextlib import redirect_stdout

class dense_conv_block(nn.Module):
    def __init__(self, in_channels=64, growth_rate = 64, kernel_size=3, stride=1, name_base = '', activation = 'relu', dropout_rate = 0.2, block=0):
        super().__init__()
        self._in_channels = in_channels
        self._growth_rate = growth_rate
        self._kernel_size = kernel_size
        self._stride = stride
        self._name_base = name_base
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._block = block
        self._dense_conv_block_module = self._dense_conv_block()

    def forward(self, x):
        dense_output = self._dense_conv_block_module(x)
        x = torch.cat([x, dense_output], dim=1)
        return x

    def _dense_conv_block(self):
        dense_conv_block = nn.Sequential()
        dense_conv_block.add_module((self._name_base+'_norm'+str(self._block)), nn.BatchNorm2d(num_features=self._in_channels))
        dense_conv_block.add_module((self._name_base+"_"+self._activation+str(self._block)), self._activate_fn(activation=self._activation))
        dense_conv_block.add_module((self._name_base+'_conv'+str(self._block)), nn.Conv2d(in_channels = self._in_channels, out_channels = self._growth_rate, kernel_size = self._kernel_size, stride = self._stride, padding='same'))
        dense_conv_block.add_module((self._name_base+'_drop'+str(self._block)), nn.Dropout(p=self._dropout_rate))
        return dense_conv_block
    
    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        else:
            return nn.Sigmoid()   
        
class dense_block(nn.Module):
    def __init__(self, block_num = 1, channels = 64, kernel_size=3, stride=1, name_base = '', activation = 'relu', drop_rate = 0.2):
        super().__init__()
        self._block_num = block_num
        self._channels = channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._name_base = name_base
        self._activation = activation
        self._drop_rate = drop_rate

        self._model = nn.Sequential()
        for block_id in range(self._block_num):
            channels_num = (block_id+1)*self._channels
            dense_block_module = dense_conv_block(in_channels=channels_num, growth_rate=self._channels, kernel_size=self._kernel_size, stride=self._stride, activation=self._activation, dropout_rate=self._drop_rate, name_base=self._name_base, block=block_id)
            self._model.add_module((name_base+'dense_conv_block'+str(block_id)), dense_block_module)

        dense_output_block = self._dense_output_module(in_channels=(block_num+1)*self._channels, out_channels=self._channels, kernel_size=self._kernel_size, stride=self._stride, activation=self._activation, name_base=self._name_base)
        self._model.add_module((name_base+'dense_output_block'), dense_output_block)

    def forward(self, x):
        x = self._model(x)
        return x

    def _dense_output_module(self, in_channels=128, out_channels=64, kernel_size=3, stride=1, activation='relu', name_base = ''):
        dense_output_module = nn.Sequential()
        dense_output_module.add_module((name_base+'_norm_last'), nn.BatchNorm2d(num_features=in_channels))
        dense_output_module.add_module((name_base+'_'+activation+'_last'), self._activate_fn(activation=activation))
        dense_output_module.add_module((name_base+'_conv_last'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding='same'))
        return dense_output_module
    
    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        else:
            return nn.Sigmoid()
       

class dense_conv_block_1(dense_conv_block):
    def __init__(self, in_channels=64, growth_rate=64, kernel_size=3, stride=1, name_base='', activation='relu', dropout_rate=0.2, block=0):
        super().__init__(in_channels, growth_rate, kernel_size, stride, name_base, activation, dropout_rate, block)
        
    def _dense_conv_block(self):
        dense_conv_block = nn.Sequential()
        dense_conv_block.add_module((self._name_base+'_norm'+str(self._block)), nn.BatchNorm2d(num_features=self._in_channels))
        dense_conv_block.add_module((self._name_base+"_"+self._activation+str(self._block)), self._activate_fn(activation=self._activation))
        dense_conv_block.add_module((self._name_base+'_conv1x1_1'+str(self._block)), nn.Conv2d(in_channels = self._in_channels, out_channels = self._growth_rate, kernel_size = 1, stride = self._stride, padding='same'))
        dense_conv_block.add_module((self._name_base+'_conv3x3'+str(self._block)), nn.Conv2d(in_channels = self._growth_rate, out_channels = self._growth_rate, kernel_size = self._kernel_size, stride = self._stride, padding='same'))
        dense_conv_block.add_module((self._name_base+'_conv1x1_2'+str(self._block)), nn.Conv2d(in_channels = self._growth_rate, out_channels = self._growth_rate, kernel_size = 1, stride = self._stride, padding='same'))
        dense_conv_block.add_module((self._name_base+'_drop'+str(self._block)), nn.Dropout(p=self._dropout_rate))
        return dense_conv_block
    
class dense_block_1(dense_block): 
    def __init__(self, block_num=1, channels=64, kernel_size=3, stride=1, name_base='', activation='relu', drop_rate=0.2):
        super().__init__(block_num, channels, kernel_size, stride, name_base, activation, drop_rate)
        self._model = nn.Sequential()
        for block_id in range(self._block_num):
            channels_num = (block_id+1)*self._channels
            dense_block_module = dense_conv_block_1(in_channels=channels_num, growth_rate=self._channels, kernel_size=self._kernel_size, stride=self._stride, activation=self._activation, dropout_rate=self._drop_rate, name_base=self._name_base, block=block_id)
            self._model.add_module((name_base+'dense_conv_block'+str(block_id)), dense_block_module)

        dense_output_block = self._dense_output_module(in_channels=(block_num+1)*self._channels, out_channels=self._channels, kernel_size=self._kernel_size, stride=self._stride, activation=self._activation, name_base=self._name_base)
        self._model.add_module((name_base+'dense_output_block'), dense_output_block)
        
class residual_dense_block(nn.Module):
    def __init__(self, block_num=1, channels=64, kernel_size=3, stride=1, name_base='', activation='relu', drop_rate=0.2, add_rate=0.5):
        super().__init__()
        self._block_num = block_num
        self._channels = channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._name_base = name_base
        self._activation = activation
        self._drop_rate = drop_rate
        self._add_rate = add_rate
        self._input_pointwise_cov_layer = nn.Sequential()
        self._input_pointwise_cov_layer.add_module((name_base + "_pointwise_conv_" + name_base), nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = (1, 1), padding='same'))

        self._model = nn.Sequential()
        for block_id in range(self._block_num):
            channels_num = (block_id+1)*self._channels
            dense_block_module = dense_conv_block_1(in_channels=channels_num, growth_rate=self._channels, kernel_size=self._kernel_size, stride=self._stride, activation=self._activation, dropout_rate=self._drop_rate, name_base=self._name_base, block=block_id)
            self._model.add_module((name_base+'dense_conv_block_1'+str(block_id)), dense_block_module)

        dense_output_block = self._dense_output_module(in_channels=(block_num+1)*self._channels, out_channels=self._channels, kernel_size=self._kernel_size, stride=self._stride, activation=self._activation, name_base=self._name_base)
        self._model.add_module((name_base+'dense_output_block'), dense_output_block)

    def forward(self, x):
        input_x = self._input_pointwise_cov_layer(x)
         
        dense_feature = self._model(x)
        
        output = torch.add(dense_feature, input_x, alpha=self._add_rate)
        return output

    def _dense_output_module(self, in_channels=128, out_channels=64, kernel_size=3, stride=1, activation='relu', name_base = ''):
        dense_output_module = nn.Sequential()
        dense_output_module.add_module((name_base+'_norm_last'), nn.BatchNorm2d(num_features=in_channels))
        dense_output_module.add_module((name_base+'_'+activation+'_last'), self._activate_fn(activation=activation))
        dense_output_module.add_module((name_base+'_conv_last'), nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding='same'))
        return dense_output_module
    
    def _activate_fn(self, activation):
        if (activation == "sigmoid"):
            return nn.Sigmoid()
        elif (activation == "relu"):
            return nn.ReLU()
        elif (activation == "gelu"):
            return nn.GELU()
        else:
            return nn.Sigmoid()
        
if __name__ == "__main__":
    folder_root = './info'
    model_name = 'DenseBlock'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = dense_block(block_num=6).to(device)
    file_path = Path(os.path.join(folder_root, (model_name+'.txt')))
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            summary(model, (64, 176, 40))
    summary(model, (64, 176, 40))