import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import os
from pathlib import Path
from contextlib import redirect_stdout
from thop import profile, clever_format

from model.DenseUNet import DenseUNet
from model.DenseBlock import dense_block_1, residual_dense_block
from model.SEBlock import squeeze_excitation_block
from model.CBDE import CBDE
from model.DGM import DGB_Module
from model.CAAPM import Topm_CrossAttention_Restormer, ch_shuffle_high_text
# from DenseUNet import DenseUNet
# from DenseBlock import dense_block_1, residual_dense_block
# from SEBlock import squeeze_excitation_block
# from CBDE import CBDE
# from DGM import DGB_Module
# from CAAPM import Topm_CrossAttention_Restormer, ch_shuffle_high_text


class DenseUNet_baseline(DenseUNet):
    def __init__(self, model_name="denseunet", conf=None, device=None):
        super().__init__(model_name, conf, device)
        
        self.layer1_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense_se')
        self.layer2_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense_se')
        self.layer3_e_dense_block = dense_block_1(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense_se')
        self.layer3_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense_se')
        self.layer2_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense_se')
        self.layer1_d_dense_block = dense_block_1(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense_se')
        
        self.layer4_e_conv_block = self._conv_block(in_channels=4*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_e')
        self.layer4_d_dconv_block = self._dconv_block(in_channels=8*self.channels, out_channels=8*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer4_d')
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
    
class DenseUNet_BFM(DenseUNet_baseline):
    def __init__(self, model_name="denseunet", conf=None, device=None):
        super().__init__(model_name, conf, device)
        
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
        layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)

        layer2_add = torch.add(layer2_e_dense_block, layer3_up_sample)
        layer2_d_dense_block = self.layer2_d_dense_block(layer2_add)
        layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        layer1_add = torch.add(layer1_e_dense_block, layer2_up_sample)
        layer1_d_dense_block = self.layer1_d_dense_block(layer1_add)
        layer1_d_dconv_block = self.layer1_d_dconv_block(layer1_d_dense_block)

        # feature fusion: BFM + layer1_d_dconv_block
        feature_fusion = BFM_module + layer1_d_dconv_block
        layer1_d_conv_block = self.layer1_d_conv_block(feature_fusion)
        layer1_d_conv_block = self.layer1_d_activation(layer1_d_conv_block)

        return layer1_d_conv_block
    
class DenseUNet_SEBlock_BFM(DenseUNet_BFM):
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
    
class Residual_DenseUNet_SEBlock_BFM(DenseUNet_SEBlock_BFM):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        self.layer1_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e_dense', add_rate=0.5)
        self.layer2_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_e_dense', add_rate=0.5)
        self.layer3_e_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_e_dense', add_rate=0.5)
        self.layer3_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=4*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer3_d_dense', add_rate=0.5)
        self.layer2_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=2*self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer2_d_dense', add_rate=0.5)
        self.layer1_d_dense_block = residual_dense_block(block_num=self.denseblock_num, channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_d_dense', add_rate=0.5)
        
        if self.init_weights:
            self._initialize_weights()






class Residual_DenseUNet_SEBlock_BFM_CAAPM(Residual_DenseUNet_SEBlock_BFM):

    
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 64,

        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        device = "cuda:0",
        # decoder = False,
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        model_name="denseunet", 
        seblock_pos="encoder", 
        conf=None
    ):
        super().__init__(model_name, seblock_pos, conf, device)
        self.layer1_e_dgb_block = DGB_Module(in_channels=64, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')
        
        self.device = device        
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 512),
        #     nn.Linear(512, 512),
        # )
        
        self.encoder_shuffle_channel1 = ch_shuffle_high_text(ch_dim = dim,num_heads=heads[0],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias,lin_ch=256) # encoder level1 shuffle
        self.encoder_shuffle_channel2 = ch_shuffle_high_text(ch_dim = int(dim*2**1),num_heads=heads[1],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias,lin_ch=256) # encoder level2 shuffle
        self.encoder_shuffle_channel3 = ch_shuffle_high_text(ch_dim = int(dim*2**2),num_heads=heads[2],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias,lin_ch=256) # encoder level3 shuffle  
        self.latent_shuffle_channel = ch_shuffle_high_text(ch_dim = int(dim*2**3),num_heads=heads[3],LayerNorm_type=LayerNorm_type,ffn_expansion_factor=ffn_expansion_factor,bias=bias,lin_ch=256) # latent latent shuffle

        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*2**0), kernel_size=1, bias=bias)
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        if self.init_weights:
            self._initialize_weights()


            
    def forward(self, x, degradation_representation, degradation_embedding):

        # BFM
        BFM_module = self.BFM(x)
        # encoder
        # layer1 
        layer1_e_conv_block_feat = self.layer1_e_dgb_block(BFM_module, degradation_representation)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block_feat)

        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer1_e_se_block = self._layer1_e_se_block(layer1_e_dense_block)
            layer1_down_sample = self.layer1_down_sample(layer1_e_se_block)
        else:
            layer1_e_se_block = layer1_e_dense_block
            layer1_down_sample = self.layer1_down_sample(layer1_e_dense_block)
        # layer2
        layer2_e_conv_block = self.layer2_e_conv_block(layer1_down_sample)
        layer2_e_dense_block = self.layer2_e_dense_block(layer2_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer2_e_se_block = self._layer2_e_se_block(layer2_e_dense_block)
            layer2_down_sample = self.layer2_down_sample(layer2_e_se_block)
        else:
            layer2_e_se_block = layer2_e_dense_block
            layer2_down_sample = self.layer2_down_sample(layer2_e_dense_block)
        # layer3
        layer3_e_conv_block = self.layer3_e_conv_block(layer2_down_sample)
        layer3_e_dense_block = self.layer3_e_dense_block(layer3_e_conv_block)
        if (self.seblock_pos == "encoder") or (self.seblock_pos == "both"):
            layer3_e_se_block = self._layer3_e_se_block(layer3_e_dense_block)
            layer3_down_sample = self.layer3_down_sample(layer3_e_se_block)
        else:
            layer3_e_se_block = layer3_e_dense_block
            layer3_down_sample = self.layer3_down_sample(layer3_e_dense_block)
        # layer4
        layer4_e_conv_block = self.layer4_e_conv_block(layer3_down_sample)


        # decoder
        layer4_d_dconv_block = self.layer4_d_dconv_block(layer4_e_conv_block)
        # DGPB4
        dgpb4_out, _ = self.latent_shuffle_channel(layer4_d_dconv_block, degradation_embedding)
        layer4_up_sample = self.layer4_up_sample(dgpb4_out)
        # DGPB3
        dgpb3_out, _  = self.encoder_shuffle_channel3(layer3_e_se_block, degradation_embedding)
        input_layer3 = torch.cat([layer4_up_sample, dgpb3_out], dim=1)
        input_layer3 = self.reduce_chan_level3(input_layer3)

        layer3_d_dense_block = self.layer3_d_dense_block(input_layer3)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer3_d_se_block = self._layer3_d_se_block(layer3_d_dense_block)
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_se_block)
        else:
            layer3_d_dconv_block = self.layer3_d_dconv_block(layer3_d_dense_block)
        layer3_up_sample = self.layer3_up_sample(layer3_d_dconv_block)
        # DGPB2
        dgpb2_out, _ = self.encoder_shuffle_channel2(layer2_e_se_block, degradation_embedding)
        input_layer2 = torch.cat([layer3_up_sample, dgpb2_out], dim=1)
        input_layer2 = self.reduce_chan_level2(input_layer2)

        layer2_d_dense_block = self.layer2_d_dense_block(input_layer2)
        if (self.seblock_pos == "decoder") or (self.seblock_pos == "both"):
            layer2_d_se_block = self._layer2_d_se_block(layer2_d_dense_block)
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_se_block)
        else:
            layer2_d_dconv_block = self.layer2_d_dconv_block(layer2_d_dense_block)
        layer2_up_sample = self.layer2_up_sample(layer2_d_dconv_block)

        # DGPB1
        dgpb1_out, _ = self.encoder_shuffle_channel1(layer1_e_se_block, degradation_embedding)
        input_layer1 = torch.cat([layer2_up_sample, dgpb1_out], dim=1)
        input_layer1 = self.reduce_chan_level1(input_layer1)

        layer1_d_dense_block = self.layer1_d_dense_block(input_layer1)
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



class Residual_DenseUNet_SEBlock_BFM_DGB(Residual_DenseUNet_SEBlock_BFM):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__(model_name, seblock_pos, conf, device)
        
        self.layer1_e_dgb_block = DGB_Module(in_channels=64, out_channels=self.channels, kernel_size=3, stride=1, activation=self.activation, name_base='layer1_e')

        if self.init_weights:
            self._initialize_weights()
            
    def forward(self, x, degradation_representation):
        # BFM
        BFM_module = self.BFM(x)
        # encoder
        layer1_e_conv_block_feat = self.layer1_e_dgb_block(x=BFM_module, degradation_representation=degradation_representation)
        layer1_e_dense_block = self.layer1_e_dense_block(layer1_e_conv_block_feat)
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
    
class All_in_One_Residual_DenseUNet(nn.Module):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__()
        if conf == None:
            self.activation = 'relu'
            self.init_weights = False
            self.degradation_encoder = CBDE(batch_size=16, device=device)
        else:
            self.activation = conf["Model"]["activation"]
            self.init_weights = conf["Model"]["init_weight"]
            self.degradation_encoder = CBDE(batch_size=conf['Model']['batch_size'], device=device)
        
       
        self.degradation_restorer = Residual_DenseUNet_SEBlock_BFM_CAAPM(model_name=model_name, seblock_pos=seblock_pos, conf=conf, device=device)
        
        if self.init_weights:
            self._initialize_weights()
            
    def forward(self, x_query, x_key):
        degradation_embedding, logits, labels, degradation_representation = self.degradation_encoder(x_query, x_key)
        
        restored = self.degradation_restorer(x_query, degradation_representation,degradation_embedding)

        return restored, logits, labels
        
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
                    
                    
if __name__ == "__main__":
    folder_root = './info'
    model_name = 'All_in_One_Residual_DenseUNet'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = All_in_One_Residual_DenseUNet().to(device)
    
    
    print(model)

    summary_file_path = Path(os.path.join(folder_root, (model_name+'.txt')))
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    with open(summary_file_path, 'w') as f:
        with redirect_stdout(f):
            summary(model, input_size = [(1, 88, 88), (1, 88, 88)])
    summary(model, input_size = [(1, 88, 88 ), (1, 88, 88)])
    
    # x = torch.randn((1, 1, 176, 40), dtype=torch.float, requires_grad=False)
    # make_dot(model(x), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(filename=os.path.join(folder_root, (model_name)), format='png')
    
    input = torch.randn(1, 1, 88, 88)
    macs, params = profile(model, inputs=(input, input))
    macs, params = clever_format([macs, params], "%.3f")
    print('flops: {flops}, params: {params}'.format(flops=(macs*2), params=params))
    
    file_path = Path(os.path.join(folder_root, (model_name+'.txt')))
    with open(file_path, 'w') as f:
        with redirect_stdout(f):
            summary(model, (1, 88, 88))
    summary(model, (1, 88, 88))