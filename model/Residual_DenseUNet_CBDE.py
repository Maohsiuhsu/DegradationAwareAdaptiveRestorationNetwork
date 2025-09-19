import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import os
from pathlib import Path
from contextlib import redirect_stdout

from model.DenseUNet import Residual_Dense_UNet_v1_updown
from model.CBDE import CBDE
from model.DGM import DGB_Module

class Residual_Dense_UNet_v1_updown_with_DGB(Residual_Dense_UNet_v1_updown):
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
        layer1_e_dense_block_feat = self.layer1_e_dense_block(layer1_e_conv_block_feat)
        layer1_down_sample_feat = self.layer1_down_sample(layer1_e_dense_block_feat)

        layer2_e_conv_block_feat = self.layer2_e_conv_block(layer1_down_sample_feat)
        layer2_e_dense_block_feat = self.layer2_e_dense_block(layer2_e_conv_block_feat)
        layer2_down_sample_feat = self.layer2_down_sample(layer2_e_dense_block_feat)

        layer3_e_conv_block_feat = self.layer3_e_conv_block(layer2_down_sample_feat)
        layer3_e_dense_block_feat = self.layer3_e_dense_block(layer3_e_conv_block_feat)
        layer3_down_sample_feat = self.layer3_down_sample(layer3_e_dense_block_feat)

        layer4_e_conv_block_feat = self.layer4_e_conv_block(layer3_down_sample_feat)

        # decoder
        layer4_d_dconv_block_feat = self.layer4_d_dconv_block(layer4_e_conv_block_feat)
        layer4_up_sample_feat = self.layer4_up_sample(layer4_d_dconv_block_feat)

        layer3_add_feat = torch.add(layer3_e_dense_block_feat, layer4_up_sample_feat)
        layer3_d_dense_block_feat = self.layer3_d_dense_block(layer3_add_feat)
        layer3_d_se_block_feat = self._layer3_d_se_block(layer3_d_dense_block_feat)
        layer3_d_dconv_block_feat = self.layer3_d_dconv_block(layer3_d_se_block_feat)
        layer3_up_sample_feat = self.layer3_up_sample(layer3_d_dconv_block_feat)

        layer2_add_feat = torch.add(layer2_e_dense_block_feat, layer3_up_sample_feat)
        layer2_d_dense_block_feat = self.layer2_d_dense_block(layer2_add_feat)
        layer2_d_se_block_feat = self._layer2_d_se_block(layer2_d_dense_block_feat)
        layer2_d_dconv_block_feat = self.layer2_d_dconv_block(layer2_d_se_block_feat)
        layer2_up_sample_feat = self.layer2_up_sample(layer2_d_dconv_block_feat)

        layer1_add_feat = torch.add(layer1_e_dense_block_feat, layer2_up_sample_feat)
        layer1_d_dense_block_feat = self.layer1_d_dense_block(layer1_add_feat)
        layer1_d_se_block_feat = self._layer1_d_se_block(layer1_d_dense_block_feat)
        layer1_d_dconv_block_feat = self.layer1_d_dconv_block(layer1_d_se_block_feat)

        # feature fusion: BFM + layer1_d_dconv_block
        feature_fusion_feat = BFM_module + layer1_d_dconv_block_feat
        layer1_d_conv_block_feat = self.layer1_d_conv_block(feature_fusion_feat)
        layer1_d_conv_block_feat = self.layer1_d_activation(layer1_d_conv_block_feat)

        return layer1_d_conv_block_feat

class Residual_Dense_UNet_v1_air(nn.Module):
    def __init__(self, model_name="denseunet", seblock_pos="encoder", conf=None, device=None):
        super().__init__()
        if conf == None:
            self.activation = 'relu'
            self.init_weights = False
        else:
            self.activation = conf["Model"]["activation"]
            self.init_weights = conf["Model"]["init_weight"]
        
        self.degradation_encoder = CBDE(batch_size=conf['Model']['batch_size'])
        self.degradation_restorer = Residual_Dense_UNet_v1_updown_with_DGB(model_name=model_name, seblock_pos=seblock_pos, conf=conf, device=device)
        
        if self.init_weights:
            self._initialize_weights()
            
    def forward(self, x_query, x_key):
        fea, logits, labels, degradation_representation = self.degradation_encoder(x_query, x_key)
        
        restored = self.degradation_restorer(x_query, degradation_representation)

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