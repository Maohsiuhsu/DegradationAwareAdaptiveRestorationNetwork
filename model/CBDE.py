# reference https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/net/encoder.py
import torch
import torch.nn as nn
from model.MoCo import MoCo
# from MoCo import MoCo


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            self._activate_fn(activation='gelu'),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )
        
        self.activation = self._activate_fn(activation='gelu')

    def forward(self, x):
        return self.activation(self.backbone(x) + self.shortcut(x))
    
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


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()

        self.E_pre = ResBlock(in_feat=1, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            self._activate_fn(activation='gelu'),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter
    
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


class CBDE(nn.Module):
    def __init__(self, batch_size, device=None):
        super(CBDE, self).__init__()

        dim = 256

        # Encoder
        self.E = MoCo(base_encoder=ResEncoder, dim=dim, K=batch_size * dim, device=device)

    def forward(self, x_query, x_key):
        # degradation-aware represenetion learning
        embedding, logits, labels, inter = self.E(x_query, x_key)

        return embedding, logits, labels, inter