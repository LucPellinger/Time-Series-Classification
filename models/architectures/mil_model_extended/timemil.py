# timemil.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from mil_model_extended.inceptiontime import InceptionTimeFeatureExtractor
from mil_model_extended.nystrom_attention import NystromAttention
from mil_model_extended.multichannel_attention import CrossChannelAttention

"""
https://github.com/xiwenc1/TimeMIL/tree/main

@article{chen2024timemil,
  title={TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning},
  author={Chen, Xiwen and Qiu, Peijie and Zhu, Wenhui and Li, Huayu and Wang, Hao and Sotiras, Aristeidis and Wang, Yalin and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2405.03140},
  year={2024}
}
"""

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
                

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dropout=0.2, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,       # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,           # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
    

### Define Wavelet Kernel
def mexican_hat_wavelet(size, scale, shift):  # size: d*kernelsize, scale: d*1, shift: d*1
    """
    Generate a Mexican Hat wavelet kernel.

    Parameters:
    size (tuple): Size of the kernel.
    scale (torch.Tensor): Scale of the wavelet.
    shift (torch.Tensor): Shift of the wavelet.

    Returns:
    torch.Tensor: Mexican Hat wavelet kernel.
    """
    device = scale.device
    x = torch.linspace(-(size[1] - 1) // 2, (size[1] - 1) // 2, size[1], device=device)
    x = x.reshape(1, -1).repeat(size[0], 1)
    x = x - shift  # Apply the shift

    # Mexican Hat wavelet formula
    C = 2 / (3 ** 0.5 * torch.pi ** 0.25)
    wavelet = C * (1 - (x / scale) ** 2) * torch.exp(-(x / scale) ** 2 / 2) * 1 / (torch.abs(scale) ** 0.5)

    return wavelet  # d*L

class WaveletEncoding(nn.Module):
    def __init__(self, dim=512, max_len=256, hidden_len=512, dropout=0.0):
        super().__init__()

        # n_w =3
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim, dim)

    def forward(self, x, wave1, wave2, wave3):
        cls_token, feat_token = x[:, 0], x[:, 1:]

        x = feat_token.transpose(1, 2)

        D = x.shape[1]
        scale1, shift1 = wave1[0, :], wave1[1, :]
        wavelet_kernel1 = mexican_hat_wavelet(size=(D, 19), scale=scale1, shift=shift1)
        scale2, shift2 = wave2[0, :], wave2[1, :]
        wavelet_kernel2 = mexican_hat_wavelet(size=(D, 19), scale=scale2, shift=shift2)
        scale3, shift3 = wave3[0, :], wave3[1, :]
        wavelet_kernel3 = mexican_hat_wavelet(size=(D, 19), scale=scale3, shift=shift3)

        # Eq. 11
        pos1 = torch.nn.functional.conv1d(x, wavelet_kernel1.unsqueeze(1), groups=D, padding='same')
        pos2 = torch.nn.functional.conv1d(x, wavelet_kernel2.unsqueeze(1), groups=D, padding='same')
        pos3 = torch.nn.functional.conv1d(x, wavelet_kernel3.unsqueeze(1), groups=D, padding='same')
        x = x.transpose(1, 2)   # B*N*D

        # Eq. 10
        x = x + self.proj_1(pos1.transpose(1, 2) + pos2.transpose(1, 2) + pos3.transpose(1, 2))  # + mixup_encording

        # Mixup token information
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TimeMIL_extended(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400, dropout=0.):
        super().__init__()

        # Define backbone (can be replaced here)
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features)
        # added cross channel attention
        self.cross_channel_attention = CrossChannelAttention(D=mDim, num_heads=4, dropout=dropout)

        # Define WPE    
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.wave1 = torch.randn(2, mDim, 1)
        self.wave1[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale >0
        self.wave1 = nn.Parameter(self.wave1)

        self.wave2 = torch.zeros(2, mDim, 1)
        self.wave2[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale >0
        self.wave2 = nn.Parameter(self.wave2)

        self.wave3 = torch.zeros(2, mDim, 1)
        self.wave3[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale >0
        self.wave3 = nn.Parameter(self.wave3)

        self.wave1_ = torch.randn(2, mDim, 1)
        self.wave1_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale >0
        self.wave1_ = nn.Parameter(self.wave1_)

        self.wave2_ = torch.zeros(2, mDim, 1)
        self.wave2_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale >0
        self.wave2_ = nn.Parameter(self.wave2_)

        self.wave3_ = torch.zeros(2, mDim, 1)
        self.wave3_[0] = torch.ones(mDim, 1) + torch.randn(mDim, 1)  # make sure scale >0
        self.wave3_ = nn.Parameter(self.wave3_)

        hidden_len = 2 * max_seq_len

        # Define class token      
        self.cls_token = nn.Parameter(torch.randn(1, 1, mDim))
        self.pos_layer = WaveletEncoding(mDim, max_seq_len, hidden_len)
        self.pos_layer2 = WaveletEncoding(mDim, max_seq_len, hidden_len)
        # self.pos_layer = ConvPosEncoding1D(mDim)
        self.layer1 = TransLayer(dim=mDim, dropout=dropout)
        self.layer2 = TransLayer(dim=mDim, dropout=dropout)
        self.norm = nn.LayerNorm(mDim)
        # self._fc2 = nn.Linear(mDim, n_classes)
        self._fc2 = nn.Sequential(
            nn.Linear(mDim, mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mDim, n_classes)
        )

        self.alpha = nn.Parameter(torch.ones(1))

        initialize_weights(self)

    def forward(self, x, warmup=False):
        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        # Apply cross-channel attention
        x1 = self.cross_channel_attention(x1)
        x = x1
        
        B, seq_len, D = x.shape

        view_x = x.clone()

        global_token = x.mean(dim=1)

        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)    # B * 1 * d

        x = torch.cat((cls_tokens, x), dim=1)
        # WPE1
        x = self.pos_layer(x, self.wave1, self.wave2, self.wave3)

        # TransLayer x1
        x = self.layer1(x)
        # WPE2
        x = self.pos_layer2(x, self.wave1_, self.wave2_, self.wave3_)

        # TransLayer x2
        x = self.layer2(x)

        # Only cls_token is used for classification
        x = x[:, 0]

        # Stability of training random initialized global token
        if warmup:
            x = 0.1 * x + 0.99 * global_token

        logits = self._fc2(x)
        attention_weights = self.layer2.attn.attn_weights  # Assuming attn_weights are stored

        return logits, attention_weights

if __name__ == "__main__":
    x = torch.randn(3, 400, 4).cuda()
    model = TimeMIL(in_features=4, mDim=128).cuda()
    logits, attn_weights = model(x)
    print(logits.shape)
    print(attn_weights.shape)
