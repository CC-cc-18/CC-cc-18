
colors = {
    # BGR
    0: [250, 250, 250],
    1: [51, 153, 153],
    2: [241, 249, 0],
    3: [102, 51, 204],
    4: [51, 102, 204],
    5: [0, 255, 0],
    6: [39, 169, 0],
    7: [0, 0, 153],
    8: [255, 45, 0],
    9: [51, 51, 102],
    10: [51, 102, 0],
    11: [0, 144, 255],
    12: [153, 204, 255],
    13: [0, 0, 255],
    14: [102, 153, 51],
    15: [255, 153, 255],
    16: [255, 153, 102],
    17: [204, 255, 102],
    18: [0, 255, 255],
    19: [204, 204, 255],
    20: [255, 51, 102],
    21: [169, 39, 153],
    22: [51, 153, 39],
    23: [204, 204, 45],
}

colorsf = {
    0: [250/255, 250/255, 250/255],
    1: [153/255, 153/255, 51/255],
    2: [0/255, 249/255, 241/255],
    3: [204/255, 51/255, 102/255],
    4: [204/255, 102/255, 51/255],
    5: [0/255, 255/255, 0/255],
    6: [0/255, 169/255, 39/255],
    7: [153/255, 0/255, 0/255],
    8: [0/255, 45/255, 255/255],
    9: [102/255, 51/255, 51/255],
    10: [0/255, 102/255, 51/255],
    11: [255/255, 144/255, 0/255],
    12: [255/255, 204/255, 153/255],
    13: [255/255, 0/255, 0/255],
    14: [51/255, 153/255, 102/255],
    15: [255/255, 153/255, 255/255],
    16: [102/255, 153/255, 255/255],
    17: [102/255, 255/255, 204/255],
    18: [255/255, 255/255, 0/255],
    19: [255/255, 204/255, 204/255],
    20: [102/255, 51/255, 255/255],
    21: [153/255, 39/255, 169/255],
    22: [39/255, 153/255, 51/255],
    23: [45/255, 204/255, 204/255],
}

import numpy as np
from time import time
import cv2
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import math
from torchvision.ops.deform_conv import deform_conv2d
from sklearn.metrics import confusion_matrix
from einops.layers.torch import Rearrange
import torchvision

from sklearn.manifold import TSNE




class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        out_shape = x.shape
        channels_per_group = out_shape[-1] // self.groups

        # Reshape
        x = x.view(-1, self.groups, channels_per_group)

        # Transpose
        x = x.transpose(-1, -2).contiguous()

        # Flatten
        x = x.view(*out_shape)
        return x


class TaylorLayer(nn.Module):

    def __init__(self, input_dim, out_dim, order, groups=1, addbias=True):
        super(TaylorLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.groups = groups
        self.addbias = addbias


        assert input_dim % groups == 0 and out_dim % groups == 0, "input_dim and out_dim must be divisible by groupss"
        self.groups_in_dim = input_dim // groups
        self.groups_out_dim = out_dim // groups


        self.coeffs = nn.Parameter(torch.randn(groups, self.groups_out_dim, self.groups_in_dim, order) * 0.02)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))


        if self.groups > 1:
            self.shuffle = ChannelShuffle(self.groups)

    def forward(self, x):
        shape = x.shape
        outshape = shape[:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))


        x_grouped = x.view(-1, self.groups, self.groups_in_dim)


        x_expanded = x_grouped.unsqueeze(2).expand(-1, -1, self.groups_out_dim, -1)


        y = torch.zeros((x.shape[0], self.groups, self.groups_out_dim), device=x.device)
        for i in range(self.order):
            term = (x_expanded ** i) * self.coeffs[:, :, :, i]
            y += term.sum(dim=-1)


        y = y.view(-1, self.out_dim)

        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        if self.groups > 1:
            y = self.shuffle(y)
        return y

#
# class TaylorLayer(nn.Module):
#     def __init__(self, input_dim, out_dim, order, addbias=True):
#         super(TaylorLayer, self).__init__()
#         self.input_dim = input_dim
#         self.out_dim = out_dim
#         self.order = order
#         self.addbias = addbias
#
#         # Initialize Taylor coefficients
#         self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
#         if self.addbias:
#             self.bias = nn.Parameter(torch.zeros(1, out_dim))
#
#     def forward(self, x):
#         shape = x.shape
#         outshape = shape[0:-1] + (self.out_dim,)
#         x = torch.reshape(x, (-1, self.input_dim))
#
#         x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)
#
#         # Compute and accumulate each term of the Taylor expansion
#         y = torch.zeros((x.shape[0], self.out_dim), device=x.device)
#
#         for i in range(self.order):
#             term = (x_expanded ** i) * self.coeffs[:, :, i]
#             y += term.sum(dim=-1)
#
#         if self.addbias:
#             y += self.bias
#
#         y = torch.reshape(y, outshape)
#         return y

# class Spatial_Shift(nn.Module):
#     def __init__(self):
#         super(Spatial_Shift, self).__init__()
#
#     def forward(self, x):
#         b, w, h, c = x.size()
#         first_quarter = torch.roll(x[:, :, :, :c // 4], shifts=1, dims=1)
#         x[:, 1:, :, :c // 4] = first_quarter[:, 1:, :, :]
#         second_quarter = torch.roll(x[:, :, :, c // 4:c // 2], shifts=-1, dims=1)
#         x[:, :w - 1, :, c // 4:c // 2] = second_quarter[:, :w - 1, :, :]
#         third_quarter = torch.roll(x[:, :, :, c // 2:c * 3 // 4], shifts=1, dims=2)
#         x[:, :, 1:, c // 2:c * 3 // 4] = third_quarter[:, :, 1:, :]
#         last_quarter = torch.roll(x[:, :, :, 3 * c // 4:], shifts=-1, dims=2)
#         x[:, :, :h - 1, 3 * c // 4:] = last_quarter[:, :, :h - 1, :]
#
#         return x




class DynamicDeformConv2D(nn.Module):
    def __init__(self, channels, learnable_offsets=True):
        super().__init__()
        self.channels = channels
        self.learnable_offsets = learnable_offsets


        self.deform_conv = torchvision.ops.DeformConv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=None, groups=channels)
        self.deform_conv.weight.data.fill_(1.0)
        self.deform_conv.weight.requires_grad = False


        if self.learnable_offsets:
            self.offset_generator = nn.Sequential(
                nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=1, padding=1),
                nn.AvgPool2d(2, 2),
                nn.ReLU(),
                nn.Conv2d(channels * 2, 1 * channels, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool2d(1),
                # nn.Tanh(),
            )

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        fixed_offsets = self.create_offsets(batch_size, channels, height, width, device=x.device)

        if self.learnable_offsets:

            learned_offsets = self.offset_generator(x).expand(batch_size, channels, height, width)

            offsets = fixed_offsets
            offsets[:, :channels] = offsets[:, :channels] + learned_offsets * 0.5
        else:
            offsets = fixed_offsets
        offsets = torch.clamp(offsets, -2, 2)
        return self.deform_conv(x, offsets)

    def create_offsets(self, batch_size, channels, height, width, device):

        offsets = torch.zeros(batch_size, 2 * channels, height, width, device=device)

        c_per_group = channels // 5
        for i in range(4):
            if width > 1:
                offsets[:, 0::2, :, i*c_per_group + 1:(i+1)*c_per_group] = 1
                offsets[:, 0::2, :, i*c_per_group + 1:(i+1)*c_per_group] = -1
            if height > 1:
                offsets[:, 1::2, 1:(height - 1), i*c_per_group:(i+1)*c_per_group] = 1
                offsets[:, 1::2, 1:(height - 1), i*c_per_group:(i+1)*c_per_group] = -1



        return offsets

class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Embedding, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 5, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 5 * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 5, 5, 1, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 5, 3, 1, 1),
            nn.MaxPool2d(3, 1, 1)
        )

    def forward(self, x):

        return torch.concat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv4(x)], dim=1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        residual = x
        return  self.fn(x) + F.dropout(residual)


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_conv1 = nn.Sequential(
            nn.Conv3d(1, 8, (15, 3, 3), padding=(0, 1, 1), stride=(3, 1, 1)),
            nn.BatchNorm3d(8),
        )
        self.emb_conv2 = nn.Sequential(
            nn.Conv3d(1, 8, (15, 1, 1), padding=(0, 0, 0), stride=(3, 1, 1)),
            nn.BatchNorm3d(8),
        )

        self.mlp = nn.Sequential(

            Residual(
                nn.Sequential(
                    nn.Conv3d(16, 32, (1, 3, 3), padding=(0, 1, 1)),
                    nn.ReLU(),
                    nn.Conv3d(32, 16, (7, 1, 1), padding=(3, 0, 0)),
                )
            ),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, (15, 1, 1), stride=(5, 1, 1)),
            Residual(
                nn.Sequential(
                    nn.Conv3d(32, 64, (1, 3, 3), padding=(0, 1, 1)),
                    nn.ReLU(),
                    nn.Conv3d(64, 32, (7, 1, 1), padding=(3, 0, 0)),
                )
            ),

            nn.BatchNorm3d(32),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = torch.cat((self.emb_conv1(x), self.emb_conv2(x)), dim=1)

        return self.mlp(F.relu(x))



class ATKAN(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=[7, 2],
                 in_channels=3,
                 num_classes=1000,
                 ):
        super().__init__()

        # self.layers_hidden = [60, 120, 240]
        self.layers_hidden = [30, 40, 50]
        self.emb = Embedding(in_channels, self.layers_hidden[0])
        # self.emb = nn.Conv2d(in_channels, self.layers_hidden[0], 3, 1, 1)
        self.mlp = Branch()
        self.conv = nn.Sequential(
            nn.Conv2d(self.layers_hidden[0], self.layers_hidden[1], (3, 3), padding=1, stride=1),
            Residual(nn.Sequential(
                nn.Conv2d(self.layers_hidden[1], self.layers_hidden[2], (3, 3), padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(self.layers_hidden[2], self.layers_hidden[1], (3, 3), padding=1, stride=1),
            )),
            nn.BatchNorm2d(self.layers_hidden[1]),
            nn.Conv2d(self.layers_hidden[1], self.layers_hidden[2], (3, 3), padding=1, stride=1),
            nn.BatchNorm2d(self.layers_hidden[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.layers = nn.ModuleList([])
        for idx, (in_features, out_features) in enumerate(zip(self.layers_hidden, self.layers_hidden[1:])):
            # fourier_layer = GroupedFourierKANLayer(
            #     in_features,
            #     out_features,
            #     groups=4,
            # )
            taylor_layer = TaylorLayer(in_features, out_features, order=3, addbias=False, groups=5)
            setattr(self, f'taylor_layer__{idx}', taylor_layer)
            self.layers.append(
                nn.Sequential(
                    DynamicDeformConv2D(in_features) if idx < 1 else nn.Identity(),
                    Rearrange('b c h w -> b h w c'),
                    taylor_layer,
                    Rearrange('b h w c -> b c h w'),
                    nn.AvgPool2d(2, 2) if idx >= 1 else nn.Identity(),
                    nn.BatchNorm2d(out_features),
                    # nn.AvgPool2d(2, 2) if idx > 0 else DynamicDeformConv2D(out_features),
                )
            )


        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.cls = nn.Sequential(
            nn.Linear(self.layers_hidden[-1] * 2 + 32, num_classes),
            # TaylorLayer(self.layers_hidden[-1], num_classes, order=2, addbias=False)
        )

    def forward(self, x: torch.Tensor, update_grid=False):
        z = x.unsqueeze(1)
        z = self.mlp(z)
        x = self.emb(x)
        s = self.conv(x)
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        x = self.pool(x)
        fe = torch.cat((x, z, s), dim=1)
        # fe = x + z + s
        return self.cls(fe), fe




if __name__ == '__main__':
    input = torch.randn((8, 200, 15, 15)).cuda()
    output, _ = ATKAN(image_size=15, patch_size=3, in_channels=200, num_classes=10).cuda()(input)
    print(output.shape)
