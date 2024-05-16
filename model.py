#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
from torch.nn import init


class Encoder(nn.Sequential):
    @staticmethod
    def create_block(in_channels: int, hidden_channels: int):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden_channels, 3, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2),
        )

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__(
            *[
                self.create_block(in_channels, hidden_channels),
                self.create_block(hidden_channels, hidden_channels * 2),
                self.create_block(hidden_channels * 2, hidden_channels * 4),
                nn.AdaptiveMaxPool2d((8, 8)),
            ]
        )


class Network(nn.Module):
    def __init__(
        self,
        num_shapes: int,
        param_per_stroke: int = 4,
        num_strokes: int = 8,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 3,
        n_dec_layers: int = 3,
    ):
        """Network for generating strokes and determining which indices to paint onto the canvas

        Args:
            num_shapes : int
                Number of shapes
            param_per_stroke : int, optional
                Number of parameters per stroke (4 for center x, center y, scale, and rotation), by default 4
            num_strokes : int, optional
                Total number of strokes, by default 8
            hidden_dim : int, optional
                Hidden dimension, by default 256
            n_heads : int, optional
                Number of heads, by default 8
            n_enc_layers : int, optional
                Number of encoder layers, by default 3
            n_dec_layers : int, optional
                Number of decoder layers, by default 3
        """
        super().__init__()
        self.enc_img = Encoder(3, 32)
        self.enc_canvas = Encoder(3, 32)
        self.conv = nn.Conv2d(128 * 2, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, n_heads, n_enc_layers, n_dec_layers
        )
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, param_per_stroke),
            nn.Sigmoid(),
        )
        hidden_idx = 2 ** round(math.log2((2 / 3) * max(hidden_dim, num_shapes)))
        self.linear_idx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_idx),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_idx, num_shapes),
        )
        self.query_pos = nn.Parameter(torch.rand(num_strokes, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))

    def forward(self, img, canvas):
        b = img.shape[0]
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)
        h, w = img_feat.shape[-2:]
        feat = torch.cat([img_feat, canvas_feat], dim=1)
        feat_conv = self.conv(feat)

        pos_embed = (
            torch.cat(
                [
                    self.col_embed[:w].unsqueeze(0).contiguous().repeat(h, 1, 1),
                    self.row_embed[:h].unsqueeze(1).contiguous().repeat(1, w, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        hidden_state = self.transformer(
            pos_embed + feat_conv.flatten(2).permute(2, 0, 1).contiguous(),
            self.query_pos.unsqueeze(1).contiguous().repeat(1, b, 1),
        )
        hidden_state = hidden_state.permute(1, 0, 2).contiguous()
        param = self.linear_param(hidden_state)
        idx = self.linear_idx(hidden_state)
        return param, idx


def init_func(m: torch.Tensor):
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
