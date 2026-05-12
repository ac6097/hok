#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Neural network model for Gorge Chase PPO (Enhanced).
峡谷追猎 PPO 神经网络模型（增强版）。

Architecture / 架构:
  Spatial pathway:  4×21×21 → Conv2d → Conv2d → AdaptiveAvgPool → FC → 128D
  Scalar pathway:   58D → FC → FC → 128D
  Fusion:           concat(256D) → FC → 128D
  Actor head:       128D → 16 (action logits)
  Critic head:      128D → 1  (state value)
"""

import torch
import torch.nn as nn

from ..conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """CNN + MLP dual pathway with fusion.

    CNN + MLP 双通道融合架构。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_enhanced"
        self.device = device

        scalar_dim = Config.SCALAR_DIM           # 58
        spatial_ch = Config.SPATIAL_CHANNELS      # 4
        action_num = Config.ACTION_NUM             # 16
        pool_size = Config.CNN_POOL_SIZE           # 3

        # --- Spatial pathway (CNN) ---
        self.conv1 = nn.Conv2d(spatial_ch, Config.CNN_CHANNELS[0], 3, padding=1)
        self.conv2 = nn.Conv2d(Config.CNN_CHANNELS[0], Config.CNN_CHANNELS[1], 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        cnn_flat_dim = Config.CNN_CHANNELS[1] * pool_size * pool_size  # 32*3*3=288
        self.spatial_fc = make_fc_layer(cnn_flat_dim, Config.CNN_FC_OUT)

        # Init conv layers
        nn.init.orthogonal_(self.conv1.weight.data)
        nn.init.zeros_(self.conv1.bias.data)
        nn.init.orthogonal_(self.conv2.weight.data)
        nn.init.zeros_(self.conv2.bias.data)

        # --- Scalar pathway (MLP) ---
        self.scalar_fc1 = make_fc_layer(scalar_dim, Config.SCALAR_HIDDEN)
        self.scalar_fc2 = make_fc_layer(Config.SCALAR_HIDDEN, Config.SCALAR_FC_OUT)

        # --- Fusion ---
        self.fusion_fc = make_fc_layer(Config.FUSION_DIM, Config.FUSION_HIDDEN)

        # --- Actor & Critic heads ---
        self.actor_head = make_fc_layer(Config.FUSION_HIDDEN, action_num)
        self.critic_head = make_fc_layer(Config.FUSION_HIDDEN, Config.VALUE_NUM)

        self.relu = nn.ReLU()

    def forward(self, obs, inference=False):
        """Forward pass. obs shape: (batch, 1822).

        前向传播。obs 形状：(batch, 1822)。
        """
        # Split obs into scalar and spatial parts
        scalar = obs[:, :Config.SCALAR_DIM]                         # (B, 58)
        spatial_flat = obs[:, Config.SCALAR_DIM:]                   # (B, 1764)
        spatial = spatial_flat.view(
            -1,
            Config.SPATIAL_CHANNELS,
            Config.SPATIAL_SIZE,
            Config.SPATIAL_SIZE,
        )                                                            # (B, 4, 21, 21)

        # Spatial pathway
        x_s = self.relu(self.conv1(spatial))                        # (B, 16, 21, 21)
        x_s = self.relu(self.conv2(x_s))                            # (B, 32, 21, 21)
        x_s = self.pool(x_s)                                        # (B, 32, 3, 3)
        x_s = x_s.view(x_s.size(0), -1)                            # (B, 288)
        x_s = self.relu(self.spatial_fc(x_s))                       # (B, 128)

        # Scalar pathway
        x_m = self.relu(self.scalar_fc1(scalar))                    # (B, 128)
        x_m = self.relu(self.scalar_fc2(x_m))                       # (B, 128)

        # Fusion
        fused = torch.cat([x_s, x_m], dim=1)                       # (B, 256)
        fused = self.relu(self.fusion_fc(fused))                    # (B, 128)

        # Heads
        logits = self.actor_head(fused)                             # (B, 16)
        value = self.critic_head(fused)                             # (B, 1)

        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
