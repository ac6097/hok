#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration of dimensions
# 关于维度的配置
class Config:

    STATE_SIZE = 64 * 64
    ACTION_SIZE = 4
    GAMMA = 0.9
    THETA = 1e-3
    EPISODES = 100

    # Dimension of observation
    # 观察维度
    OBSERVATION_SHAPE = 250
