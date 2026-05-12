#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions, GAE computation for Gorge Chase PPO (Enhanced).
峡谷追猎 PPO 数据类定义与 GAE 计算（增强版）。
"""

from common_python.utils.common_func import create_cls
from ..conf.conf import Config


# ObsData: feature=1822D vector, legal_action=16D mask
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: action, d_action(greedy), prob, value
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: single-frame sample with int dims
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,      # 1822
    legal_action=Config.ACTION_NUM,      # 16
    act=1,
    reward=Config.VALUE_NUM,             # 1
    reward_sum=Config.VALUE_NUM,         # 1
    done=1,
    value=Config.VALUE_NUM,              # 1
    next_value=Config.VALUE_NUM,         # 1
    advantage=Config.VALUE_NUM,          # 1
    prob=Config.ACTION_NUM,              # 16
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value
