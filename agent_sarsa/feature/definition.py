#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


def sample_process(list_game_data):
    """
    Process game data into sample format for training
    将游戏数据处理为训练样本格式

    Args:
        list_game_data: List of game frames / 游戏帧列表

    Returns:
        List of processed samples (dict format) / 处理后的样本列表（字典格式）
    """
    return [
        {
            "state": frame.state,
            "action": frame.action,
            "reward": frame.reward,
            "next_state": frame.next_state,
            "next_action": frame.next_action,
        }
        for frame in list_game_data
    ]


def reward_shaping(env_reward, env_obs):
    """
    Shape reward signal for better learning
    塑形奖励信号以改善学习效果

    Args:
        env_reward: Original environment reward (unused) / 原始环境奖励（未使用）
        env_obs: Environment observation / 环境观测

    Returns:
        Shaped reward value / 塑形后的奖励值
    """
    score = env_obs["observation"]["env_info"]["score"]
    terminated = env_obs["terminated"]

    reward = 0

    # Reward for winning (final score when episode ends)
    # 获胜奖励（回合结束时的最终得分）
    if terminated:
        reward += score

    # Reward for collecting treasure chests during episode
    # 回合期间收集宝箱的奖励
    if score > 0 and not terminated:
        reward += score

    return reward
