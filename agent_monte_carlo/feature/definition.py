#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


def sample_process(list_game_data):
    """
    Process game data into sample format for training
    将游戏数据处理为训练样本格式

    Args:
        list_game_data: List of game frames / 游戏帧列表

    Returns:
        List of processed samples (dict format) / 处理后的样本列表（字典格式）
    """
    return [{"state": frame.state, "action": frame.action, "reward": frame.reward} for frame in list_game_data]


def reward_shaping(env_reward, env_obs):
    """
    Shape reward signal for better learning (Monte Carlo specific)
    塑形奖励信号以改善学习效果（蒙特卡洛专用）

    Note: This version includes step penalty to encourage faster completion
    注意：此版本包含步数惩罚以鼓励更快完成任务

    Args:
        env_reward: Original environment reward (unused) / 原始环境奖励（未使用）
        env_obs: Environment observation / 环境观测

    Returns:
        Shaped reward value / 塑形后的奖励值
    """
    score = env_obs["observation"]["env_info"]["score"]
    terminated = env_obs["terminated"]

    reward = 0

    # Base reward: use environment score
    # 基础奖励：使用环境得分
    reward += score

    # Step penalty: encourage efficient solutions
    # 步数惩罚：鼓励高效解决方案
    if not terminated:
        reward += -1

    return reward
