#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


class Algorithm:
    def __init__(self, gamma, learning_rate, state_size, action_size):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        # Reset the Q-table
        # 重置Q表
        self.Q = np.ones([self.state_size, self.action_size])

    def learn(self, list_sample_data):
        """
        Update the Q-table using the Q-learning algorithm with the given sample data.

        Args:
            list_sample_data: A list of sample dictionaries, each containing:
                - state: Current state index
                - action: Action taken in the current state
                - reward: Reward received after taking the action
                - next_state: Next state index after taking the action

        Q-Learning Update Formula:
            Q(s,a) := Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]

        Where:
            - Q(s,a): Q-value for taking action a in state s
            - lr (learning_rate): Learning rate, controls the magnitude of each update
            - R(s,a): Reward received for taking action a in state s
            - gamma: Discount factor, balances the importance of immediate vs future rewards
            - max Q(s',a'): Maximum Q-value over all possible actions a' in next state s'

        使用给定的样本数据通过 Q-learning 算法更新 Q 表。

        参数:
            list_sample_data: 样本字典列表，每个样本包含:
                - state: 当前状态索引
                - action: 在当前状态下采取的动作
                - reward: 采取动作后获得的奖励
                - next_state: 采取动作后的下一个状态索引

        Q-Learning 更新公式:
            Q(s,a) := Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]

        其中:
            - Q(s,a): 在状态 s 下采取动作 a 的 Q 值
            - lr (learning_rate): 学习率，用于控制每次更新的幅度
            - R(s,a): 在状态 s 下采取动作 a 所获得的奖励
            - gamma: 折扣因子，用于平衡当前奖励和未来奖励的重要性
            - max Q(s',a'): 在新状态 s' 下所有可能动作 a' 的最大 Q 值
        """
        sample = list_sample_data[0]
        state, action, reward, next_state = (
            sample["state"],
            sample["action"],
            sample["reward"],
            sample["next_state"],
        )

        delta = reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]

        self.Q[state, action] += self.learning_rate * delta

        return
