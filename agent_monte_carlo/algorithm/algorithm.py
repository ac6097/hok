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
    def __init__(self, gamma, state_size, action_size):
        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size

        # Initialize the policy
        # 初始化策略
        self.policy = np.random.choice(self.action_size, self.state_size)
        self.Q = np.zeros([self.state_size, self.action_size])
        self.visit = np.zeros([self.state_size, self.action_size])

    def learn(self, list_sample_data):
        """
        Update the Q-table and policy using Monte Carlo Control with First-Visit method.

        Args:
            list_sample_data: A list of sample dictionaries from a complete episode, each containing:
                - state: State index visited in the episode
                - action: Action taken in that state
                - reward: Reward received after taking the action

        Monte Carlo Return Calculation:
            G(t) = R(t+1) + gamma * R(t+2) + gamma^2 * R(t+3) + ... + gamma^(T-t-1) * R(T)

        Where:
            - G(t): Return (cumulative discounted reward) starting from time step t
            - R(t+1), R(t+2), ..., R(T): Rewards received at each subsequent time step
            - gamma: Discount factor, determines the importance of future rewards
            - T: Terminal time step (end of episode)

        Algorithm Steps:
            1. Calculate returns G(t) for each state-action pair by working backwards from episode end
            2. Update Q(s,a) using incremental mean for first visit to each (s,a) pair:
               Q(s,a) := Q(s,a) + (G - Q(s,a)) / N(s,a)
            3. Update policy to be greedy with respect to Q: π(s) = argmax_a Q(s,a)

        Key Features:
            - First-Visit: Only the first occurrence of each (s,a) pair in an episode is used
            - On-Policy: Learns the value of the policy being followed
            - Episode-based: Requires complete episodes (no bootstrapping)

        使用蒙特卡洛控制首次访问方法更新 Q 表和策略。

        参数:
            list_sample_data: 完整回合的样本字典列表，每个样本包含:
                - state: 回合中访问的状态索引
                - action: 在该状态下采取的动作
                - reward: 采取动作后获得的奖励

        蒙特卡洛回报计算公式:
            G(t) = R(t+1) + gamma * R(t+2) + gamma^2 * R(t+3) + ... + gamma^(T-t-1) * R(T)

        其中:
            - G(t): 从时间步 t 开始的回报（累积折扣奖励）
            - R(t+1), R(t+2), ..., R(T): 后续每个时间步收到的奖励
            - gamma: 折扣因子，决定未来奖励的重要性
            - T: 终止时间步（回合结束）

        算法步骤:
            1. 从回合末尾向前计算每个状态-动作对的回报 G(t)
            2. 对每个 (s,a) 对的首次访问，使用递增均值更新 Q(s,a):
               Q(s,a) := Q(s,a) + (G - Q(s,a)) / N(s,a)
            3. 更新策略为关于 Q 的贪婪策略: π(s) = argmax_a Q(s,a)

        关键特性:
            - 首次访问: 每个回合中只使用每个 (s,a) 对的首次出现
            - 同策略: 学习正在遵循的策略的价值
            - 基于回合: 需要完整的回合（无自举）
        """
        G, state_action_return = 0, []

        # Calculate the return for each state-action pair
        # 计算每个状态-动作对的回报
        for sample in reversed(list_sample_data[:-1]):
            state_action_return.append((sample["state"], sample["action"], G))
            G = self.gamma * G + sample["reward"]

        state_action_return.reverse()

        # Update the Q-table
        # 更新Q表
        seen_state_action = set()
        for state, action, G in state_action_return:
            if (state, action) not in seen_state_action:
                self.visit[state][action] += 1

                # calculate incremental mean
                # 计算递增均值
                self.Q[state, action] = self.Q[state, action] + (G - self.Q[state, action]) / self.visit[state, action]
                seen_state_action.add((state, action))

        # Update policy
        # 更新策略
        for state in range(self.state_size):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = best_action

        return
