#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from common_python.utils.common_func import create_cls
from agent_monte_carlo.conf.conf import Config
from agent_monte_carlo.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        """
        Initialize Monte Carlo agent
        初始化蒙特卡洛智能体

        Args:
            agent_type: Type of agent / 智能体类型
            device: Computing device / 计算设备
            logger: Logger instance / 日志记录器实例
            monitor: Monitor instance / 监控实例
        """
        self.logger = logger

        # Initialize hyperparameters
        # 初始化超参数
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.epsilon = Config.EPSILON
        self.algorithm = Algorithm(Config.GAMMA, self.state_size, self.action_size)

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        """
        Predict action using epsilon-greedy policy
        使用epsilon-greedy策略预测动作

        Args:
            list_obs_data: List of observation data / 观测数据列表

        Returns:
            List of action data / 动作数据列表
        """
        state = list_obs_data[0].feature
        action = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=int(action))]

    def exploit(self, env_obs):
        """
        Exploit current policy for evaluation
        利用当前策略进行评估

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            Action to take / 要执行的动作
        """
        obs_data = self.observation_process(env_obs)
        state = obs_data.feature
        act_data = ActData(act=int(self.algorithm.policy[state]))
        action = self.action_process(act_data)
        return action

    def _epsilon_greedy(self, state, epsilon=0.1):
        """
        Epsilon-greedy algorithm for action selection
        ε-贪心算法用于动作选择

        Args:
            state: Current state / 当前状态
            epsilon: Exploration rate / 探索率

        Returns:
            Selected action / 选择的动作
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.action_size))
        else:
            return int(self.algorithm.policy[state])

    def learn(self, list_sample_data):
        """
        Update policy using Monte Carlo algorithm
        使用蒙特卡洛算法更新策略

        Args:
            list_sample_data: List of episode samples / 回合样本列表

        Returns:
            Learning result / 学习结果
        """
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env_obs):
        """
        Process environment observation into feature representation
        将环境观测处理为特征表示

        Note: Combines position and treasure chest status into a single feature.
        If additional feature processing is performed, corresponding modifications are needed
        for the policy structure and algorithm methods.
        注意：将位置和宝箱状态组合成单一特征。如进行额外特征处理，则需要对策略结构
        和算法方法进行相应的改动。

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            ObsData with processed features / 处理后的观测数据
        """
        obs = env_obs["observation"]
        pos = [obs["frame_state"]["hero"]["pos"]["x"], obs["frame_state"]["hero"]["pos"]["z"]]

        # Position feature: encode 2D position as 1D index
        # 位置特征：将2D位置编码为1D索引
        pos_feature = int(pos[0] * 64 + pos[1])

        return ObsData(feature=pos_feature)

    def action_process(self, act_data):
        """
        Process action data into executable action
        将动作数据处理为可执行动作

        Args:
            act_data: Action data / 动作数据

        Returns:
            Executable action / 可执行动作
        """
        return act_data.act

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.algorithm.policy)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.policy = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)
