#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
from common_python.utils.common_func import create_cls
from kaiwudrl.interface.agent import BaseAgent
from agent_dynamic_programming.conf.conf import Config
from agent_dynamic_programming.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        self.algorithm = Algorithm(
            Config.GAMMA, Config.THETA, Config.EPISODES, Config.STATE_SIZE, Config.ACTION_SIZE, self.logger
        )

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        """
        Predict action based on current policy
        根据当前策略预测动作

        Args:
            list_obs_data: List of observation data / 观测数据列表

        Returns:
            List of action data / 动作数据列表
        """
        state = int(list_obs_data[0].feature)
        action = int(np.argmax(self.algorithm.agent_policy[state]))

        return [ActData(act=action)]

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
        act_data = ActData(act=int(np.argmax(self.algorithm.agent_policy[state])))
        action = self.action_process(act_data)
        return action

    def learn(self, state_transition_function):
        """
        Learn optimal policy using dynamic programming
        使用动态规划学习最优策略

        Args:
            state_transition_function: State transition probability matrix / 状态转移概率矩阵
        """
        self.algorithm.learn(state_transition_function)

    def observation_process(self, env_obs):
        """
        Process environment observation into feature representation
        将环境观测处理为特征表示

        Note: By default, only positional information is used as features. If additional feature processing
        is performed, corresponding modifications are needed for the Policy structure, predict, exploit,
        and learn methods of the algorithm.
        注意：默认仅使用位置信息作为特征。如进行额外特征处理，则需要对算法的Policy结构、
        predict、exploit、learn进行相应的改动。

        Args:
            env_obs: Environment observation / 环境观测

        Returns:
            ObsData with processed features / 处理后的观测数据
        """
        obs = env_obs["observation"]
        pos = [obs["frame_state"]["hero"]["pos"]["x"], obs["frame_state"]["hero"]["pos"]["z"]]

        # Feature: Current state of the agent (1-dimensional representation)
        # 特征：智能体当前状态（1维表示）
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
        np.save(model_file_path, self.algorithm.agent_policy)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.agent_policy = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"file {model_file_path} not found")
            exit(1)
