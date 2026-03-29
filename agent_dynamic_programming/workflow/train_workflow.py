#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import time
import os
from tools.map_data_utils import read_map_data
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    Training workflow for Dynamic Programming agent
    动态规划智能体的训练工作流

    Args:
        envs: List of environment instances / 环境实例列表
        agents: List of agent instances / 智能体实例列表
        logger: Logger instance / 日志记录器实例
        monitor: Monitor instance for metrics reporting / 监控实例用于指标上报
    """
    try:
        # Read and validate configuration file
        # 读取并验证配置文件
        usr_conf = read_usr_conf("agent_dynamic_programming/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_dynamic_programming/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]

        # Initialize monitoring data
        # 初始化监控数据
        monitor_data = {
            "reward": 0,
        }

        logger.info("Start Training...")
        start_time = time.time()

        # Load state transition function from map data
        # 从地图数据加载状态转移函数
        map_data_file = "conf/map_data/F_level_1.json"
        map_data = read_map_data(map_data_file)
        if map_data is None:
            logger.error(f"Failed to read map_data from file {map_data_file}, please check")
            return

        # Train agent using dynamic programming
        # 使用动态规划训练智能体
        agent.learn(map_data)

        logger.info(f"Training time cost: {time.time() - start_time} s")

        # Report training progress to monitor
        # 上报训练进度到监控系统
        monitor_data["reward"] = 0
        if monitor:
            monitor.put_data({os.getpid(): monitor_data})

        # Save trained model
        # 保存训练好的模型
        agent.save_model()

        # Retrieve and log training metrics
        # 获取并记录训练指标
        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

    except Exception as e:
        raise RuntimeError(f"workflow error: {e}")
