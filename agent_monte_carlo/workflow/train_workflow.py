#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import time
import math
import os
from common_python.utils.common_func import Frame
from agent_monte_carlo.feature.definition import sample_process, reward_shaping
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    Training workflow for Monte Carlo agent
    蒙特卡洛智能体的训练工作流

    Args:
        envs: List of environment instances / 环境实例列表
        agents: List of agent instances / 智能体实例列表
        logger: Logger instance / 日志记录器实例
        monitor: Monitor instance for metrics reporting / 监控实例用于指标上报
    """
    try:
        # Read and validate configuration file
        # 读取并验证配置文件
        usr_conf = read_usr_conf("agent_monte_carlo/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_monte_carlo/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        EPISODES = 1000

        # Initialize monitoring data
        # 初始化监控数据
        monitor_data = {
            "reward": 0,
        }
        last_report_monitor_time = 0
        last_get_training_metrics_time = 0

        logger.info("Start Training...")
        start_time = time.time()
        last_save_model_time = start_time

        # Initialize training statistics
        # 初始化训练统计数据
        total_reward = 0
        episode_count = 0
        win_count = 0

        for episode in range(EPISODES):
            # Retrieve training metrics every 15 seconds
            # 每15秒获取训练指标
            if time.time() - last_get_training_metrics_time > 15:
                last_get_training_metrics_time = time.time()
                training_metrics = get_training_metrics()
                if training_metrics:
                    logger.info(f"training_metrics is {training_metrics}")

            # Reset environment and get initial observation
            # 重置环境并获取初始观测
            env_obs = env.reset(usr_conf=usr_conf)

            # Handle disaster recovery
            # 处理容灾
            if handle_disaster_recovery(env_obs, logger):
                continue

            # Adjust exploration rate based on win rate (adaptive epsilon)
            # 根据胜率自适应调整探索率
            agent.epsilon = max(0.1, math.exp(-0.5 / (1 - win_count / (episode + 1))))

            # Episode loop: collect full episode trajectory
            # 回合循环：收集完整回合轨迹
            done = False
            episode_trajectory = []

            while not done:
                # Process current observation
                # 处理当前观测
                obs_data = agent.observation_process(env_obs)

                # Select action using epsilon-greedy policy
                # 使用epsilon-greedy策略选择动作
                act_data = agent.predict(list_obs_data=[obs_data])
                act_data = act_data[0]

                # Extract executable action
                # 提取可执行动作
                current_action = agent.action_process(act_data)

                # Execute action and get next observation
                # 执行动作并获取下一个观测
                env_reward, env_obs = env.step(current_action)

                # Handle disaster recovery
                # 处理容灾
                if handle_disaster_recovery(env_obs, logger):
                    break

                terminated, truncated = env_obs["terminated"], env_obs["truncated"]

                # Calculate shaped reward
                # 计算塑形后的奖励
                reward = reward_shaping(env_reward, env_obs)

                # Check if episode is done and update win count
                # 检查回合是否结束并更新胜利计数
                done = terminated or truncated
                if terminated:
                    win_count += 1
                    # Last frame has no action
                    # 最后一帧没有动作
                    current_action = None

                # Store trajectory frame
                # 存储轨迹帧
                frame = Frame(state=obs_data.feature, action=current_action, reward=reward)
                episode_trajectory.append(frame)
                total_reward += reward

            # Process episode trajectory
            # 处理回合轨迹
            episode_trajectory = sample_process(episode_trajectory)

            # Update policy using Monte Carlo learning
            # 使用蒙特卡洛学习更新策略
            agent.learn(episode_trajectory)

            # Update episode counter
            # 更新回合计数器
            episode_count += 1
            now = time.time()

            # Check convergence criterion
            # 检查收敛条件
            is_converged = win_count / (episode + 1) > 0.9 and episode > 200

            # Report training progress every 15 seconds or upon convergence
            # 每15秒或收敛时上报训练进度
            if now - last_report_monitor_time > 15 or is_converged:
                avg_reward = total_reward / episode_count
                logger.info(f"Episode: {episode + 1}, Avg Reward: {avg_reward}")
                logger.info(f"Training Win Rate: {win_count / (episode + 1)}")
                monitor_data["reward"] = avg_reward
                if monitor:
                    monitor.put_data({os.getpid(): monitor_data})

                total_reward = 0
                episode_count = 0
                last_report_monitor_time = now

            # Stop training if converged
            # 如果收敛则停止训练
            if is_converged:
                logger.info(f"Training Converged at Episode: {episode + 1}")
                break

            # Save model checkpoint every 5 minutes
            # 每5分钟保存模型检查点
            if now - last_save_model_time > 300:
                logger.info(f"Saving Model at Episode: {episode + 1}")
                agent.save_model()
                last_save_model_time = now

        end_time = time.time()
        logger.info(f"Training Time for {episode + 1} episodes: {end_time - start_time} s")
        agent.episodes = episode + 1

        # Save final model
        # 保存最终模型
        agent.save_model()

    except Exception as e:
        raise RuntimeError(f"workflow error: {e}")
