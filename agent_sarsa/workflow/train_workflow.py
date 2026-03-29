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
from agent_sarsa.feature.definition import sample_process, reward_shaping
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    Training workflow for SARSA agent
    SARSA智能体的训练工作流

    Args:
        envs: List of environment instances / 环境实例列表
        agents: List of agent instances / 智能体实例列表
        logger: Logger instance / 日志记录器实例
        monitor: Monitor instance for metrics reporting / 监控实例用于指标上报
    """
    try:
        # Read and validate configuration file
        # 读取并验证配置文件
        usr_conf = read_usr_conf("agent_sarsa/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_sarsa/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        EPISODES = 10000

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

            # Process first frame observation and select initial action
            # 处理首帧观测并选择初始动作
            obs_data = agent.observation_process(env_obs)
            act_data = agent.predict(list_obs_data=[obs_data])
            act_data = act_data[0]

            # Extract executable action
            # 提取可执行动作
            current_action = agent.action_process(act_data)

            # Episode loop
            # 回合循环
            done = False
            agent.epsilon = 1.0

            while not done:
                # Decay exploration rate exponentially
                # 指数衰减探索率
                agent.epsilon = max(0.1, agent.epsilon * math.exp(-(1 / EPISODES) * episode))

                # Execute action and get next observation
                # 执行动作并获取下一个观测
                next_env_reward, next_env_obs = env.step(current_action)

                # Handle disaster recovery
                # 处理容灾
                if handle_disaster_recovery(next_env_obs, logger):
                    break

                terminated, truncated = next_env_obs["terminated"], next_env_obs["truncated"]

                # Process next observation
                # 处理下一个观测
                next_obs_data = agent.observation_process(next_env_obs)

                # Calculate shaped reward
                # 计算塑形后的奖励
                reward = reward_shaping(next_env_reward, next_env_obs)

                # Check if episode is done and update win count
                # 检查回合是否结束并更新胜利计数
                done = terminated or truncated
                if terminated:
                    win_count += 1
                    next_action = -1
                else:
                    # Select next action for SARSA update
                    # 为SARSA更新选择下一个动作
                    next_act_data = agent.predict(list_obs_data=[next_obs_data])
                    next_act_data = next_act_data[0]
                    next_action = agent.action_process(next_act_data)

                # Create training sample
                # 创建训练样本
                sample = Frame(
                    state=obs_data.feature,
                    action=current_action,
                    reward=reward,
                    next_state=next_obs_data.feature,
                    next_action=next_action,
                )

                # Process and learn from sample
                # 处理样本并学习
                sample = sample_process([sample])
                agent.learn(sample)

                # Update cumulative reward and transition to next state
                # 更新累积奖励并转移到下一个状态
                total_reward += reward
                obs_data = next_obs_data
                current_action = next_action

            # Update episode counter
            # 更新回合计数器
            episode_count += 1
            now = time.time()

            # Check convergence criterion
            # 检查收敛条件
            is_converged = win_count / (episode + 1) > 0.9 and episode > 100

            # Report training progress every 15 seconds or upon convergence
            # 每15秒或收敛时上报训练进度
            if is_converged or now - last_report_monitor_time > 15:
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
