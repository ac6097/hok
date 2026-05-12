#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for Gorge Chase PPO (Enhanced).
峡谷追猎 PPO 训练工作流（增强版）。

Enhancements / 增强:
  - Curriculum Learning (4 stages)
  - Train/Val map split (1-8 train, 9-10 val)
  - Enhanced terminal rewards
  - Richer monitoring
"""

import copy
import os
import random
import time

import numpy as np
from ..conf.conf import Config
from ..feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read user config / 读取用户配置
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


# ===================== Curriculum Config Builder =====================

def _build_curriculum_conf(base_conf, episode_cnt, logger=None):
    """Build environment config based on curriculum stage.

    根据课程学习阶段构建环境配置。
    """
    conf = copy.deepcopy(base_conf)
    env_conf = conf.setdefault("env_conf", {})

    stages = Config.CURRICULUM_STAGES  # [0, 2000, 5000, 10000]

    if episode_cnt < stages[1]:
        # Stage 1: Pure survival / 纯生存
        env_conf["treasure_count"] = 0
        env_conf["buff_count"] = 0
        env_conf["monster_interval"] = 2000
        env_conf["monster_speedup"] = 2000
        env_conf["max_step"] = 500
        stage = 1
    elif episode_cnt < stages[2]:
        # Stage 2: Introduce resources / 引入资源
        env_conf["treasure_count"] = 5
        env_conf["buff_count"] = 1
        env_conf["monster_interval"] = 500
        env_conf["monster_speedup"] = 800
        env_conf["max_step"] = 700
        stage = 2
    elif episode_cnt < stages[3]:
        # Stage 3: Standard difficulty / 标准难度
        env_conf["treasure_count"] = 10
        env_conf["buff_count"] = 2
        env_conf["monster_interval"] = 300
        env_conf["monster_speedup"] = 500
        env_conf["max_step"] = 1000
        stage = 3
    else:
        # Stage 4: Hard generalization / 强化泛化
        env_conf["treasure_count"] = random.randint(7, 10)
        env_conf["buff_count"] = random.randint(0, 2)
        env_conf["monster_interval"] = random.randint(200, 400)
        env_conf["monster_speedup"] = random.randint(300, 500)
        env_conf["max_step"] = 1000
        stage = 4

    # Always use training maps with random selection
    env_conf["map"] = [1, 2, 3, 4, 5, 6, 7, 8]
    env_conf["map_random"] = True

    return conf, stage


def _build_val_conf(base_conf):
    """Build validation config (maps 9-10).

    构建验证配置（地图9-10）。
    """
    conf = copy.deepcopy(base_conf)
    env_conf = conf.setdefault("env_conf", {})
    env_conf["map"] = [9, 10]
    env_conf["map_random"] = True
    env_conf["treasure_count"] = 10
    env_conf["buff_count"] = 2
    env_conf["monster_interval"] = 300
    env_conf["monster_speedup"] = 500
    env_conf["max_step"] = 1000
    return conf


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.current_stage = 1

        # Validation tracking
        self.val_interval = 200      # run val every N train episodes
        self.val_episodes = 5        # number of val episodes each time
        self.val_scores = []

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        执行单局对局并 yield 训练样本。
        """
        while True:
            # Periodically fetch training metrics / 定期获取训练指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Run validation episodes periodically
            if self.episode_cnt > 0 and self.episode_cnt % self.val_interval == 0:
                self._run_validation()

            # Build curriculum config for this episode
            train_conf, self.current_stage = _build_curriculum_conf(
                self.usr_conf, self.episode_cnt, self.logger
            )

            # Reset env / 重置环境
            env_obs = self.env.reset(train_conf)

            # Disaster recovery / 容灾处理
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # Reset agent & load latest model / 重置 Agent 并加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Initial observation / 初始观测处理
            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(
                f"Episode {self.episode_cnt} start (stage={self.current_stage})"
            )

            while not done:
                # Predict action / Agent 推理（随机采样）
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Step env / 与环境交互
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery / 容灾处理
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next observation / 处理下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Step reward / 每步即时奖励
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)
                total_reward += float(reward[0])

                # Terminal reward / 终局奖励
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = env_info.get("total_score", 0)
                    treasures_collected = env_info.get("treasures_collected", 0)
                    flash_count = env_info.get("flash_count", 0)

                    if terminated:
                        final_reward[0] = Config.TERMINAL_DEATH
                        result_str = "FAIL"
                    else:
                        final_reward[0] = Config.TERMINAL_SURVIVE
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} stage:{self.current_stage} "
                        f"steps:{step} result:{result_str} "
                        f"score:{total_score:.1f} treasures:{treasures_collected} "
                        f"flashes:{flash_count} reward:{total_reward:.3f}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                # Episode end / 对局结束
                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward

                    # Monitor report / 监控上报
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "curriculum_stage": self.current_stage,
                        }
                        if done:
                            env_info_final = env_obs["observation"]["env_info"]
                            monitor_data["total_score"] = env_info_final.get("total_score", 0)
                            monitor_data["treasures_collected"] = env_info_final.get("treasures_collected", 0)
                            monitor_data["flash_count"] = env_info_final.get("flash_count", 0)
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                # Update state / 状态更新
                obs_data = _obs_data
                remain_info = _remain_info

    def _run_validation(self):
        """Run validation episodes on maps 9-10 with greedy policy.

        在地图9-10上用贪心策略运行验证。
        """
        val_conf = _build_val_conf(self.usr_conf)
        scores = []
        survived = 0

        for i in range(self.val_episodes):
            env_obs = self.env.reset(val_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            done = False
            val_step = 0

            while not done:
                # Use exploit (greedy) for validation
                act = self.agent.exploit(env_obs)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                val_step += 1
                done = terminated or truncated

            if done:
                env_info = env_obs["observation"]["env_info"]
                score = env_info.get("total_score", 0)
                scores.append(score)
                if not terminated:
                    survived += 1

        if scores:
            avg_score = sum(scores) / len(scores)
            survival_rate = survived / len(scores)
            self.logger.info(
                f"[VAL] ep:{self.episode_cnt} avg_score:{avg_score:.1f} "
                f"survival_rate:{survival_rate:.2f} n={len(scores)}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): {
                    "val_avg_score": round(avg_score, 1),
                    "val_survival_rate": round(survival_rate, 2),
                }})
