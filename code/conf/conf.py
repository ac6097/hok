#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO (Enhanced).
峡谷追猎 PPO 配置（增强版）。
"""


class Config:

    # ===================== Feature Dimensions / 特征维度 =====================

    # Scalar features breakdown / 标量特征分解
    HERO_DIM = 6          # pos_x, pos_z, flash_cd, buff_remain, has_buff, flash_available
    PHASE_DIM = 4         # step_norm, is_speedup, steps_until_speedup, steps_until_2nd_monster
    MONSTER_DIM = 7       # is_alive, rel_dx, rel_dz, speed, distance, dir_sin, dir_cos
    TREASURE_DIM = 6      # is_valid, rel_dx, rel_dz, distance, dir_sin, dir_cos
    TREASURE_SLOTS = 3    # top-3 nearest treasures
    BUFF_DIM = 4          # is_available, rel_dx, rel_dz, distance
    BUFF_SLOTS = 2
    SAFETY_DIM = 3        # corridor_score, dead_end_score, encirclement_angle
    HISTORY_DIM = 5       # wall_hit_recent, repeat_visit_norm, new_cell_flag, local_visit_density, recent_loop_score

    SCALAR_DIM = (
        HERO_DIM
        + PHASE_DIM
        + MONSTER_DIM * 2
        + TREASURE_DIM * TREASURE_SLOTS
        + BUFF_DIM * BUFF_SLOTS
        + SAFETY_DIM
        + HISTORY_DIM
    )  # = 6+4+14+18+8+3+5 = 58

    # Spatial features / 空间特征
    SPATIAL_CHANNELS = 4   # passability, treasure, buff, monster_heat
    SPATIAL_SIZE = 21      # 21x21 view window
    SPATIAL_DIM = SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE  # 1764

    # Total observation / 总观测维度
    DIM_OF_OBSERVATION = SCALAR_DIM + SPATIAL_DIM  # 1822

    # ===================== Action Space / 动作空间 =====================

    ACTION_NUM = 16        # 0-7 move, 8-15 flash
    VALUE_NUM = 1

    # ===================== Network Architecture / 网络架构 =====================

    # CNN pathway
    CNN_CHANNELS = [16, 32]
    CNN_POOL_SIZE = 3       # AdaptiveAvgPool2d output size
    CNN_FC_OUT = 128

    # MLP pathway
    SCALAR_HIDDEN = 128
    SCALAR_FC_OUT = 128

    # Fusion
    FUSION_DIM = CNN_FC_OUT + SCALAR_FC_OUT  # 256
    FUSION_HIDDEN = 128

    # ===================== PPO Hyperparameters / PPO 超参数 =====================

    GAMMA = 0.995
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.008
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    GRAD_CLIP_RANGE = 0.5

    # ===================== Reward Constants / 奖励常量 =====================

    SURVIVE_REWARD = 0.01
    MONSTER_DIST_SCALE = 0.15
    TREASURE_APPROACH_SCALE = 0.05    # per grid cell × safety
    BUFF_APPROACH_SCALE = 0.05        # per grid cell × safety
    TREASURE_COLLECT_REWARD = 5.0
    BUFF_COLLECT_REWARD = 4.0
    CORRIDOR_REWARD = 0.01
    DEAD_END_PENALTY = -0.03
    REPEAT_VISIT_PENALTY = -0.02
    NEW_CELL_REWARD = 0.005
    LOCAL_REPEAT_PENALTY = -0.01
    RECENT_LOOP_PENALTY = -0.03

    NO_PROGRESS_PENALTY = -0.008
    NO_PROGRESS_START = 6
    NO_PROGRESS_MAX = 10

    DANGER_PENALTY_SCALE = -0.12
    DANGER_DIST_THRESHOLD = 15.0      # grid cells
    ENCIRCLEMENT_PENALTY = -0.03
    ENCIRCLEMENT_ANGLE_THRESHOLD = 0.6  # normalized, ~108 degrees
    WALL_HIT_PENALTY = -0.05
    FLASH_ESCAPE_REWARD = 1.0
    FLASH_ABUSE_PENALTY = -0.5
    FLASH_ESCAPE_DIST_THRESHOLD = 5.0  # grid cells
    PRE_SPEEDUP_BUFFER_STEPS = 50
    PRE_SPEEDUP_DIST_MULTIPLIER = 1.5
    TERMINAL_DEATH = -15.0
    TERMINAL_SURVIVE = 15.0

    # ===================== Map Constants / 地图常量 =====================

    MAP_SIZE = 128.0
    MAX_DIST = 181.0       # sqrt(128^2 + 128^2)
    MAX_FLASH_CD = 100.0
    MAX_BUFF_DURATION = 50.0
    MAX_MONSTER_SPEED = 3.0

    # ===================== Curriculum / 课程学习 =====================

    CURRICULUM_STAGES = [0, 2000, 5000, 10000]
