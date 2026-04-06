#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Cache previous frame stats per env to compute progress/stagnation rewards.
_PREV_STEP_CACHE = {}


# Reward shaping constants tuned for treasure-first greedy behavior with finish guarantee.
STEP_PENALTY = 0.06
CHEST_REWARD = 150.0
WIN_REWARD = 220.0
TIMEOUT_PENALTY = 180.0

CHEST_PROGRESS_WEIGHT = 4.2
GOAL_PROGRESS_WEIGHT = 3.4
SECONDARY_PROGRESS_WEIGHT = 0.6

CHEST_VIEW_BONUS = 0.60
OBSTACLE_VIEW_PENALTY = 0.03

STAGNATION_BASE_PENALTY = 1.2
STAGNATION_STEP_PENALTY = 0.25
STAGNATION_CAP = 5

CHEST_GOAL_MARGIN = 2
MODE_HOLD_STEPS = 8
UNKNOWN_POS_CUTOFF = 0.20
UNKNOWN_POS_BONUS = 0.50
NO_TREASURE_EXPLORE_BONUS = 0.20
EXPLORATION_PHASE_STEPS = 1500


def _safe_get_game_info(env_obs):
    obs = env_obs.get("observation", {})
    # Some environments expose score fields under game_info, others under env_info.
    return obs.get("game_info") or obs.get("env_info") or {}


def _extract_position(env_obs):
    hero = env_obs.get("observation", {}).get("frame_state", {}).get("hero", {})
    pos = hero.get("pos", {})
    return int(pos.get("x", 0)), int(pos.get("z", 0))


def _extract_distances(env_obs):
    organs = env_obs.get("observation", {}).get("frame_state", {}).get("organs", [])
    end_dist = None
    nearest_treasure_dist = None
    for organ in organs:
        sub_type = organ.get("sub_type")
        dist = organ.get("dist")
        status = organ.get("status", 0)
        if dist is None:
            continue
        dist_int = int(dist)
        if sub_type == 2:
            end_dist = dist_int
        elif sub_type == 1 and status == 1:
            if nearest_treasure_dist is None or dist_int < int(nearest_treasure_dist):
                nearest_treasure_dist = dist_int
    return end_dist, nearest_treasure_dist


def _extract_local_view_signals(env_obs):
    game_info = _safe_get_game_info(env_obs)
    local_view = game_info.get("local_view") or []
    if not local_view:
        return 0, 0
    treasure_cells = int(sum(1 for v in local_view if v == 4))
    obstacle_cells = int(sum(1 for v in local_view if v == 0))
    return treasure_cells, obstacle_cells


def _extract_location_visit_value(env_obs, x, z):
    game_info = _safe_get_game_info(env_obs)
    location_memory = game_info.get("location_memory") or []
    idx = int(x) * 64 + int(z)
    if not location_memory or idx < 0 or idx >= len(location_memory):
        return 1.0
    try:
        return float(location_memory[idx])
    except (TypeError, ValueError):
        return 1.0


def sample_process(list_game_data):
    """
    Process game data into sample format for training
    将游戏数据处理为训练样本格式

    Args:
        list_game_data: List of game frames / 游戏帧列表

    Returns:
        List of processed samples (dict format) / 处理后的样本列表（字典格式）
    """
    return [
        {
            "state": frame.state,
            "action": frame.action,
            "reward": frame.reward,
            "next_state": frame.next_state,
        }
        for frame in list_game_data
    ]


def reward_shaping(env_reward, env_obs):
    """
    Shape reward signal for better learning
    塑形奖励信号以改善学习效果

    Args:
        env_reward: Original environment reward (unused) / 原始环境奖励（未使用）
        env_obs: Environment observation / 环境观测

    Returns:
        Shaped reward value / 塑形后的奖励值
    """
    game_info = _safe_get_game_info(env_obs)
    terminated = bool(env_obs.get("terminated", False))
    truncated = bool(env_obs.get("truncated", False))
    step_no = int(env_obs.get("observation", {}).get("step_no", 0))
    env_id = env_obs.get("env_id", "default")

    x, z = _extract_position(env_obs)
    end_dist, nearest_treasure_dist = _extract_distances(env_obs)
    treasure_cells, obstacle_cells = _extract_local_view_signals(env_obs)
    location_visit_value = _extract_location_visit_value(env_obs, x, z)
    treasure_count = int(game_info.get("treasure_count", 0))
    score = float(game_info.get("score", 0))
    mode = "CHEST_FIRST"

    # Reset env cache at episode start.
    if step_no <= 1:
        _PREV_STEP_CACHE[env_id] = {
            "x": x,
            "z": z,
            "end_dist": end_dist,
            "nearest_treasure_dist": nearest_treasure_dist,
            "treasure_count": treasure_count,
            "stagnation": 0,
            "mode": mode,
            "last_switch_step": step_no,
        }

    prev = _PREV_STEP_CACHE.get(env_id)
    if prev is None:
        prev = {
            "x": x,
            "z": z,
            "end_dist": end_dist,
            "nearest_treasure_dist": nearest_treasure_dist,
            "treasure_count": treasure_count,
            "stagnation": 0,
            "mode": mode,
            "last_switch_step": step_no,
        }

    mode = str(prev.get("mode") or "CHEST_FIRST")
    last_switch_step = int(prev.get("last_switch_step") or step_no)

    # Greedy target selection: prefer chest when cost is comparable, otherwise return to goal.
    has_available_treasure = nearest_treasure_dist is not None
    desired_mode = "CHEST_FIRST"
    if not has_available_treasure:
        desired_mode = "GOAL_FIRST"
    elif end_dist is not None and nearest_treasure_dist is not None:
        if nearest_treasure_dist > end_dist + CHEST_GOAL_MARGIN:
            desired_mode = "GOAL_FIRST"

    # Hold current mode for a few steps to avoid frequent oscillation from discrete dist changes.
    if desired_mode != mode and (step_no - last_switch_step) >= MODE_HOLD_STEPS:
        mode = desired_mode
        last_switch_step = step_no

    reward = 0.0

    # 1) Per-step penalty to prefer shorter paths.
    reward -= STEP_PENALTY

    # 2) Chest event reward (incremental, avoids repeated credit).
    prev_treasure_count = int(prev.get("treasure_count") or 0)
    treasure_delta = max(0, treasure_count - prev_treasure_count)
    reward += CHEST_REWARD * treasure_delta

    # 3) Progress shaping using discrete shortest-path distances.
    prev_end_dist = prev.get("end_dist")
    end_progress = 0.0
    if prev_end_dist is not None and end_dist is not None:
        end_progress = float(int(prev_end_dist) - int(end_dist))

    prev_t_dist = prev.get("nearest_treasure_dist")
    chest_progress = 0.0
    if prev_t_dist is not None and nearest_treasure_dist is not None:
        chest_progress = float(int(prev_t_dist) - int(nearest_treasure_dist))

    # 4) Greedy dual-target shaping: chest-first when profitable, otherwise goal-first.
    if mode == "CHEST_FIRST":
        reward += CHEST_PROGRESS_WEIGHT * chest_progress
        reward += SECONDARY_PROGRESS_WEIGHT * end_progress
    else:
        reward += GOAL_PROGRESS_WEIGHT * end_progress
        reward += SECONDARY_PROGRESS_WEIGHT * chest_progress

    # Local-view guidance keeps exploration pressure toward visible treasure with mild obstacle aversion.
    reward += CHEST_VIEW_BONUS * float(treasure_cells)
    reward -= OBSTACLE_VIEW_PENALTY * float(obstacle_cells)

    # Encourage visiting unseen areas so hidden treasures can enter local view.
    if location_visit_value < UNKNOWN_POS_CUTOFF:
        novelty_ratio = (UNKNOWN_POS_CUTOFF - location_visit_value) / UNKNOWN_POS_CUTOFF
        reward += UNKNOWN_POS_BONUS * novelty_ratio
        if treasure_cells == 0 and mode == "CHEST_FIRST" and step_no <= EXPLORATION_PHASE_STEPS:
            reward += NO_TREASURE_EXPLORE_BONUS * novelty_ratio

    # In late episode, enforce stronger return-to-goal pressure to reduce timeout risk.
    if step_no > 1400:
        reward += 0.8 * end_progress
        if end_progress <= 0:
            reward -= 0.6

    # 5) Penalize invalid/ineffective moves (position unchanged).
    if x == prev.get("x") and z == prev.get("z"):
        stagnation = int(prev.get("stagnation") or 0) + 1
        reward -= STAGNATION_BASE_PENALTY + min(stagnation, STAGNATION_CAP) * STAGNATION_STEP_PENALTY
    else:
        stagnation = 0

    # 6) Terminal shaping: strongly prefer successful finish before max steps.
    if terminated:
        reward += WIN_REWARD + max(0.0, score) + 35.0 * float(treasure_count)
    elif truncated:
        reward -= TIMEOUT_PENALTY

    # Record current frame as previous for next reward computation.
    _PREV_STEP_CACHE[env_id] = {
        "x": x,
        "z": z,
        "end_dist": end_dist,
        "nearest_treasure_dist": nearest_treasure_dist,
        "treasure_count": treasure_count,
        "stagnation": stagnation,
        "mode": mode,
        "last_switch_step": last_switch_step,
    }

    # Fallback: keep sparse environment signal when game_info fields are unavailable.
    if not game_info and isinstance(env_reward, dict):
        reward += float(env_reward.get("reward", 0))

    return float(reward)
