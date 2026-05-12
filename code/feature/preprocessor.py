#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and reward design for Gorge Chase PPO (Enhanced).
峡谷追猎 PPO 特征预处理与奖励设计（增强版）。

Features / 特征:
  Scalar (58D): hero(6) + phase(4) + monster×2(14) + treasure×3(18) + buff×2(8) + safety(3) + history(5)
  Spatial (4×21×21 = 1764D): passability + treasure_map + buff_map + monster_heat

Reward / 奖励:
  Dense: survive, monster_dist_shaping, treasure_approach, buff_approach,
         corridor, dead_end, danger, encirclement, repeat_visit, pre_speedup_buffer
  Sparse: treasure_collect, buff_collect, wall_hit, flash_escape, flash_abuse
"""

import math
from collections import deque
import numpy as np
from ..conf.conf import Config

# 8 directions: right, right-up, up, left-up, left, left-down, down, right-down
_DIRS_8 = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]
_MAP_CENTER = 10  # center of 21x21 grid


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.monster_speedup = 500
        self.monster_interval = 300

        # State tracking for reward shaping
        self.last_hero_pos = None
        self.last_min_monster_dist = None
        self.last_nearest_treasure_dist = None
        self.last_nearest_buff_dist = None
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_flash_cd = 0
        self.visit_counts = {}
        self.recent_cells = deque(maxlen=6)
        self.no_progress_steps = 0

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量(1822D)、合法动作掩码(16D)和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation.get("map_info")
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)
        self.monster_speedup = env_info.get("monster_speedup", 500)
        self.monster_interval = env_info.get("monster_interval", 300)

        # === Parse hero ===
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x = float(hero_pos["x"])
        hero_z = float(hero_pos["z"])
        flash_cd = float(hero.get("flash_cooldown", 0))
        buff_remain = float(hero.get("buff_remaining_time", 0))
        treasure_count = int(hero.get("treasure_collected_count", env_info.get("treasures_collected", 0)))
        buff_collected = int(env_info.get("collected_buff", 0))

        # === Parse monsters ===
        monsters = frame_state.get("monsters", [])
        monster_infos = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                m_pos = m["pos"]
                is_in_view = bool(m.get("is_in_view", 0))
                if is_in_view and m_pos.get("x") is not None:
                    mx, mz = float(m_pos["x"]), float(m_pos["z"])
                else:
                    mx, mz = self._estimate_pos_from_bucket(
                        hero_x, hero_z,
                        m.get("hero_l2_distance", 5),
                        m.get("hero_relative_direction", 0),
                    )
                    is_in_view = False
                speed = float(m.get("speed", 1))
                dist = math.sqrt((hero_x - mx) ** 2 + (hero_z - mz) ** 2)
                monster_infos.append({
                    "alive": True,
                    "in_view": is_in_view,
                    "x": mx, "z": mz,
                    "speed": speed,
                    "dist": dist,
                    "rel_dx": mx - hero_x,
                    "rel_dz": mz - hero_z,
                })
            else:
                monster_infos.append({"alive": False, "dist": Config.MAX_DIST})

        # === Parse organs (treasures & buffs) ===
        organs = frame_state.get("organs", [])
        treasures = []
        buffs = []
        for org in organs:
            sub_type = int(org.get("sub_type", 0))
            status = int(org.get("status", 0))
            if status != 1:
                continue
            org_pos = org.get("pos", {})
            ox = float(org_pos.get("x", 0))
            oz = float(org_pos.get("z", 0))
            dist = math.sqrt((hero_x - ox) ** 2 + (hero_z - oz) ** 2)
            info = {"x": ox, "z": oz, "dist": dist, "rel_dx": ox - hero_x, "rel_dz": oz - hero_z}
            if sub_type == 1:
                treasures.append(info)
            elif sub_type == 2:
                buffs.append(info)

        treasures.sort(key=lambda t: t["dist"])
        buffs.sort(key=lambda b: b["dist"])

        # ===================== SCALAR FEATURES (58D) =====================

        scalar = []

        # --- Hero features (6D) ---
        scalar.append(hero_x / Config.MAP_SIZE)
        scalar.append(hero_z / Config.MAP_SIZE)
        scalar.append(_clamp(flash_cd / Config.MAX_FLASH_CD, 0, 1))
        scalar.append(_clamp(buff_remain / Config.MAX_BUFF_DURATION, 0, 1))
        scalar.append(1.0 if buff_remain > 0 else 0.0)
        scalar.append(1.0 if flash_cd == 0 else 0.0)

        # --- Phase features (4D) ---
        step_norm = _clamp(self.step_no / self.max_step, 0, 1)
        is_speedup = 1.0 if self.step_no >= self.monster_speedup else 0.0
        steps_until_speedup = _clamp((self.monster_speedup - self.step_no) / self.max_step, 0, 1)
        steps_until_2nd = _clamp((self.monster_interval - self.step_no) / self.max_step, 0, 1)
        scalar.extend([step_norm, is_speedup, steps_until_speedup, steps_until_2nd])

        # --- Monster features (7D × 2 = 14D) ---
        for i in range(2):
            mi = monster_infos[i]
            if mi["alive"]:
                rel_dx = mi["rel_dx"]
                rel_dz = mi["rel_dz"]
                dist = mi["dist"]
                angle = math.atan2(rel_dz, rel_dx)
                scalar.append(1.0)
                scalar.append(rel_dx / Config.MAP_SIZE)
                scalar.append(rel_dz / Config.MAP_SIZE)
                scalar.append(_clamp(mi["speed"] / Config.MAX_MONSTER_SPEED, 0, 1))
                scalar.append(_clamp(dist / Config.MAX_DIST, 0, 1))
                scalar.append(math.sin(angle))
                scalar.append(math.cos(angle))
            else:
                scalar.extend([0.0] * 7)

        # --- Nearest 3 treasures (6D × 3 = 18D) ---
        for i in range(Config.TREASURE_SLOTS):
            if i < len(treasures):
                t = treasures[i]
                angle = math.atan2(t["rel_dz"], t["rel_dx"])
                scalar.append(1.0)
                scalar.append(t["rel_dx"] / Config.MAP_SIZE)
                scalar.append(t["rel_dz"] / Config.MAP_SIZE)
                scalar.append(_clamp(t["dist"] / Config.MAX_DIST, 0, 1))
                scalar.append(math.sin(angle))
                scalar.append(math.cos(angle))
            else:
                scalar.extend([0.0] * 6)

        # --- 2 Buffs (4D × 2 = 8D) ---
        for i in range(Config.BUFF_SLOTS):
            if i < len(buffs):
                b = buffs[i]
                scalar.append(1.0)
                scalar.append(b["rel_dx"] / Config.MAP_SIZE)
                scalar.append(b["rel_dz"] / Config.MAP_SIZE)
                scalar.append(_clamp(b["dist"] / Config.MAX_DIST, 0, 1))
            else:
                scalar.extend([0.0] * 4)

        # --- Safety metrics (3D) ---
        corridor_score, dead_end_score = self._compute_spatial_safety(map_info)
        encirclement = self._compute_encirclement(monster_infos)
        scalar.extend([corridor_score, dead_end_score, encirclement])

        # --- History features (5D) ---
        wall_hit = 0.0
        if self.last_hero_pos is not None and last_action >= 0:
            dx = hero_x - self.last_hero_pos[0]
            dz = hero_z - self.last_hero_pos[1]
            if abs(dx) < 0.5 and abs(dz) < 0.5:
                wall_hit = 1.0

        cell_key = (int(hero_x), int(hero_z))
        new_cell_flag = 1.0 if self.visit_counts.get(cell_key, 0) == 0 else 0.0
        local_visit_density = self._compute_local_visit_density(cell_key)
        recent_loop_score = self._compute_recent_loop_score(cell_key)
        self.visit_counts[cell_key] = self.visit_counts.get(cell_key, 0) + 1
        repeat_visit_norm = _clamp(self.visit_counts[cell_key] / 10.0, 0, 1)
        scalar.extend([
            wall_hit,
            repeat_visit_norm,
            new_cell_flag,
            local_visit_density,
            recent_loop_score,
        ])

        scalar_feat = np.array(scalar, dtype=np.float32)
        assert scalar_feat.shape[0] == Config.SCALAR_DIM

        # ===================== SPATIAL FEATURES (4×21×21 = 1764D) =====================

        spatial = self._build_spatial_channels(
            map_info, hero_x, hero_z, treasures, buffs, monster_infos
        )
        spatial_flat = spatial.flatten()

        # ===================== CONCATENATE =====================

        feature = np.concatenate([scalar_feat, spatial_flat])

        # ===================== LEGAL ACTION MASK (16D) =====================

        legal_action = self._parse_legal_action(legal_act_raw)

        # ===================== REWARD =====================

        min_monster_dist = min(mi["dist"] for mi in monster_infos if mi["alive"]) if any(mi["alive"] for mi in monster_infos) else Config.MAX_DIST
        nearest_treasure_dist = treasures[0]["dist"] if treasures else Config.MAX_DIST
        nearest_buff_dist = buffs[0]["dist"] if buffs else Config.MAX_DIST

        reward_total = self._compute_reward(
            hero_x=hero_x,
            hero_z=hero_z,
            min_monster_dist=min_monster_dist,
            nearest_treasure_dist=nearest_treasure_dist,
            nearest_buff_dist=nearest_buff_dist,
            treasure_count=treasure_count,
            buff_collected=buff_collected,
            flash_cd=flash_cd,
            wall_hit=wall_hit,
            repeat_visit_norm=repeat_visit_norm,
            new_cell_flag=new_cell_flag,
            local_visit_density=local_visit_density,
            recent_loop_score=recent_loop_score,
            corridor_score=corridor_score,
            dead_end_score=dead_end_score,
            encirclement=encirclement,
            monster_infos=monster_infos,
            treasures=treasures,
        )

        # Update state for next step
        self.last_hero_pos = (hero_x, hero_z)
        self.last_min_monster_dist = min_monster_dist
        self.last_nearest_treasure_dist = nearest_treasure_dist
        self.last_nearest_buff_dist = nearest_buff_dist
        self.last_treasure_count = treasure_count
        self.last_buff_count = buff_collected
        self.last_flash_cd = flash_cd
        self.recent_cells.append(cell_key)

        return feature, legal_action, [reward_total]

    # ===================== SPATIAL HELPERS =====================

    def _build_spatial_channels(self, map_info, hero_x, hero_z, treasures, buffs, monster_infos):
        """Build 4-channel spatial feature map (4×21×21).

        构建4通道空间特征图。
        """
        sz = Config.SPATIAL_SIZE  # 21
        spatial = np.zeros((Config.SPATIAL_CHANNELS, sz, sz), dtype=np.float32)

        # Channel 0: Passability / 通行性
        if map_info is not None and len(map_info) >= sz:
            for r in range(sz):
                for c in range(sz):
                    if r < len(map_info) and c < len(map_info[r]):
                        spatial[0, r, c] = 1.0 if map_info[r][c] != 0 else 0.0

        # Channel 1: Treasure locations / 宝箱位置
        view_x_min = hero_x - _MAP_CENTER
        view_z_min = hero_z - _MAP_CENTER
        for t in treasures:
            lc = int(round(t["x"] - view_x_min))
            lr = int(round(t["z"] - view_z_min))
            if 0 <= lr < sz and 0 <= lc < sz:
                spatial[1, lr, lc] = 1.0

        # Channel 2: Buff locations / Buff位置
        for b in buffs:
            lc = int(round(b["x"] - view_x_min))
            lr = int(round(b["z"] - view_z_min))
            if 0 <= lr < sz and 0 <= lc < sz:
                spatial[2, lr, lc] = 1.0

        # Channel 3: Monster danger heat / 怪物危险热力图
        for mi in monster_infos:
            if not mi["alive"] or not mi.get("in_view", False):
                continue
            lc = int(round(mi["x"] - view_x_min))
            lr = int(round(mi["z"] - view_z_min))
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = lr + dr, lc + dc
                    if 0 <= nr < sz and 0 <= nc < sz:
                        manhattan = abs(dr) + abs(dc)
                        heat = max(0.0, 1.0 - manhattan / 5.0)
                        spatial[3, nr, nc] = max(spatial[3, nr, nc], heat)

        return spatial

    def _compute_spatial_safety(self, map_info):
        """Compute corridor_score and dead_end_score using 8-direction ray probing.

        使用8方向射线探测计算开阔度和死角度（基于原始21×21地图）。
        """
        sz = Config.SPATIAL_SIZE  # 21
        if map_info is None or len(map_info) < sz:
            return 0.5, 0.5

        depths = []
        for dx, dz in _DIRS_8:
            depth = 0
            for step in range(1, 11):  # probe up to 10 cells
                r = _MAP_CENTER + dz * step
                c = _MAP_CENTER + dx * step
                if 0 <= r < sz and 0 <= c < sz and map_info[r][c] != 0:
                    depth += 1
                else:
                    break
            depths.append(depth)

        corridor_score = sum(depths) / (8.0 * 10.0)
        blocked = sum(1 for d in depths if d < 2)
        dead_end_score = blocked / 8.0

        return corridor_score, dead_end_score

    def _compute_encirclement(self, monster_infos):
        """Compute encirclement angle (0-1) from two monsters.

        计算两怪包夹角度（归一化到0-1）。
        """
        alive = [mi for mi in monster_infos if mi["alive"]]
        if len(alive) < 2:
            return 0.0

        m1, m2 = alive[0], alive[1]
        v1x, v1z = m1["rel_dx"], m1["rel_dz"]
        v2x, v2z = m2["rel_dx"], m2["rel_dz"]

        dot = v1x * v2x + v1z * v2z
        mag1 = math.sqrt(v1x ** 2 + v1z ** 2)
        mag2 = math.sqrt(v2x ** 2 + v2z ** 2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            return 0.0

        cos_angle = _clamp(dot / (mag1 * mag2), -1, 1)
        angle = math.acos(float(cos_angle))
        return angle / math.pi  # normalize to [0, 1], 1 = 180° (sandwiched)

    def _compute_local_visit_density(self, cell_key):
        """Measure how heavily the nearby 3x3 area has been revisited.

        计算当前格子周围 3x3 区域的重复访问密度。
        """
        cx, cz = cell_key
        visit_sum = 0.0
        neighbor_cnt = 0
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue
                neighbor_cnt += 1
                visit_sum += self.visit_counts.get((cx + dx, cz + dz), 0)

        if neighbor_cnt == 0:
            return 0.0

        return _clamp((visit_sum / neighbor_cnt) / 5.0, 0, 1)

    def _compute_recent_loop_score(self, cell_key):
        """Detect short loops from the recent cell sequence.

        基于最近路径识别短周期循环。
        """
        history = list(self.recent_cells)
        if not history:
            return 0.0

        candidate_path = history + [cell_key]
        loop_score = 0.0

        for back_steps, weight in ((2, 0.45), (3, 0.35), (4, 0.25), (5, 0.15)):
            if len(candidate_path) > back_steps and candidate_path[-1] == candidate_path[-1 - back_steps]:
                loop_score += weight

        repeat_hits = sum(1 for old_cell in history if old_cell == cell_key)
        loop_score = max(loop_score, repeat_hits / 4.0)

        return _clamp(loop_score, 0, 1)

    def _estimate_pos_from_bucket(self, hero_x, hero_z, bucket, direction):
        """Estimate monster position from distance bucket and direction enum.

        从距离桶和方向枚举估算怪物位置。
        """
        dist_map = {0: 15, 1: 45, 2: 75, 3: 105, 4: 135, 5: 165}
        est_dist = dist_map.get(int(bucket), 90)

        dir_angles = {
            0: 0, 1: 0, 2: -math.pi / 4, 3: -math.pi / 2, 4: -3 * math.pi / 4,
            5: math.pi, 6: 3 * math.pi / 4, 7: math.pi / 2, 8: math.pi / 4,
        }
        angle = dir_angles.get(int(direction), 0)

        mx = hero_x + est_dist * math.cos(angle)
        mz = hero_z + est_dist * math.sin(angle)
        return _clamp(mx, 0, Config.MAP_SIZE), _clamp(mz, 0, Config.MAP_SIZE)

    # ===================== LEGAL ACTION =====================

    def _parse_legal_action(self, legal_act_raw):
        """Parse 16D legal action mask from environment.

        解析环境返回的16维合法动作掩码。
        """
        legal_action = [0] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        # Safety: ensure at least movement actions are available
        if sum(legal_action[:8]) == 0:
            legal_action[:8] = [1] * 8
        return legal_action

    # ===================== REWARD =====================

    def _compute_reward(
        self, hero_x, hero_z, min_monster_dist, nearest_treasure_dist,
        nearest_buff_dist, treasure_count, buff_collected, flash_cd, wall_hit,
        repeat_visit_norm, new_cell_flag, local_visit_density, recent_loop_score,
        corridor_score, dead_end_score, encirclement,
        monster_infos, treasures,
    ):
        """Compute total step reward from all reward components.

        计算所有奖励项的总和。
        """
        reward = 0.0

        # 1. Survive reward / 生存奖励
        reward += Config.SURVIVE_REWARD

        # 2. Monster distance shaping / 怪物距离塑形
        if self.last_min_monster_dist is not None:
            delta_dist = min_monster_dist - self.last_min_monster_dist
            dist_shaping = Config.MONSTER_DIST_SCALE * delta_dist / Config.MAP_SIZE

            # Pre-speedup buffer: amplify distance shaping near speedup transition
            if (self.monster_speedup - Config.PRE_SPEEDUP_BUFFER_STEPS
                    <= self.step_no < self.monster_speedup):
                dist_shaping *= Config.PRE_SPEEDUP_DIST_MULTIPLIER

            reward += dist_shaping

        # 3. Safety score for approach rewards
        safety = self._compute_safety(
            min_monster_dist, monster_infos, treasures, dead_end_score
        )

        # 3.5. Useful-move judgement / 有收益移动判断
        # 用于避免“合理回头/逃命/接近宝箱”被重复访问惩罚误伤。
        delta_treasure = 0.0
        if self.last_nearest_treasure_dist is not None:
            delta_treasure = self.last_nearest_treasure_dist - nearest_treasure_dist

        delta_monster = 0.0
        if self.last_min_monster_dist is not None:
            delta_monster = min_monster_dist - self.last_min_monster_dist

        useful_move = False

        # 安全时：接近宝箱才算明确收益。
        if min_monster_dist >= Config.DANGER_DIST_THRESHOLD:
            if delta_treasure > 0.5:
                useful_move = True
        # 危险时：优先远离怪物，避免为了宝箱继续冒险。
        else:
            if delta_monster > 0.5:
                useful_move = True

        # 新格子不能单独算“有收益”，必须带有目标意义，
        # 否则模型会利用新格子奖励走低效画圈路线。
        if new_cell_flag > 0.5 and (delta_treasure > 0.2 or local_visit_density < 0.25):
            useful_move = True

        # 4. Treasure approach reward / 宝箱接近奖励
        if self.last_nearest_treasure_dist is not None and nearest_treasure_dist < Config.MAX_DIST:
            delta_cells = float(_clamp(self.last_nearest_treasure_dist - nearest_treasure_dist, -5, 5))
            reward += float(Config.TREASURE_APPROACH_SCALE) * delta_cells * float(safety)

        # 5. Buff approach reward / Buff接近奖励
        if self.last_nearest_buff_dist is not None and nearest_buff_dist < Config.MAX_DIST:
            delta_cells = float(_clamp(self.last_nearest_buff_dist - nearest_buff_dist, -5, 5))
            reward += float(Config.BUFF_APPROACH_SCALE) * delta_cells * float(safety)

        # 6. Treasure collect reward / 宝箱收集奖励
        new_treasures = treasure_count - self.last_treasure_count
        if new_treasures > 0:
            reward += Config.TREASURE_COLLECT_REWARD * new_treasures

        # 7. Buff collect reward / Buff收集奖励
        new_buffs = buff_collected - self.last_buff_count
        if new_buffs > 0:
            reward += Config.BUFF_COLLECT_REWARD * new_buffs

        # 8. Wall hit penalty / 撞墙惩罚
        if wall_hit > 0.5:
            reward += Config.WALL_HIT_PENALTY

        # 9. Repeat visit penalty / 重复探访惩罚
        reward += Config.REPEAT_VISIT_PENALTY * repeat_visit_norm

        # 10. New cell reward / 新格子奖励
        # 只有“新格子 + 当前移动有收益”才奖励，避免低效画圈刷新格子奖励。
        if new_cell_flag > 0.5 and useful_move:
            reward += Config.NEW_CELL_REWARD

        # 10.5. No-progress penalty / 连续无进展惩罚
        # 不等绕完整圈才惩罚；连续几步没有接近宝箱/远离怪物等有效进展，就逐步扣分。
        if useful_move:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        if self.no_progress_steps >= Config.NO_PROGRESS_START:
            penalty_scale = min(
                self.no_progress_steps - Config.NO_PROGRESS_START + 1,
                Config.NO_PROGRESS_MAX,
            ) / float(Config.NO_PROGRESS_MAX)
            reward += Config.NO_PROGRESS_PENALTY * penalty_scale

        # 11. Local repeat penalty / 局部重复区域惩罚
        # 只有“局部重复明显 + 当前移动没有带来收益”时才惩罚，避免误伤正常绕路/逃命。
        if local_visit_density > 0.5 and not useful_move:
            reward += Config.LOCAL_REPEAT_PENALTY * local_visit_density

        # 12. Recent loop penalty / 最近循环惩罚
        # 只有“短循环明显 + 当前移动没有带来收益”时才惩罚。
        if recent_loop_score > 0.5 and not useful_move:
            reward += Config.RECENT_LOOP_PENALTY * recent_loop_score

        # 13. Corridor reward / 开阔度奖励
        reward += Config.CORRIDOR_REWARD * corridor_score

        # 14. Dead end penalty / 死角惩罚
        reward += Config.DEAD_END_PENALTY * dead_end_score

        # 15. Danger penalty / 危险惩罚
        if min_monster_dist < Config.DANGER_DIST_THRESHOLD:
            danger_intensity = 1.0 - min_monster_dist / Config.DANGER_DIST_THRESHOLD
            reward += Config.DANGER_PENALTY_SCALE * (danger_intensity ** 2)

        # 16. Encirclement penalty / 包夹惩罚
        if encirclement > Config.ENCIRCLEMENT_ANGLE_THRESHOLD:
            reward += Config.ENCIRCLEMENT_PENALTY

        # 17. Flash escape/abuse reward / 闪现脱险/滥用
        flash_just_used = (self.last_flash_cd == 0 and flash_cd > 0)
        if flash_just_used and self.last_min_monster_dist is not None:
            dist_gain = min_monster_dist - self.last_min_monster_dist
            if dist_gain >= Config.FLASH_ESCAPE_DIST_THRESHOLD:
                reward += Config.FLASH_ESCAPE_REWARD
            else:
                reward += Config.FLASH_ABUSE_PENALTY

        return reward

    def _compute_safety(self, min_monster_dist, monster_infos, treasures, dead_end_score):
        """Compute safety score (0-1) for approach rewards.

        计算安全度用于接近奖励的调制。
        """
        safety = _clamp(min_monster_dist / Config.MAX_DIST, 0, 1)

        if treasures and any(mi["alive"] for mi in monster_infos):
            nearest_t = treasures[0]
            nearest_m = min(
                (mi for mi in monster_infos if mi["alive"]),
                key=lambda mi: mi["dist"]
            )
            t_dx, t_dz = nearest_t["rel_dx"], nearest_t["rel_dz"]
            m_dx, m_dz = nearest_m["rel_dx"], nearest_m["rel_dz"]
            t_mag = math.sqrt(t_dx ** 2 + t_dz ** 2)
            m_mag = math.sqrt(m_dx ** 2 + m_dz ** 2)
            if t_mag > 1e-6 and m_mag > 1e-6:
                cos_sim = (t_dx * m_dx + t_dz * m_dz) / (t_mag * m_mag)
                if cos_sim > 0.5:
                    safety *= 0.3

        if dead_end_score > 0.5:
            safety *= 0.5

        if self.step_no >= self.monster_speedup:
            safety *= 0.6

        return float(_clamp(safety, 0, 1))
