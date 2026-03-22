"""
伪标签引擎 + 自训练策略。

基于 BEV 坐标系下的运动学规则生成关系伪标签，
并通过三阶段自训练策略逐步用模型预测替代/增强规则标签。

关系类别 (8 类):
  实体间 (5): independent, following, parallel, approaching, crossing
  实体-环境 (3): in_zone, entering_zone, leaving_zone
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


# 关系类别常量
RELATION_CLASSES = [
    "independent",      # 0: 无明显交互
    "following",        # 1: 跟车
    "parallel",         # 2: 并行
    "approaching",      # 3: 接近中
    "crossing",         # 4: 交叉通过
    "in_zone",          # 5: 在区域内
    "entering_zone",    # 6: 进入区域
    "leaving_zone",     # 7: 离开区域
]

NUM_RELATIONS = len(RELATION_CLASSES)


class RelationLabeler:
    """基于 BEV 运动学规则的伪标签生成器。"""

    def __init__(
        self,
        cos_sim_threshold: float = 0.7,
        crossing_cos_threshold: float = -0.3,
        closing_rate_threshold: float = 2.0,
        distance_near: float = 15.0,
        following_lateral_threshold: float = 3.0,
    ):
        """
        Args:
            cos_sim_threshold: 速度方向余弦相似度阈值 (>此值→同向)
            crossing_cos_threshold: 交叉判定阈值 (<此值→交叉)
            closing_rate_threshold: 接近速率阈值 (米/帧)
            distance_near: 近距离阈值 (米)
            following_lateral_threshold: 跟车横向偏移阈值 (米)
        """
        self.cos_sim_threshold = cos_sim_threshold
        self.crossing_cos_threshold = crossing_cos_threshold
        self.closing_rate_threshold = closing_rate_threshold
        self.distance_near = distance_near
        self.following_lateral_threshold = following_lateral_threshold

    def label_entity_relations(
        self,
        bev_positions: np.ndarray,
        bev_velocities: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """为实体间的边生成伪标签。

        Args:
            bev_positions: (N, 2) BEV 位置 (米)
            bev_velocities: (N, 2) BEV 速度 (米/帧)
            edge_index: (2, E) 边索引

        Returns:
            (E,) 标签索引 [0..4]
        """
        pos = np.array(bev_positions, dtype=np.float64)
        vel = np.array(bev_velocities, dtype=np.float64)
        edges = np.array(edge_index)  # (2, E)
        num_edges = edges.shape[1]

        labels = np.zeros(num_edges, dtype=np.int64)  # 默认 independent

        for e in range(num_edges):
            i, j = edges[0, e], edges[1, e]
            dp = pos[j] - pos[i]          # 位置差
            dist = np.linalg.norm(dp)     # 距离 (米)

            vi = vel[i]
            vj = vel[j]
            speed_i = np.linalg.norm(vi)
            speed_j = np.linalg.norm(vj)

            # --- 跳过几乎静止的节点对 ---
            if speed_i < 0.5 and speed_j < 0.5:
                labels[e] = 0  # independent
                continue

            # --- 速度方向余弦相似度 ---
            if speed_i > 0.5 and speed_j > 0.5:
                cos_sim = np.dot(vi, vj) / (speed_i * speed_j)
            else:
                cos_sim = 0.0

            # --- 接近速率: 相对速度在连线方向的投影 ---
            dv = vj - vi
            if dist > 1e-6:
                closing_rate = -np.dot(dv, dp) / dist
            else:
                closing_rate = 0.0

            # --- 判定逻辑 ---
            if cos_sim < self.crossing_cos_threshold and dist < self.distance_near:
                # 速度方向接近反向/垂直 + 距离近 → crossing
                labels[e] = 4
            elif closing_rate > self.closing_rate_threshold and dist < self.distance_near:
                # 快速接近 → approaching
                labels[e] = 3
            elif cos_sim > self.cos_sim_threshold and dist < self.distance_near:
                # 同向、距离近 → following 或 parallel
                if speed_i > 0.5:
                    # 在 i 的行驶方向上分解 dp 为纵向/横向
                    heading = vi / speed_i
                    lateral = abs(dp[0] * (-heading[1]) + dp[1] * heading[0])
                    if lateral < self.following_lateral_threshold:
                        labels[e] = 1  # following
                    else:
                        labels[e] = 2  # parallel
                else:
                    labels[e] = 2  # parallel (i 几乎静止)
            else:
                labels[e] = 0  # independent

        return labels

    def label_environment_relations(
        self,
        current_zones: np.ndarray,
        track_ids: np.ndarray,
        previous_zones_dict: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """为实体-环境边生成伪标签。

        Args:
            current_zones: (N,) 当前帧每个实体所在区域索引
            track_ids: (N,) 当前帧每个实体的 tracking ID
            previous_zones_dict: 字典, 记录前一帧 {track_id: zone_index} (None→全in_zone)

        Returns:
            (N,) 标签索引 [5..7] (in_zone / entering_zone / leaving_zone)
        """
        n = len(current_zones)
        labels = np.full(n, 5, dtype=np.int64)  # 默认 in_zone

        if previous_zones_dict is not None:
            for i in range(n):
                tid = int(track_ids[i])
                if tid in previous_zones_dict:
                    if current_zones[i] != previous_zones_dict[tid]:
                        # 区域发生变化
                        labels[i] = 6  # entering_zone (进入新区域)

        return labels


class SelfTrainingScheduler:
    """三阶段自训练调度器。

    Stage 1 (warm-up): 完全使用规则伪标签
    Stage 2 (transition): 高置信度模型预测替代规则标签
    Stage 3 (refinement): 降低置信度阈值，扩大模型标签覆盖
    """

    def __init__(
        self,
        total_epochs: int,
        stage1_ratio: float = 0.3,
        stage2_ratio: float = 0.4,
        confidence_threshold_stage2: float = 0.85,
        confidence_threshold_stage3: float = 0.7,
    ):
        self.total_epochs = total_epochs
        self.stage1_end = int(total_epochs * stage1_ratio)
        self.stage2_end = int(total_epochs * (stage1_ratio + stage2_ratio))
        self.conf_thresh_s2 = confidence_threshold_stage2
        self.conf_thresh_s3 = confidence_threshold_stage3

    def get_stage(self, epoch: int) -> int:
        """返回当前阶段: 1, 2, 或 3。"""
        if epoch < self.stage1_end:
            return 1
        elif epoch < self.stage2_end:
            return 2
        else:
            return 3

    def get_confidence_threshold(self, epoch: int) -> float:
        """返回当前阶段的置信度阈值。"""
        stage = self.get_stage(epoch)
        if stage == 1:
            return 1.0  # Stage 1 不用模型标签
        elif stage == 2:
            return self.conf_thresh_s2
        else:
            return self.conf_thresh_s3

    def refine_labels(
        self,
        rule_labels: torch.Tensor,
        model_logits: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """根据自训练阶段精炼伪标签。

        Args:
            rule_labels: (E,) 规则生成的伪标签
            model_logits: (E, C) 模型输出 logits
            epoch: 当前 epoch

        Returns:
            (E,) 精炼后的标签
        """
        stage = self.get_stage(epoch)

        if stage == 1:
            return rule_labels.clone()

        # Stage 2/3: 高置信度模型预测替代规则标签
        probs = torch.softmax(model_logits, dim=-1)
        max_conf, model_preds = probs.max(dim=-1)

        threshold = self.get_confidence_threshold(epoch)
        refined = rule_labels.clone()
        high_conf_mask = max_conf > threshold
        refined[high_conf_mask] = model_preds[high_conf_mask]

        return refined

    def get_classification_weight(self, epoch: int) -> float:
        """返回分类损失的权重 alpha。

        Stage 1: 低权重 (规则可能噪声大)
        Stage 2/3: 逐步提高
        """
        stage = self.get_stage(epoch)
        if stage == 1:
            return 0.3
        elif stage == 2:
            return 0.5
        else:
            return 0.7
