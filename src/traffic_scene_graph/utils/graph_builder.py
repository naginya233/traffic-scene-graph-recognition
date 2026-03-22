"""
动态拓扑建图模块：基于卡尔曼外推的稀疏图构建。

规则:
  - 外推未来 bbox，计算 IoU > 0 的节点对建边
  - 当前空间距离极近(< threshold)的节点对也建边
  - 输出 edge_index (COO) + 边特征 ΔF_ij

边特征 ΔF_ij = [Δx, Δy, Δvx, Δvy]
"""

import numpy as np
import torch
from typing import Tuple, Optional

from .kalman_extrapolator import KalmanExtrapolator


class GraphBuilder:
    """基于运动学外推的稀疏图构建器。"""

    def __init__(
        self,
        kalman_predict_steps: int = 5,
        iou_threshold: float = 0.0,
        distance_threshold: float = 150.0,
    ):
        """
        Args:
            kalman_predict_steps: 卡尔曼外推步数
            iou_threshold: IoU 阈值 (IoU > threshold 建边)
            distance_threshold: 距离阈值 (像素)
        """
        self.kalman_predict_steps = kalman_predict_steps
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.extrapolator = KalmanExtrapolator()

    def build_graph(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        bboxes: np.ndarray,
        track_ids: np.ndarray,
        node_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建单帧的稀疏图。

        Args:
            positions: (N, 2) 中心坐标 [x, y]
            velocities: (N, 2) 速度矢量 [vx, vy]
            bboxes: (N, 4) 边界框 [x, y, w, h]
            track_ids: (N,) 跟踪 ID
            node_features: (N, D) 节点特征 (可选)

        Returns:
            edge_index: (2, E) COO 格式边索引
            edge_attr: (E, 4) 边特征 [Δx, Δy, Δvx, Δvy]
        """
        n = len(positions)

        if n <= 1:
            # 少于 2 个节点无法建边
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 4, dtype=torch.float32),
            )

        # 1. 更新所有目标的卡尔曼滤波器
        for i in range(n):
            tid = int(track_ids[i])
            self.extrapolator.update(
                tid,
                float(bboxes[i, 0]),
                float(bboxes[i, 1]),
                float(bboxes[i, 2]),
                float(bboxes[i, 3]),
                float(velocities[i, 0]),
                float(velocities[i, 1]),
            )

        # 2. 获取未来预测框
        future_bboxes = {}
        for i in range(n):
            tid = int(track_ids[i])
            fb = self.extrapolator.predict_future_bbox(tid, self.kalman_predict_steps)
            if fb is not None:
                future_bboxes[i] = fb

        # 3. 计算节点间连边关系
        src_list = []
        dst_list = []
        edge_features = []

        for i in range(n):
            for j in range(i + 1, n):
                should_connect = False

                # 条件 1: 检查未来预测框的 IoU
                if i in future_bboxes and j in future_bboxes:
                    for step in range(self.kalman_predict_steps):
                        iou = self._compute_iou(
                            future_bboxes[i][step], future_bboxes[j][step]
                        )
                        if iou > self.iou_threshold:
                            should_connect = True
                            break

                # 条件 2: 当前空间距离极近
                if not should_connect:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.distance_threshold:
                        should_connect = True

                if should_connect:
                    # 计算边特征 ΔF_ij = [Δx, Δy, Δvx, Δvy]
                    delta_pos = positions[i] - positions[j]
                    delta_vel = velocities[i] - velocities[j]
                    edge_feat = np.concatenate([delta_pos, delta_vel]).astype(np.float32)

                    # 无向边：添加两个方向
                    src_list.extend([i, j])
                    dst_list.extend([j, i])
                    edge_features.append(edge_feat)
                    # 反向边的特征取反
                    edge_features.append(-edge_feat)

        if len(src_list) == 0:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 4, dtype=torch.float32),
            )

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)

        return edge_index, edge_attr

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个 bbox 的 IoU。

        bbox 格式: [cx, cy, w, h] (中心坐标 + 宽高)
        """
        # 转换为角点坐标
        x1_min = bbox1[0] - bbox1[2] / 2
        y1_min = bbox1[1] - bbox1[3] / 2
        x1_max = bbox1[0] + bbox1[2] / 2
        y1_max = bbox1[1] + bbox1[3] / 2

        x2_min = bbox2[0] - bbox2[2] / 2
        y2_min = bbox2[1] - bbox2[3] / 2
        x2_max = bbox2[0] + bbox2[2] / 2
        y2_max = bbox2[1] + bbox2[3] / 2

        # 交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(
            0, inter_y_max - inter_y_min
        )

        # 并集
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def reset(self):
        """重置卡尔曼滤波器状态。"""
        self.extrapolator.reset()
