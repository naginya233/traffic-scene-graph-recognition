"""
虚拟环境区域管理模块。

在 BEV 空间中划分九宫格 (3×3) 区域，为每个区域生成虚拟节点特征，
用于建模交通实体与环境之间的关系 (in_zone / entering_zone / leaving_zone)。
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


class ZoneManager:
    """BEV 空间虚拟环境区域管理器。"""

    # 区域名称映射 (行优先, 从左到右、从上到下)
    ZONE_NAMES = [
        "top_left", "top_center", "top_right",
        "mid_left", "center", "mid_right",
        "bot_left", "bot_center", "bot_right",
    ]

    def __init__(
        self,
        bev_range: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 100.0),
        grid_rows: int = 3,
        grid_cols: int = 3,
        node_feature_dim: int = 12,
    ):
        """
        Args:
            bev_range: BEV 空间范围 (x_min, y_min, x_max, y_max) 单位:米
            grid_rows: 行数
            grid_cols: 列数
            node_feature_dim: 节点特征维度
        """
        self.x_min, self.y_min, self.x_max, self.y_max = bev_range
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.node_feature_dim = node_feature_dim

        self.cell_w = (self.x_max - self.x_min) / grid_cols
        self.cell_h = (self.y_max - self.y_min) / grid_rows
        self.num_zones = grid_rows * grid_cols

        # 预计算每个区域的中心点
        self._zone_centers = self._compute_centers()

    def _compute_centers(self) -> np.ndarray:
        """计算每个区域的中心坐标。返回 (num_zones, 2)。"""
        centers = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cx = self.x_min + (c + 0.5) * self.cell_w
                cy = self.y_min + (r + 0.5) * self.cell_h
                centers.append([cx, cy])
        return np.array(centers, dtype=np.float64)

    def get_zone_index(self, bev_pos: np.ndarray) -> np.ndarray:
        """确定每个 BEV 点所在的区域索引。

        Args:
            bev_pos: (N, 2) BEV 坐标

        Returns:
            (N,) 区域索引 [0, num_zones-1]，超出范围则 clip 到最近区域
        """
        pos = np.array(bev_pos, dtype=np.float64).reshape(-1, 2)
        col = np.clip(
            ((pos[:, 0] - self.x_min) / self.cell_w).astype(int),
            0, self.grid_cols - 1,
        )
        row = np.clip(
            ((pos[:, 1] - self.y_min) / self.cell_h).astype(int),
            0, self.grid_rows - 1,
        )
        return row * self.grid_cols + col

    def get_zone_name(self, zone_idx: int) -> str:
        """获取区域名称。"""
        if 0 <= zone_idx < len(self.ZONE_NAMES):
            return self.ZONE_NAMES[zone_idx]
        return f"zone_{zone_idx}"

    def generate_zone_features(self) -> torch.Tensor:
        """生成所有区域的虚拟节点特征向量。

        特征格式 (dim=12):
          [0:5]  one-hot 类别 (全0, 区域非交通目标)
          [5:7]  归一化 BEV 中心坐标
          [7:9]  归一化 BEV 尺寸 (cell_w, cell_h)
          [9:11] 速度 (0,0) — 区域静止
          [11]   is_zone 标志 = 1.0

        Returns:
            (num_zones, node_feature_dim) 张量
        """
        features = torch.zeros(self.num_zones, self.node_feature_dim)

        range_w = self.x_max - self.x_min
        range_h = self.y_max - self.y_min

        for i, (cx, cy) in enumerate(self._zone_centers):
            # 归一化中心坐标 [0, 1]
            features[i, 5] = (cx - self.x_min) / range_w
            features[i, 6] = (cy - self.y_min) / range_h
            # 归一化区域尺寸
            features[i, 7] = self.cell_w / range_w
            features[i, 8] = self.cell_h / range_h
            # is_zone 标志
            features[i, 11] = 1.0

        return features

    def build_entity_zone_edges(
        self,
        entity_bev_pos: np.ndarray,
        num_entity_nodes: int,
        zone_node_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建实体节点与区域节点之间的边。

        每个实体连接其所在区域节点。边特征为相对位置差。

        Args:
            entity_bev_pos: (N, 2) 实体 BEV 坐标
            num_entity_nodes: 实体节点数量
            zone_node_offset: 区域节点在节点列表中的起始偏移

        Returns:
            edge_index: (2, E) 边索引
            edge_attr: (E, 4) 边特征 [Δx, Δy, 0, 0]
        """
        pos = np.array(entity_bev_pos, dtype=np.float64).reshape(-1, 2)
        zone_indices = self.get_zone_index(pos)

        src_list = []
        dst_list = []
        attr_list = []

        for entity_idx in range(pos.shape[0]):
            zone_idx = zone_indices[entity_idx]
            zone_center = self._zone_centers[zone_idx]

            # 实体 → 区域
            src_list.append(entity_idx)
            dst_list.append(zone_node_offset + zone_idx)

            dx = zone_center[0] - pos[entity_idx, 0]
            dy = zone_center[1] - pos[entity_idx, 1]
            attr_list.append([dx, dy, 0.0, 0.0])

            # 区域 → 实体 (双向边)
            src_list.append(zone_node_offset + zone_idx)
            dst_list.append(entity_idx)
            attr_list.append([-dx, -dy, 0.0, 0.0])

        if len(src_list) == 0:
            return (
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 4, dtype=torch.float32),
            )

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(attr_list, dtype=torch.float32)
        return edge_index, edge_attr

    @property
    def zone_centers(self) -> np.ndarray:
        """返回 (num_zones, 2) 区域中心坐标。"""
        return self._zone_centers.copy()

    def __repr__(self) -> str:
        return (
            f"ZoneManager({self.grid_rows}x{self.grid_cols}, "
            f"range=[{self.x_min},{self.y_min}]-[{self.x_max},{self.y_max}])"
        )
