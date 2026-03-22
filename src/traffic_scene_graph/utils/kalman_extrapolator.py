"""
卡尔曼滤波外推模块：利用跟踪器的卡尔曼滤波状态，外推实体未来 1-2 秒的预测边界框。

状态向量: [x, y, w, h, vx, vy]
观测向量: [x, y, w, h]
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Dict, Tuple, Optional


class KalmanExtrapolator:
    """基于卡尔曼滤波的目标状态外推器。

    为每个跟踪目标维护一个独立的 KalmanFilter，
    支持预测未来 N 步的边界框位置。
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 1.0):
        """
        Args:
            dt: 时间步长（帧间隔）
            process_noise: 过程噪声系数
        """
        self.dt = dt
        self.process_noise = process_noise
        self.trackers: Dict[int, KalmanFilter] = {}

    def _create_filter(self) -> KalmanFilter:
        """创建一个新的卡尔曼滤波器。

        状态向量: [x, y, w, h, vx, vy] (6维)
        观测向量: [x, y, w, h] (4维)
        """
        kf = KalmanFilter(dim_x=6, dim_z=4)

        # 状态转移矩阵 F: 匀速运动模型
        kf.F = np.array(
            [
                [1, 0, 0, 0, self.dt, 0],       # x += vx * dt
                [0, 1, 0, 0, 0, self.dt],        # y += vy * dt
                [0, 0, 1, 0, 0, 0],              # w 不变
                [0, 0, 0, 1, 0, 0],              # h 不变
                [0, 0, 0, 0, 1, 0],              # vx 不变
                [0, 0, 0, 0, 0, 1],              # vy 不变
            ],
            dtype=np.float64,
        )

        # 观测矩阵 H: 只观测 [x, y, w, h]
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # 测量噪声
        kf.R *= 10.0

        # 过程噪声
        kf.Q = np.eye(6, dtype=np.float64) * self.process_noise
        kf.Q[4:, 4:] *= 2.0  # 速度的过程噪声稍大

        # 初始协方差
        kf.P *= 100.0

        return kf

    def update(self, track_id: int, x: float, y: float, w: float, h: float,
               vx: float = 0.0, vy: float = 0.0):
        """更新指定目标的卡尔曼滤波器状态。

        Args:
            track_id: 目标跟踪 ID
            x, y: 边界框中心坐标
            w, h: 边界框宽高
            vx, vy: 速度矢量（用于初始化）
        """
        if track_id not in self.trackers:
            kf = self._create_filter()
            # 初始状态
            kf.x = np.array([[x], [y], [w], [h], [vx], [vy]], dtype=np.float64)
            self.trackers[track_id] = kf
        else:
            kf = self.trackers[track_id]
            kf.predict()
            kf.update(np.array([[x], [y], [w], [h]], dtype=np.float64))

    def predict_future_bbox(
        self, track_id: int, steps: int = 5
    ) -> Optional[np.ndarray]:
        """外推目标未来 N 步的边界框。

        Args:
            track_id: 目标跟踪 ID
            steps: 外推步数

        Returns:
            future_bboxes: (steps, 4) 数组 [x, y, w, h]
            如果目标不存在则返回 None
        """
        if track_id not in self.trackers:
            return None

        kf = self.trackers[track_id]
        # 保存当前状态（外推不应修改原始状态）
        x_save = kf.x.copy()
        P_save = kf.P.copy()

        future_bboxes = []
        for _ in range(steps):
            kf.predict()
            # 提取 [x, y, w, h]
            bbox = kf.x[:4, 0].copy()
            # 确保宽高非负
            bbox[2] = max(bbox[2], 1.0)
            bbox[3] = max(bbox[3], 1.0)
            future_bboxes.append(bbox)

        # 恢复原始状态
        kf.x = x_save
        kf.P = P_save

        return np.array(future_bboxes)

    def get_state(self, track_id: int) -> Optional[np.ndarray]:
        """获取目标当前滤波状态。

        Returns:
            state: (6,) [x, y, w, h, vx, vy] 或 None
        """
        if track_id not in self.trackers:
            return None
        return self.trackers[track_id].x[:, 0].copy()

    def reset(self):
        """清空所有跟踪状态。"""
        self.trackers.clear()

    def remove_track(self, track_id: int):
        """移除指定目标的跟踪器。"""
        self.trackers.pop(track_id, None)
