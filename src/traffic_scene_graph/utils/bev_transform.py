"""
BEV (Bird's Eye View) 鸟瞰变换模块。

功能:
  - 从 DAIR-V2X 相机标定参数 (cam_K + R + t) 自动推导单应矩阵 H
  - 从手动四点标定计算 H
  - 默认梯形→矩形估算 (fallback)
  - 像素坐标 → BEV 地面坐标 (米)
  - 像素速度 → BEV 速度 (米/帧)

数学推导:
  投影矩阵 P = K @ [R | t]  (3x4)
  地面平面假设 Z_lidar = ground_z
  消去 Z 后得到 3x3 单应矩阵:
    H_cam2ground = K @ [r1, r2, R@[0,0,ground_z] + t]
  H_inv 用于 像素→BEV 反投影
"""

import json
import numpy as np
import cv2
from typing import Optional, List, Tuple, Union
from pathlib import Path


class BEVTransform:
    """BEV 透视变换：像素坐标 ↔ 地面坐标 (米)。"""

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        """
        初始化 BEV 变换。

        三种方式 (优先级从高到低):
          1. 直接传入 homography_matrix (3x3)
          2. 传入 src_points (4个像素点) + dst_points (4个BEV坐标)
          3. 都不传 → 使用默认梯形估算

        Args:
            homography_matrix: 3x3 像素→BEV 单应矩阵
            src_points: (4, 2) 像素参考点
            dst_points: (4, 2) BEV 参考点 (米)
            frame_width: 画面宽度 (仅 fallback 使用)
            frame_height: 画面高度 (仅 fallback 使用)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        if homography_matrix is not None:
            self.H = np.array(homography_matrix, dtype=np.float64).reshape(3, 3)
        elif src_points is not None and dst_points is not None:
            src = np.array(src_points, dtype=np.float32).reshape(4, 2)
            dst = np.array(dst_points, dtype=np.float32).reshape(4, 2)
            self.H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        else:
            self.H = self._default_homography(frame_width, frame_height)

        # 反向矩阵: BEV → 像素
        self.H_inv = np.linalg.inv(self.H)

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_dair_calibration(
        cls,
        intrinsic_path: str,
        extrinsic_path: str,
        ground_z: float = -1.0,
    ) -> "BEVTransform":
        """从 DAIR-V2X 标定 JSON 自动计算单应矩阵。

        Args:
            intrinsic_path: camera_intrinsic JSON 路径
            extrinsic_path: virtuallidar_to_camera JSON 路径
            ground_z: 地面在 lidar 坐标系中的 Z 值 (默认 -1.0m)

        Returns:
            BEVTransform 实例
        """
        # --- 加载内参 ---
        with open(intrinsic_path, "r") as f:
            intrinsic = json.load(f)

        cam_K = np.array(intrinsic["cam_K"], dtype=np.float64).reshape(3, 3)
        width = int(intrinsic.get("width", 1920))
        height = int(intrinsic.get("height", 1080))

        # --- 加载外参 ---
        with open(extrinsic_path, "r") as f:
            extrinsic = json.load(f)

        R = np.array(extrinsic["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.array(extrinsic["translation"], dtype=np.float64).reshape(3, 1)

        # --- 推导地面平面单应矩阵 ---
        # 从 lidar 坐标 (X, Y, ground_z) → 相机坐标:
        #   p_cam = R @ p_lidar + t
        #   p_pixel = K @ p_cam (齐次除法)
        #
        # 令 Z_lidar = ground_z, 则:
        #   p_cam = R @ [X, Y, ground_z]^T + t
        #         = r1*X + r2*Y + (r3*ground_z + t)
        #
        # 投影: s*[u,v,1]^T = K @ [r1, r2, r3*gz+t] @ [X, Y, 1]^T
        #
        # H_ground2pixel = K @ [r1 | r2 | r3*ground_z + t]
        # H_pixel2ground = inv(H_ground2pixel)

        r1 = R[:, 0:1]  # (3,1)
        r2 = R[:, 1:2]  # (3,1)
        r3 = R[:, 2:3]  # (3,1)

        # 3x3: [r1, r2, r3*gz + t]
        H_ground2pixel = cam_K @ np.hstack([r1, r2, r3 * ground_z + t])

        # H: 像素 → BEV (地面, 米)
        H_pixel2ground = np.linalg.inv(H_ground2pixel)

        instance = cls.__new__(cls)
        instance.frame_width = width
        instance.frame_height = height
        instance.H = H_pixel2ground
        instance.H_inv = H_ground2pixel
        return instance

    @classmethod
    def from_calibration_file(cls, json_path: str) -> "BEVTransform":
        """从之前保存的标定 JSON 加载。

        JSON 格式:
          {"homography": [[...]], "frame_size": [w, h]}
          或
          {"src_points": [...], "dst_points": [...], "frame_size": [w, h]}
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        w = data.get("frame_size", [1920, 1080])[0]
        h = data.get("frame_size", [1920, 1080])[1]

        if "homography" in data:
            H = np.array(data["homography"], dtype=np.float64)
            return cls(homography_matrix=H, frame_width=w, frame_height=h)
        elif "src_points" in data and "dst_points" in data:
            return cls(
                src_points=data["src_points"],
                dst_points=data["dst_points"],
                frame_width=w,
                frame_height=h,
            )
        else:
            raise ValueError(f"标定文件格式无效: {json_path}")

    # ------------------------------------------------------------------
    # 坐标变换
    # ------------------------------------------------------------------

    def pixel_to_bev(self, points: np.ndarray) -> np.ndarray:
        """像素坐标 → BEV 地面坐标 (米)。

        Args:
            points: (N, 2) 像素坐标 [[x, y], ...]

        Returns:
            (N, 2) BEV 坐标 [[X_m, Y_m], ...]
        """
        pts = np.array(points, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        # cv2.perspectiveTransform 需要 (1, N, 2)
        pts_reshaped = pts.reshape(1, -1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(pts_reshaped, self.H.astype(np.float64))
        return result.reshape(-1, 2)

    def bev_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """BEV 地面坐标 → 像素坐标。

        Args:
            points: (N, 2) BEV 坐标 [[X_m, Y_m], ...]

        Returns:
            (N, 2) 像素坐标 [[x, y], ...]
        """
        pts = np.array(points, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        pts_reshaped = pts.reshape(1, -1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(pts_reshaped, self.H_inv.astype(np.float64))
        return result.reshape(-1, 2)

    def velocity_to_bev(
        self, pixel_pos: np.ndarray, pixel_vel: np.ndarray
    ) -> np.ndarray:
        """像素速度 → BEV 速度 (米/帧)。

        透视变换是非线性的，不能直接用 H 乘速度。
        正确做法: 分别投影 pos 和 pos+vel，然后在 BEV 空间求差。

        Args:
            pixel_pos: (N, 2) 像素位置
            pixel_vel: (N, 2) 像素速度 (像素/帧)

        Returns:
            (N, 2) BEV 速度 (米/帧)
        """
        pos = np.array(pixel_pos, dtype=np.float64).reshape(-1, 2)
        vel = np.array(pixel_vel, dtype=np.float64).reshape(-1, 2)

        if pos.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        bev_pos = self.pixel_to_bev(pos)
        bev_pos_next = self.pixel_to_bev(pos + vel)
        return bev_pos_next - bev_pos

    def bbox_to_bev_footprint(self, bboxes: np.ndarray) -> np.ndarray:
        """将像素 bbox 底边中点投影到 BEV 空间。

        底边中点代表物体在地面的 "脚印" 位置，投影最准确。

        Args:
            bboxes: (N, 4) [x_min, y_min, x_max, y_max]

        Returns:
            (N, 2) BEV 地面坐标
        """
        bboxes = np.array(bboxes, dtype=np.float64).reshape(-1, 4)
        if bboxes.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float64)

        # 底边中点: ((x_min+x_max)/2, y_max)
        bottom_centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2.0,
            bboxes[:, 3],
        ])
        return self.pixel_to_bev(bottom_centers)

    def bbox_to_bev_boxes(self, bboxes: np.ndarray) -> np.ndarray:
        """将像素检测框 [cx, cy, w, h] 转换为 BEV 空间近似框 [x, y, w_m, h_m] (米)。
        主要用于建图外推时的 IoU 计算。
        """
        bboxes = np.array(bboxes, dtype=np.float64).reshape(-1, 4)
        if bboxes.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float64)

        cx = bboxes[:, 0]
        cy = bboxes[:, 1]
        w = bboxes[:, 2]
        h = bboxes[:, 3]

        # 1. 位置: 用底边中心作为其在地面上的真实锚点
        bottom_centers = np.column_stack([cx, cy + h / 2.0])
        bev_centers = self.pixel_to_bev(bottom_centers)

        # 2. 宽度: 通过投影底边的左右端点获取物理横向宽度
        bl = np.column_stack([cx - w / 2.0, cy + h / 2.0])
        br = np.column_stack([cx + w / 2.0, cy + h / 2.0])
        bev_bl = self.pixel_to_bev(bl)
        bev_br = self.pixel_to_bev(br)
        bev_w = np.linalg.norm(bev_br - bev_bl, axis=1)

        # 3. 长度: 由于车辆往往带视角透视，此处根据物理先验(汽车长宽比一般~2.1)估算车长
        bev_h = np.clip(bev_w * 2.1, 2.0, 15.0)

        return np.column_stack([bev_centers[:, 0], bev_centers[:, 1], bev_w, bev_h])


    # ------------------------------------------------------------------
    # 可视化 & 序列化
    # ------------------------------------------------------------------

    def warp_frame(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int] = (800, 800),
        bev_range: Optional[Tuple[float, float, float, float]] = None,
    ) -> np.ndarray:
        """将整帧图像变换为鸟瞰图 (用于可视化调试)。

        Args:
            frame: 输入图像 (H, W, 3)
            output_size: 输出图像尺寸 (W, H)
            bev_range: BEV 空间范围 [x_min, y_min, x_max, y_max] (米)
                       若指定则缩放到输出尺寸

        Returns:
            鸟瞰图图像
        """
        if bev_range is not None:
            x_min, y_min, x_max, y_max = bev_range
            ow, oh = output_size

            # 额外缩放矩阵: BEV米坐标 → 输出像素
            sx = ow / (x_max - x_min)
            sy = oh / (y_max - y_min)
            S = np.array([
                [sx, 0, -x_min * sx],
                [0, sy, -y_min * sy],
                [0, 0, 1],
            ], dtype=np.float64)

            H_combined = S @ self.H
        else:
            H_combined = self.H

        return cv2.warpPerspective(
            frame, H_combined, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def save_calibration(self, json_path: str) -> None:
        """保存标定参数到 JSON。"""
        data = {
            "homography": self.H.tolist(),
            "frame_size": [self.frame_width, self.frame_height],
        }
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _default_homography(width: int, height: int) -> np.ndarray:
        """默认梯形→矩形单应矩阵 (假设固定俯仰角路侧相机)。

        将画面中的梯形区域映射到一个 100m × 100m 的 BEV 矩形区域。
        """
        # 源: 画面中的梯形 (近处宽、远处窄)
        src = np.array([
            [width * 0.1, height * 0.95],   # 左下
            [width * 0.9, height * 0.95],   # 右下
            [width * 0.65, height * 0.3],   # 右上
            [width * 0.35, height * 0.3],   # 左上
        ], dtype=np.float32)

        # 目标: BEV 矩形 (米)
        dst = np.array([
            [0.0, 100.0],    # 左下
            [100.0, 100.0],  # 右下
            [100.0, 0.0],    # 右上
            [0.0, 0.0],      # 左上
        ], dtype=np.float32)

        return cv2.getPerspectiveTransform(src, dst).astype(np.float64)

    def __repr__(self) -> str:
        return (
            f"BEVTransform(frame={self.frame_width}x{self.frame_height}, "
            f"H_det={np.linalg.det(self.H):.4f})"
        )
