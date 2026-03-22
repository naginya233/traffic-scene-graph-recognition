"""
测试动态拓扑建图模块：验证稀疏建图逻辑、IoU 计算、边特征。
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic_scene_graph.utils import GraphBuilder, KalmanExtrapolator


class TestKalmanExtrapolator:
    """卡尔曼滤波外推器测试。"""

    def test_create_and_update(self):
        """测试创建和更新滤波器。"""
        kf = KalmanExtrapolator()
        kf.update(track_id=1, x=100, y=200, w=80, h=50, vx=5, vy=3)

        state = kf.get_state(1)
        assert state is not None
        assert len(state) == 6

    def test_predict_future_bbox(self):
        """测试未来 bbox 预测形状。"""
        kf = KalmanExtrapolator()
        kf.update(1, 100, 200, 80, 50, 5, 3)

        future = kf.predict_future_bbox(1, steps=5)
        assert future is not None
        assert future.shape == (5, 4)

    def test_predict_nonexistent_track(self):
        """测试查询不存在的 track 返回 None。"""
        kf = KalmanExtrapolator()
        assert kf.predict_future_bbox(999) is None
        assert kf.get_state(999) is None

    def test_state_not_modified_after_predict(self):
        """测试外推不修改原始状态。"""
        kf = KalmanExtrapolator()
        kf.update(1, 100, 200, 80, 50, 5, 3)

        state_before = kf.get_state(1).copy()
        kf.predict_future_bbox(1, steps=10)
        state_after = kf.get_state(1)

        np.testing.assert_array_almost_equal(state_before, state_after)

    def test_reset(self):
        """测试重置功能。"""
        kf = KalmanExtrapolator()
        kf.update(1, 100, 200, 80, 50)
        kf.reset()
        assert kf.get_state(1) is None


class TestGraphBuilder:
    """稀疏图构建器测试。"""

    def test_build_graph_basic(self):
        """测试基础建图。"""
        builder = GraphBuilder(distance_threshold=500.0)

        positions = np.array([[100.0, 200.0], [150.0, 220.0], [800.0, 800.0]])
        velocities = np.array([[5.0, 0.0], [-5.0, 0.0], [0.0, 0.0]])
        bboxes = np.array([
            [100.0, 200.0, 80.0, 50.0],
            [150.0, 220.0, 80.0, 50.0],
            [800.0, 800.0, 80.0, 50.0],
        ])
        track_ids = np.array([1, 2, 3])

        edge_index, edge_attr = builder.build_graph(
            positions, velocities, bboxes, track_ids
        )

        # edge_index 应该是 (2, E) 形状
        assert edge_index.shape[0] == 2
        # edge_attr 应该是 (E, 4) 形状
        assert edge_attr.shape[-1] == 4
        # 无向边：边数应该是偶数
        assert edge_index.shape[1] % 2 == 0

    def test_no_edges_single_node(self):
        """测试单节点无法建边。"""
        builder = GraphBuilder()

        positions = np.array([[100.0, 200.0]])
        velocities = np.array([[5.0, 0.0]])
        bboxes = np.array([[100.0, 200.0, 80.0, 50.0]])
        track_ids = np.array([1])

        edge_index, edge_attr = builder.build_graph(
            positions, velocities, bboxes, track_ids
        )

        assert edge_index.shape[1] == 0
        assert edge_attr.shape[0] == 0

    def test_close_nodes_connect(self):
        """测试距离极近的节点会建边。"""
        builder = GraphBuilder(distance_threshold=100.0)

        positions = np.array([[100.0, 200.0], [110.0, 210.0]])
        velocities = np.array([[5.0, 0.0], [5.0, 0.0]])
        bboxes = np.array([
            [100.0, 200.0, 80.0, 50.0],
            [110.0, 210.0, 80.0, 50.0],
        ])
        track_ids = np.array([1, 2])

        edge_index, edge_attr = builder.build_graph(
            positions, velocities, bboxes, track_ids
        )

        # 距离 ~14.14 < 100, 应该建边
        assert edge_index.shape[1] > 0

    def test_edge_attr_antisymmetry(self):
        """测试无向边的边特征反对称性 (ΔF_ij = -ΔF_ji)。"""
        builder = GraphBuilder(distance_threshold=500.0)

        positions = np.array([[100.0, 200.0], [300.0, 400.0]])
        velocities = np.array([[5.0, 3.0], [-2.0, 1.0]])
        bboxes = np.array([
            [100.0, 200.0, 80.0, 50.0],
            [300.0, 400.0, 80.0, 50.0],
        ])
        track_ids = np.array([1, 2])

        edge_index, edge_attr = builder.build_graph(
            positions, velocities, bboxes, track_ids
        )

        if edge_index.shape[1] >= 2:
            # 反向边的特征应该取反
            np.testing.assert_array_almost_equal(
                edge_attr[0].numpy(), -edge_attr[1].numpy(), decimal=5
            )

    def test_iou_computation(self):
        """测试 IoU 计算正确性。"""
        # 完全重叠
        bbox1 = np.array([100, 100, 50, 50])
        bbox2 = np.array([100, 100, 50, 50])
        assert GraphBuilder._compute_iou(bbox1, bbox2) == pytest.approx(1.0)

        # 完全不重叠
        bbox3 = np.array([0, 0, 10, 10])
        bbox4 = np.array([100, 100, 10, 10])
        assert GraphBuilder._compute_iou(bbox3, bbox4) == pytest.approx(0.0)

        # 部分重叠
        bbox5 = np.array([0, 0, 20, 20])
        bbox6 = np.array([10, 0, 20, 20])
        iou = GraphBuilder._compute_iou(bbox5, bbox6)
        assert 0.0 < iou < 1.0

    def test_reset(self):
        """测试重置功能。"""
        builder = GraphBuilder()
        positions = np.array([[100.0, 200.0], [150.0, 220.0]])
        velocities = np.array([[5.0, 0.0], [-5.0, 0.0]])
        bboxes = np.array([[100.0, 200.0, 80.0, 50.0], [150.0, 220.0, 80.0, 50.0]])
        track_ids = np.array([1, 2])

        builder.build_graph(positions, velocities, bboxes, track_ids)
        builder.reset()
        # 重置后应无报错
        builder.build_graph(positions, velocities, bboxes, track_ids)
