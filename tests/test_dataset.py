"""
测试数据加载模块：验证 CSV 加载、节点特征维度、数据格式正确性。
"""

import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic_scene_graph.data import TrafficFrameDataset


@pytest.fixture
def sample_csv(tmp_path):
    """创建临时测试 CSV 文件。"""
    data = {
        "frame_id": [0, 0, 0, 1, 1, 1, 2, 2],
        "track_id": [1, 2, 3, 1, 2, 3, 1, 2],
        "class": ["car", "truck", "pedestrian", "car", "truck", "pedestrian", "car", "truck"],
        "x": [100.0, 300.0, 500.0, 105.0, 295.0, 510.0, 110.0, 290.0],
        "y": [200.0, 400.0, 600.0, 205.0, 395.0, 610.0, 210.0, 390.0],
        "w": [80.0, 120.0, 30.0, 80.0, 120.0, 30.0, 80.0, 120.0],
        "h": [50.0, 70.0, 60.0, 50.0, 70.0, 60.0, 50.0, 70.0],
        "vx": [5.0, -5.0, 10.0, 5.0, -5.0, 10.0, 5.0, -5.0],
        "vy": [5.0, -5.0, 10.0, 5.0, -5.0, 10.0, 5.0, -5.0],
    }
    csv_path = str(tmp_path / "test_detections.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


class TestTrafficFrameDataset:
    """数据集模块测试。"""

    def test_dataset_creation(self, sample_csv):
        """测试数据集能正确创建。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=1)
        assert len(dataset) == 3  # 3 帧

    def test_feature_dimension(self, sample_csv):
        """测试节点特征维度 = 12。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=1)
        assert dataset.get_feature_dim() == 12

    def test_getitem_shape(self, sample_csv):
        """测试 __getitem__ 返回的张量形状。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=1)
        sample = dataset[0]

        assert "node_features" in sample
        assert "num_nodes" in sample
        assert "track_ids" in sample

        # 第 0 帧有 3 个节点
        assert sample["num_nodes"][0].item() == 3
        # 特征维度 = 12
        assert sample["node_features"].shape[-1] == 12

    def test_one_hot_encoding(self, sample_csv):
        """测试类别 One-Hot 编码正确性。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=1)
        sample = dataset[0]

        features = sample["node_features"][0]  # (max_N, 12)
        n = sample["num_nodes"][0].item()

        for i in range(n):
            one_hot = features[i, :5]
            # One-Hot 只有一个 1
            assert one_hot.sum().item() == 1.0
            assert one_hot.max().item() == 1.0

    def test_coordinate_normalization(self, sample_csv):
        """测试坐标归一化到 [0, 1]。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=1)
        sample = dataset[0]

        features = sample["node_features"][0]
        n = sample["num_nodes"][0].item()

        for i in range(n):
            x_norm = features[i, 5].item()
            y_norm = features[i, 6].item()
            assert 0.0 <= x_norm <= 1.0, f"x_norm={x_norm} out of range"
            assert 0.0 <= y_norm <= 1.0, f"y_norm={y_norm} out of range"

    def test_sequence_mode(self, sample_csv):
        """测试序列模式 (sequence_length > 1)。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=2)
        assert len(dataset) == 2  # 3帧, seq_len=2 → 2个序列

        sample = dataset[0]
        assert sample["node_features"].shape[0] == 2  # 2帧
        assert sample["num_nodes"].shape[0] == 2

    def test_frame_count(self, sample_csv):
        """测试总帧数统计。"""
        dataset = TrafficFrameDataset(sample_csv, sequence_length=1)
        assert dataset.get_frame_count() == 3
