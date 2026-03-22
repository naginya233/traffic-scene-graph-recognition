"""
端到端集成测试：CSV → 建图 → 模型推理 → 输出关系向量。
"""

import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic_scene_graph.data import TrafficFrameDataset
from traffic_scene_graph.utils import GraphBuilder
from traffic_scene_graph.models import SceneGraphModel


@pytest.fixture
def sample_csv(tmp_path):
    """创建多帧测试数据。"""
    records = []
    np.random.seed(42)

    for frame in range(10):
        for tid in range(5):
            records.append({
                "frame_id": frame,
                "track_id": tid + 1,
                "class": ["car", "truck", "bus", "pedestrian", "cyclist"][tid],
                "x": 200.0 + tid * 100 + frame * 5 + np.random.normal(0, 2),
                "y": 300.0 + tid * 80 + frame * 3 + np.random.normal(0, 2),
                "w": 80.0,
                "h": 50.0,
                "vx": 5.0 + np.random.normal(0, 0.5),
                "vy": 3.0 + np.random.normal(0, 0.5),
            })

    csv_path = str(tmp_path / "e2e_test.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config():
    return {
        "data": {
            "csv_path": "",
            "frame_width": 1920,
            "frame_height": 1080,
        },
        "node_feature": {"input_dim": 12},
        "graph": {
            "kalman_predict_steps": 3,
            "iou_threshold": 0.0,
            "distance_threshold": 500.0,
        },
        "model": {
            "edge_input_dim": 28,
            "edge_hidden_dim": 32,
            "relation_dim": 16,
            "num_heads": 2,
            "num_transformer_layers": 1,
            "dropout": 0.0,
            "gate_hidden_dim": 8,
            "relative_motion_dim": 4,
        },
    }


class TestEndToEnd:
    """端到端管道测试。"""

    def test_full_pipeline(self, sample_csv, config):
        """测试完整流程：CSV → 数据集 → 建图 → 模型推理。"""
        # 1. 加载数据
        dataset = TrafficFrameDataset(
            sample_csv, frame_width=1920, frame_height=1080, sequence_length=1
        )
        assert len(dataset) > 0

        # 2. 初始化建图器和模型
        graph_builder = GraphBuilder(
            kalman_predict_steps=3,
            distance_threshold=500.0,
        )
        model = SceneGraphModel(config)
        model.eval()

        # 3. 处理每一帧
        relation_dims = []
        for idx in range(min(5, len(dataset))):
            sample = dataset[idx]
            n = sample["num_nodes"][0].item()

            if n < 2:
                continue

            nf = sample["node_features"][0, :n]
            pos = sample["raw_positions"][0, :n].numpy()
            vel = sample["raw_velocities"][0, :n].numpy()
            bbox = sample["raw_bboxes"][0, :n].numpy()
            tids = sample["track_ids"][0, :n].numpy()

            # 建图
            edge_index, edge_attr = graph_builder.build_graph(
                pos, vel, bbox, tids
            )

            # 模型推理
            with torch.no_grad():
                tids_tensor = torch.from_numpy(tids).long()
                result = model(nf, edge_index, edge_attr, track_ids=tids_tensor)
                output = result["relation_embeddings"]

            # 验证输出
            if edge_index.shape[1] > 0:
                assert output.shape[0] == edge_index.shape[1]
                assert output.shape[1] == config["model"]["relation_dim"]
                relation_dims.append(output.shape[1])

        # 确保至少处理了一些帧
        assert len(relation_dims) > 0
        # 所有帧的关系向量维度一致
        assert all(d == relation_dims[0] for d in relation_dims)

    def test_temporal_sequence(self, sample_csv, config):
        """测试时序流处理（多帧序列）。"""
        dataset = TrafficFrameDataset(
            sample_csv, frame_width=1920, frame_height=1080, sequence_length=3
        )

        graph_builder = GraphBuilder(distance_threshold=500.0)
        model = SceneGraphModel(config)
        model.eval()

        sample = dataset[0]
        T = sample["node_features"].shape[0]
        assert T == 3

        model.reset_temporal_state()
        graph_builder.reset()

        outputs = []
        for t in range(T):
            n = sample["num_nodes"][t].item()
            if n < 2:
                continue

            nf = sample["node_features"][t, :n]
            pos = sample["raw_positions"][t, :n].numpy()
            vel = sample["raw_velocities"][t, :n].numpy()
            bbox = sample["raw_bboxes"][t, :n].numpy()
            tids = sample["track_ids"][t, :n].numpy()

            edge_index, edge_attr = graph_builder.build_graph(
                pos, vel, bbox, tids
            )

            with torch.no_grad():
                tids_tensor = torch.from_numpy(tids).long()
                result = model(nf, edge_index, edge_attr, track_ids=tids_tensor)
                output = result["relation_embeddings"]
                outputs.append(output)

        # 验证时序输出
        assert len(outputs) >= 2
        # 由于门控残差，连续帧的输出应有差异
        if outputs[0].shape == outputs[1].shape and outputs[0].shape[0] > 0:
            assert not torch.allclose(outputs[0], outputs[1], atol=1e-6)

    def test_anomaly_detection_sensitivity(self, sample_csv, config):
        """测试模型对异常事件的敏感性（embedding norm 变化）。"""
        dataset = TrafficFrameDataset(
            sample_csv, frame_width=1920, frame_height=1080, sequence_length=1
        )
        model = SceneGraphModel(config)
        model.eval()

        sample = dataset[0]
        n = sample["num_nodes"][0].item()
        if n < 2:
            pytest.skip("Not enough nodes")

        nf = sample["node_features"][0, :n]
        tids = sample["track_ids"][0, :n]

        # 正常场景
        positions = sample["raw_positions"][0, :n].numpy()
        velocities = sample["raw_velocities"][0, :n].numpy()
        bboxes = sample["raw_bboxes"][0, :n].numpy()

        builder = GraphBuilder(distance_threshold=500.0)
        edge_index, edge_attr = builder.build_graph(
            positions, velocities, bboxes, tids.numpy()
        )

        with torch.no_grad():
            normal_result = model(nf, edge_index, edge_attr, track_ids=tids)
            normal_output = normal_result["relation_embeddings"]

        # 注入异常（大幅改变速度差）
        perturbed_attr = edge_attr.clone()
        if perturbed_attr.shape[0] > 0:
            perturbed_attr[:, 2:] *= 10.0  # 速度差放大 10 倍

            model.reset_temporal_state()
            with torch.no_grad():
                anomaly_result = model(nf, edge_index, perturbed_attr, track_ids=tids)
                anomaly_output = anomaly_result["relation_embeddings"]

            # 异常输入应导致不同的输出
            if normal_output.shape[0] > 0:
                assert not torch.allclose(normal_output, anomaly_output, atol=1e-4)
