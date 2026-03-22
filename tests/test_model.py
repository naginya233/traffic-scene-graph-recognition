"""
测试模型模块：验证前向传播、输出维度、门控残差机制。
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic_scene_graph.models import (
    EdgeTransformerEncoder,
    EvolutionaryGatedResidual,
    SceneGraphModel,
)


class TestEdgeTransformerEncoder:
    """Edge Transformer 编码器测试。"""

    def test_forward_shape(self):
        """测试前向传播输出形状。"""
        model = EdgeTransformerEncoder(
            edge_input_dim=28, edge_hidden_dim=64,
            relation_dim=32, num_heads=4, num_layers=2,
        )

        N, D_node = 5, 12
        E = 8
        node_features = torch.randn(N, D_node)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, 4)

        output = model(node_features, edge_index, edge_attr)
        assert output.shape == (E, 32)

    def test_empty_edges(self):
        """测试无边情况。"""
        model = EdgeTransformerEncoder(edge_input_dim=28, relation_dim=32)

        node_features = torch.randn(5, 12)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 4)

        output = model(node_features, edge_index, edge_attr)
        assert output.shape == (0, 32)

    def test_gradient_flow(self):
        """测试梯度能正确反传。"""
        model = EdgeTransformerEncoder(edge_input_dim=28, relation_dim=32)

        node_features = torch.randn(5, 12, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.randn(3, 4)

        output = model(node_features, edge_index, edge_attr)
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None


class TestEvolutionaryGatedResidual:
    """演化门控残差测试。"""

    def test_forward_shape(self):
        """测试输出形状正确。"""
        edge_transformer = EdgeTransformerEncoder(
            edge_input_dim=28, relation_dim=32
        )
        model = EvolutionaryGatedResidual(
            relation_dim=32, relative_motion_dim=4,
            edge_transformer=edge_transformer,
        )

        node_features = torch.randn(5, 12)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.randn(3, 4)
        track_ids = torch.tensor([10, 20, 30, 40, 50])

        output = model(node_features, edge_index, edge_attr, track_ids=track_ids)
        assert output.shape == (3, 32)

    def test_temporal_residual(self):
        """测试时序残差累积效应。"""
        edge_transformer = EdgeTransformerEncoder(
            edge_input_dim=28, relation_dim=32
        )
        model = EvolutionaryGatedResidual(
            relation_dim=32, relative_motion_dim=4,
            edge_transformer=edge_transformer,
        )

        node_features = torch.randn(3, 12)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.randn(2, 4)
        track_ids = torch.tensor([1, 2, 3])

        # 第一帧
        model.reset_hidden_states()
        out1 = model(node_features, edge_index, edge_attr, track_ids=track_ids)

        # 第二帧（用相同数据 — 应该有残差累积）
        out2 = model(node_features, edge_index, edge_attr, track_ids=track_ids)

        # 两次输出应该不同（因为残差累积）
        assert not torch.allclose(out1, out2, atol=1e-6)

    def test_reset_hidden_states(self):
        """测试重置隐状态。"""
        edge_transformer = EdgeTransformerEncoder(
            edge_input_dim=28, relation_dim=32
        )
        model = EvolutionaryGatedResidual(
            relation_dim=32, relative_motion_dim=4,
            edge_transformer=edge_transformer,
        )

        node_features = torch.randn(3, 12)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.randn(2, 4)
        track_ids = torch.tensor([1, 2, 3])

        model(node_features, edge_index, edge_attr, track_ids=track_ids)
        assert model.get_active_edge_count() > 0

        model.reset_hidden_states()
        assert model.get_active_edge_count() == 0


class TestSceneGraphModel:
    """完整场景图模型测试。"""

    @pytest.fixture
    def config(self):
        return {
            "node_feature": {"input_dim": 12},
            "model": {
                "edge_input_dim": 28,
                "edge_hidden_dim": 64,
                "relation_dim": 32,
                "num_heads": 4,
                "num_transformer_layers": 2,
                "dropout": 0.1,
                "gate_hidden_dim": 16,
                "relative_motion_dim": 4,
            },
        }

    def test_forward_shape(self, config):
        """测试完整模型输出形状。"""
        model = SceneGraphModel(config)

        node_features = torch.randn(5, 12)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        edge_attr = torch.randn(4, 4)

        output = model(node_features, edge_index, edge_attr)
        assert output["relation_embeddings"].shape == (4, 32)

    def test_extract_scene_features(self, config):
        """测试推理模式提取特征。"""
        model = SceneGraphModel(config)

        node_features = torch.randn(5, 12)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_attr = torch.randn(2, 4)

        result = model.extract_scene_features(
            node_features, edge_index, edge_attr
        )

        assert "relation_embeddings" in result
        assert "edge_index" in result
        assert "node_features" in result
        assert result["relation_embeddings"].shape == (2, 32)

    def test_param_count(self, config):
        """测试模型参数数量合理（轻量级）。"""
        model = SceneGraphModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        # 应该是轻量级模型 (< 500K 参数)
        assert total_params < 500_000, f"Model too large: {total_params:,} params"

    def test_reset_temporal_state(self, config):
        """测试重置时序状态。"""
        model = SceneGraphModel(config)
        model.reset_temporal_state()  # 应无报错
