"""
SceneGraphModel：完整的场景图模型，组合 EdgeTransformerEncoder + EvolutionaryGatedResidual。

输入: PyG-style Data (node_features, edge_index, edge_attr)
输出: Dict 包含 relation_embeddings (连续向量) + relation_logits (分类)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

from .edge_transformer import EdgeTransformerEncoder
from .gated_residual import EvolutionaryGatedResidual
from .relation_classifier import RelationClassifier


class SceneGraphModel(nn.Module):
    """交通场景图模型：端到端地从节点特征生成边关系向量。

    架构:
        1. 节点特征编码 (可选的 MLP 升维)
        2. EdgeTransformerEncoder: 边特征 → Transformer → 关系向量
        3. EvolutionaryGatedResidual: 时序门控残差更新
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典，包含 model 和 node_feature 相关配置
        """
        super().__init__()

        model_cfg = config["model"]
        node_cfg = config.get("node_feature", {})
        classifier_cfg = config.get("classifier", None)

        self.node_input_dim = node_cfg.get("input_dim", 11)
        self.edge_input_dim = model_cfg.get("edge_input_dim", 26)
        self.edge_hidden_dim = model_cfg.get("edge_hidden_dim", 64)
        self.relation_dim = model_cfg.get("relation_dim", 32)
        self.num_heads = model_cfg.get("num_heads", 4)
        self.num_transformer_layers = model_cfg.get("num_transformer_layers", 2)
        self.dropout = model_cfg.get("dropout", 0.1)
        self.relative_motion_dim = model_cfg.get("relative_motion_dim", 4)
        self.gate_hidden_dim = model_cfg.get("gate_hidden_dim", 16)

        # 1. 节点特征编码器 (可选的升维 MLP)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_input_dim, self.node_input_dim),
            nn.LayerNorm(self.node_input_dim),
            nn.GELU(),
        )

        # 2. Edge Transformer Encoder（共享实例）
        self.edge_transformer = EdgeTransformerEncoder(
            edge_input_dim=self.edge_input_dim,
            edge_hidden_dim=self.edge_hidden_dim,
            relation_dim=self.relation_dim,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            dropout=self.dropout,
        )

        # 3. 演化门控残差
        self.gated_residual = EvolutionaryGatedResidual(
            relation_dim=self.relation_dim,
            relative_motion_dim=self.relative_motion_dim,
            gate_hidden_dim=self.gate_hidden_dim,
            edge_transformer=self.edge_transformer,  # 共享 Transformer
        )

        # 4. 关系分类头 (可选)
        self.has_classifier = classifier_cfg is not None
        if self.has_classifier:
            self.relation_classifier = RelationClassifier(
                relation_dim=self.relation_dim,
                hidden_dim=classifier_cfg.get("hidden_dim", 64),
                num_classes=classifier_cfg.get("num_classes", 8),
                dropout=self.dropout,
            )
            self.class_names: List[str] = classifier_cfg.get("class_names", [])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        track_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            node_features: (N, node_input_dim) 节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, relative_motion_dim) 边特征 [Δx, Δy, Δvx, Δvy]
            track_ids: (N,) 跟踪 ID（用于时序残差）

        Returns:
            Dict 包含:
              - relation_embeddings: (E, relation_dim) 连续关系向量
              - relation_logits: (E, num_classes) 分类 logits (若有分类头)
        """
        # 节点特征编码
        encoded_nodes = self.node_encoder(node_features)  # (N, node_input_dim)

        # 通过门控残差模块（内部调用 EdgeTransformer）
        relation_embeddings = self.gated_residual(
            encoded_nodes, edge_index, edge_attr, track_ids=track_ids
        )

        result = {"relation_embeddings": relation_embeddings}

        if self.has_classifier:
            result["relation_logits"] = self.relation_classifier(relation_embeddings)

        return result

    def reset_temporal_state(self):
        """重置时序状态（新视频序列开始时调用）。"""
        self.gated_residual.reset_hidden_states()

    def get_relation_dim(self) -> int:
        """返回关系向量维度。"""
        return self.relation_dim

    @torch.no_grad()
    def extract_scene_features(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        track_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """提取场景图特征（推理模式）。

        Returns:
            dict 包含:
                - relation_embeddings: (E, relation_dim) 关系向量
                - edge_index: (2, E) 边索引
                - node_features: (N, D) 编码后的节点特征
                - relation_labels: (E,) 预测标签 (若有分类头)
                - relation_confidences: (E,) 预测置信度 (若有分类头)
        """
        self.eval()
        result = self.forward(
            node_features, edge_index, edge_attr, track_ids=track_ids
        )
        result["edge_index"] = edge_index
        result["node_features"] = self.node_encoder(node_features)

        if self.has_classifier and "relation_logits" in result:
            probs = torch.softmax(result["relation_logits"], dim=-1)
            confidences, labels = probs.max(dim=-1)
            result["relation_labels"] = labels
            result["relation_confidences"] = confidences

        return result
