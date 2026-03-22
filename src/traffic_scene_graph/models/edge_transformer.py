"""
Edge Transformer Encoder：隐式关系自学习的核心组件。

对于相连节点对 (i, j)，将它们的节点特征与相对运动特征 ΔF_ij 拼接，
通过多头自注意力机制输出连续关系向量 (Continuous Relation Embedding)。
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class EdgeTransformerEncoder(nn.Module):
    """边特征的 Transformer 编码器。

    接收边特征输入（节点对特征拼接 + 相对运动特征），
    通过 Transformer Encoder 输出连续关系向量。
    """

    def __init__(
        self,
        edge_input_dim: int = 26,
        edge_hidden_dim: int = 64,
        relation_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            edge_input_dim: 边特征输入维度 = 2*node_dim + relative_motion_dim
            edge_hidden_dim: Transformer 隐层维度
            relation_dim: 输出关系向量维度
            num_heads: 多头注意力头数
            num_layers: Transformer 编码层数
            dropout: Dropout 概率
        """
        super().__init__()

        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.relation_dim = relation_dim

        # 输入投影：将边特征映射到 hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=edge_hidden_dim,
            nhead=num_heads,
            dim_feedforward=edge_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出投影：映射到关系向量维度
        self.output_projection = nn.Sequential(
            nn.Linear(edge_hidden_dim, relation_dim),
            nn.LayerNorm(relation_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            node_features: (N, D_node) 节点特征
            edge_index: (2, E) 边索引 (COO格式)
            edge_attr: (E, D_edge) 边的相对运动特征 [Δx, Δy, Δvx, Δvy]

        Returns:
            relation_embeddings: (E, relation_dim) 连续关系向量
        """
        if edge_index.shape[1] == 0:
            # 无边时返回空张量
            return torch.zeros(
                0, self.relation_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )

        src_idx = edge_index[0]  # (E,)
        dst_idx = edge_index[1]  # (E,)

        # 提取节点对特征
        src_features = node_features[src_idx]  # (E, D_node)
        dst_features = node_features[dst_idx]  # (E, D_node)

        # 拼接边特征: [src_feat, dst_feat, ΔF_ij]
        edge_features = torch.cat(
            [src_features, dst_features, edge_attr], dim=-1
        )  # (E, 2*D_node + D_edge)

        # 输入投影
        hidden = self.input_projection(edge_features)  # (E, hidden_dim)

        # Transformer 编码（将所有边作为序列处理）
        # 增加 batch 维: (1, E, hidden_dim)
        hidden = hidden.unsqueeze(0)
        hidden = self.transformer_encoder(hidden)
        hidden = hidden.squeeze(0)  # (E, hidden_dim)

        # 输出投影
        relation_embeddings = self.output_projection(hidden)  # (E, relation_dim)

        return relation_embeddings

    def compute_edge_input(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """仅计算边输入特征（用于调试/可视化）。

        Returns:
            edge_input: (E, edge_input_dim) 拼接后的边输入特征
        """
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]
        return torch.cat([src_features, dst_features, edge_attr], dim=-1)
