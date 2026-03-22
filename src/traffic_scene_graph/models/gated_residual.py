"""
演化门控残差 (Evolutionary Gated Residuals) 模块。

核心公式:
    E_ij^(t) = E_ij^(t-1) + σ(W_gate · ΔF_ij^(t)) ⊙ Transformer_Encoder(H_i^(t), H_j^(t), ΔF_ij^(t))

原理:
    - 平稳行驶时: 门控值 → 0, 继承上一帧的平稳关系
    - 急刹/变道时: ΔF 剧变 → 门控激活 → Transformer 输出的异常信号注入残差流
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .edge_transformer import EdgeTransformerEncoder


class EvolutionaryGatedResidual(nn.Module):
    """带演化门控的时序残差边关系模块。

    维护时序 hidden state 缓存，实现关系向量的渐进式更新。
    """

    def __init__(
        self,
        relation_dim: int = 32,
        relative_motion_dim: int = 4,
        gate_hidden_dim: int = 16,
        edge_transformer: Optional[EdgeTransformerEncoder] = None,
        **transformer_kwargs,
    ):
        """
        Args:
            relation_dim: 关系向量维度（与 EdgeTransformerEncoder 输出维度一致）
            relative_motion_dim: 相对运动特征维度 [Δx, Δy, Δvx, Δvy]
            gate_hidden_dim: 门控 MLP 隐层维度
            edge_transformer: 外部传入的 EdgeTransformerEncoder (共享或独立)
            **transformer_kwargs: 如果未传入 edge_transformer，用于创建新实例
        """
        super().__init__()

        self.relation_dim = relation_dim
        self.relative_motion_dim = relative_motion_dim

        # 门控网络: σ(W_gate · ΔF_ij)
        # 输入: 相对运动特征 ΔF_ij (4维)
        # 输出: relation_dim 维的门控信号
        self.gate_network = nn.Sequential(
            nn.Linear(relative_motion_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, relation_dim),
            nn.Sigmoid(),  # σ 门控，输出 [0, 1]
        )

        # Edge Transformer Encoder
        if edge_transformer is not None:
            self.edge_transformer = edge_transformer
        else:
            self.edge_transformer = EdgeTransformerEncoder(
                relation_dim=relation_dim, **transformer_kwargs
            )

        # 时序隐状态缓存: Dict[Tuple[int, int], Tensor]
        # key = (src_track_id, dst_track_id), value = relation embedding
        self._hidden_states: Dict[Tuple[int, int], torch.Tensor] = {}

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        track_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播，计算当前帧的关系向量。

        Args:
            node_features: (N, D_node) 节点特征
            edge_index: (2, E) 边索引
            edge_attr: (E, 4) 边特征 [Δx, Δy, Δvx, Δvy]
            track_ids: (N,) 节点的跟踪 ID (用于时序状态映射)

        Returns:
            relation_embeddings: (E, relation_dim) 更新后的关系向量
        """
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            return torch.zeros(
                0, self.relation_dim,
                device=node_features.device,
                dtype=node_features.dtype,
            )

        # 1. 通过 Edge Transformer 计算当前帧的"原始"关系信号
        transformer_output = self.edge_transformer(
            node_features, edge_index, edge_attr
        )  # (E, relation_dim)

        # 2. 计算门控信号: σ(W_gate · ΔF_ij)
        gate = self.gate_network(edge_attr)  # (E, relation_dim)

        # 3. 门控后的 Transformer 输出
        gated_signal = gate * transformer_output  # (E, relation_dim) ⊙ 逐元素乘

        # 4. 残差更新: E^(t) = E^(t-1) + gated_signal
        if track_ids is not None:
            # 使用 track_id 映射 hidden state
            relation_embeddings = self._residual_update_with_tracking(
                gated_signal, edge_index, track_ids
            )
        else:
            # 无 track_id 时直接输出（不使用残差）
            relation_embeddings = gated_signal

        return relation_embeddings

    def _residual_update_with_tracking(
        self,
        gated_signal: torch.Tensor,
        edge_index: torch.Tensor,
        track_ids: torch.Tensor,
    ) -> torch.Tensor:
        """使用目标跟踪 ID 进行时序残差更新。

        Args:
            gated_signal: (E, relation_dim) 门控后的 Transformer 输出
            edge_index: (2, E) 边索引
            track_ids: (N,) 跟踪 ID

        Returns:
            updated_embeddings: (E, relation_dim) 残差更新后的关系向量
        """
        device = gated_signal.device
        num_edges = gated_signal.shape[0]
        updated = torch.zeros_like(gated_signal)

        for e in range(num_edges):
            src_node = edge_index[0, e].item()
            dst_node = edge_index[1, e].item()

            src_track = track_ids[src_node].item()
            dst_track = track_ids[dst_node].item()

            edge_key = (src_track, dst_track)

            # 获取上一帧的 hidden state (如果存在)
            if edge_key in self._hidden_states:
                prev_state = self._hidden_states[edge_key].to(device)
                # 残差更新: E^(t) = E^(t-1) + gated_signal
                updated[e] = prev_state + gated_signal[e]
            else:
                # 首次出现，直接使用当前信号
                updated[e] = gated_signal[e]

            # 缓存当前状态
            self._hidden_states[edge_key] = updated[e].detach().clone()

        return updated

    def reset_hidden_states(self):
        """清空时序隐状态缓存（新序列开始时调用）。"""
        self._hidden_states.clear()

    def get_active_edge_count(self) -> int:
        """返回当前缓存的边关系数量。"""
        return len(self._hidden_states)
