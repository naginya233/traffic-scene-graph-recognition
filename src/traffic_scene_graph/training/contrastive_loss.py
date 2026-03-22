"""
时空对比学习损失函数 (Spatio-Temporal Contrastive Loss)

自监督训练策略：
  - 正样本：时间相邻 + 运动平稳的图片段（关系向量应相近）
  - 负样本：人为注入扰动（篡改速度矢量模拟碰撞）的图片段

损失函数：InfoNCE (Noise Contrastive Estimation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class SpatioTemporalContrastiveLoss(nn.Module):
    """时空对比学习损失函数。

    通过 InfoNCE 损失，迫使模型在隐式特征空间中：
      - 将"正常行驶"聚类为紧凑流形
      - 将"异常交互"作为离群点独立出来
    """

    def __init__(
        self,
        temperature: float = 0.07,
        positive_window: int = 3,
        num_negatives: int = 5,
        perturbation_scale: float = 3.0,
    ):
        """
        Args:
            temperature: InfoNCE 温度系数 (越小越尖锐)
            positive_window: 正样本时间窗口 (±N帧)
            num_negatives: 每个 anchor 的负样本数量
            perturbation_scale: 速度扰动缩放倍数
        """
        super().__init__()
        self.temperature = temperature
        self.positive_window = positive_window
        self.num_negatives = num_negatives
        self.perturbation_scale = perturbation_scale

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """计算 InfoNCE 损失。

        Args:
            anchor_embeddings: (B, D) 锚点关系向量（当前帧）
            positive_embeddings: (B, D) 正样本关系向量（相邻帧）
            negative_embeddings: (B, K, D) 负样本关系向量（扰动帧）

        Returns:
            loss: 标量损失值
        """
        B, D = anchor_embeddings.shape
        K = negative_embeddings.shape[1]

        # L2 归一化
        anchor = F.normalize(anchor_embeddings, dim=-1)      # (B, D)
        positive = F.normalize(positive_embeddings, dim=-1)   # (B, D)
        negative = F.normalize(negative_embeddings, dim=-1)   # (B, K, D)

        # 正样本相似度: (B,)
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature

        # 负样本相似度: (B, K)
        neg_sim = torch.bmm(
            negative, anchor.unsqueeze(-1)
        ).squeeze(-1) / self.temperature  # (B, K)

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, 1+K)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    @staticmethod
    def generate_perturbation(
        edge_attr: torch.Tensor,
        scale: float = 3.0,
    ) -> torch.Tensor:
        """对边特征注入扰动，生成负样本。

        模拟异常交互：篡改速度差异，生成"碰撞"/"急刹"等虚假场景。

        Args:
            edge_attr: (E, 4) 原始边特征 [Δx, Δy, Δvx, Δvy]
            scale: 扰动缩放倍数

        Returns:
            perturbed: (E, 4) 扰动后的边特征
        """
        noise = torch.randn_like(edge_attr) * scale
        # 主要扰动速度差异 (后两个维度)
        noise[:, :2] *= 0.3  # 位置扰动较小
        noise[:, 2:] *= 1.0  # 速度扰动较大

        perturbed = edge_attr + noise
        return perturbed

    def compute_contrastive_pairs(
        self,
        sequence_embeddings: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从时间序列的关系向量中构建对比学习的正负样本对。

        Args:
            sequence_embeddings: [T 个 (E_t, D) 张量]，每帧的关系向量

        Returns:
            anchors: (B, D)
            positives: (B, D)
            negatives: (B, K, D)
        """
        T = len(sequence_embeddings)
        if T < 2:
            raise ValueError("至少需要 2 帧来构建对比样本对")

        anchors = []
        positives = []
        negatives = []

        for t in range(T):
            emb_t = sequence_embeddings[t]  # (E_t, D)
            if emb_t.shape[0] == 0:
                continue

            # 选正样本: 时间窗口内的相邻帧
            pos_indices = [
                i for i in range(max(0, t - self.positive_window),
                                  min(T, t + self.positive_window + 1))
                if i != t and sequence_embeddings[i].shape[0] > 0
            ]

            if not pos_indices:
                continue

            # 取最近的正样本帧
            pos_t = min(pos_indices, key=lambda i: abs(i - t))
            emb_pos = sequence_embeddings[pos_t]

            # 对齐边数量（取最小值）
            min_edges = min(emb_t.shape[0], emb_pos.shape[0])
            anchor_batch = emb_t[:min_edges]
            positive_batch = emb_pos[:min_edges]

            # 生成负样本: 随机打乱 + 高斯噪声
            D = emb_t.shape[1]
            neg_batch = []
            for _ in range(self.num_negatives):
                # 随机打乱 + 添加噪声
                perm = torch.randperm(min_edges)
                neg = anchor_batch[perm] + torch.randn(min_edges, D, device=emb_t.device) * 0.5
                neg_batch.append(neg)

            neg_batch = torch.stack(neg_batch, dim=1)  # (min_edges, K, D)

            anchors.append(anchor_batch)
            positives.append(positive_batch)
            negatives.append(neg_batch)

        if not anchors:
            D = sequence_embeddings[0].shape[-1] if sequence_embeddings else 32
            device = sequence_embeddings[0].device if sequence_embeddings else "cpu"
            return (
                torch.zeros(0, D, device=device),
                torch.zeros(0, D, device=device),
                torch.zeros(0, self.num_negatives, D, device=device),
            )

        return (
            torch.cat(anchors, dim=0),
            torch.cat(positives, dim=0),
            torch.cat(negatives, dim=0),
        )
