"""
关系分类头: 将 32 维 relation embedding → 8 类语义标签。

MLP 结构: relation_dim(32) → hidden(64) → ReLU → Dropout → num_classes(8)
"""

import torch
import torch.nn as nn
from typing import Optional


class RelationClassifier(nn.Module):
    """关系分类 MLP 头。"""

    def __init__(
        self,
        relation_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (E, relation_dim) 关系嵌入

        Returns:
            (E, num_classes) logits
        """
        return self.classifier(embeddings)
