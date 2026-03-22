"""
联合损失: 对比学习损失 + 分类交叉熵损失。

total_loss = contrastive_loss + alpha * classification_loss

alpha 由 SelfTrainingScheduler 根据训练阶段动态调整。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """用于应对严重类别不平衡的 Focal Loss。"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """多任务联合损失函数。"""

    def __init__(self, num_classes: int = 8, label_smoothing: float = 0.1):
        super().__init__()
        # 替换标准的 CE 为 Focal Loss 来解决绝大多数边都是 independent(0) 或 in_zone(5) 导致的 Loss 塌陷
        self.ce_loss = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")
        self.num_classes = num_classes

    def forward(
        self,
        contrastive_loss: torch.Tensor,
        relation_logits: Optional[torch.Tensor] = None,
        pseudo_labels: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """计算联合损失。

        Args:
            contrastive_loss: 已计算的对比学习损失 (标量)
            relation_logits: (E, C) 分类 logits
            pseudo_labels: (E,) 伪标签
            alpha: 分类损失权重

        Returns:
            Dict 包含: total, contrastive, classification
        """
        result = {
            "contrastive": contrastive_loss,
            "classification": torch.tensor(0.0, device=contrastive_loss.device),
            "total": contrastive_loss,
        }

        if (
            relation_logits is not None
            and pseudo_labels is not None
            and pseudo_labels.numel() > 0
        ):
            cls_loss = self.ce_loss(relation_logits, pseudo_labels)
            result["classification"] = cls_loss
            result["total"] = contrastive_loss + alpha * cls_loss

        return result
