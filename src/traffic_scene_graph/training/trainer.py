"""
训练器：管理训练循环、TensorBoard 日志、模型保存/加载。
支持对比学习 + 分类损失的联合训练，以及三阶段自训练。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from ..models import SceneGraphModel
from ..utils import GraphBuilder, BEVTransform, ZoneManager, RelationLabeler, SelfTrainingScheduler
from .contrastive_loss import SpatioTemporalContrastiveLoss
from .classification_loss import MultiTaskLoss


class Trainer:
    """交通场景图模型训练器。"""

    def __init__(self, config: Dict, device: Optional[str] = None):
        """
        Args:
            config: 完整配置字典
            device: 计算设备 (None 则自动检测)
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = config["training"]
        self.epochs = train_cfg.get("epochs", 100)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.save_every = train_cfg.get("save_every", 10)
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        self.log_dir = train_cfg.get("log_dir", "logs")

        contrastive_cfg = train_cfg.get("contrastive", {})

        # 初始化模型
        self.model = SceneGraphModel(config).to(self.device)

        # 初始化图构建器
        graph_cfg = config.get("graph", {})
        self.graph_builder = GraphBuilder(
            kalman_predict_steps=graph_cfg.get("kalman_predict_steps", 5),
            iou_threshold=graph_cfg.get("iou_threshold", 0.0),
            distance_threshold=graph_cfg.get("distance_threshold", 150.0),
        )

        # 损失函数
        self.criterion = SpatioTemporalContrastiveLoss(
            temperature=contrastive_cfg.get("temperature", 0.07),
            positive_window=contrastive_cfg.get("positive_window", 3),
            num_negatives=contrastive_cfg.get("num_negatives", 5),
            perturbation_scale=contrastive_cfg.get("perturbation_scale", 3.0),
        )

        # 联合损失 (对比 + 分类)
        classifier_cfg = config.get("classifier", None)
        self.has_classifier = classifier_cfg is not None
        if self.has_classifier:
            self.multi_task_loss = MultiTaskLoss(
                num_classes=classifier_cfg.get("num_classes", 8),
                label_smoothing=train_cfg.get("label_smoothing", 0.1),
            )
            self.classification_alpha = train_cfg.get("classification_alpha", 0.5)

        # BEV 变换
        bev_cfg = config.get("bev", {})
        try:
            self.bev_transform = BEVTransform.from_calibration_file(bev_cfg.get("calibration_file", "configs/bev_calibration.json"))
        except FileNotFoundError:
            print("[WARNING] BEV calibration file not found. Using default homography.")
            self.bev_transform = BEVTransform()

        # 区域管理器
        zone_cfg = config.get("zone", {})
        self.zone_manager = ZoneManager(
            bev_range=zone_cfg.get("bev_range", [0.0, 0.0, 100.0, 100.0]),
            grid_rows=zone_cfg.get("grid_rows", 3),
            grid_cols=zone_cfg.get("grid_cols", 3),
            node_feature_dim=config.get("node_feature", {}).get("input_dim", 12)
        )
        self.zone_features = self.zone_manager.generate_zone_features().to(self.device)

        # 伪标签与自训练
        labeler_cfg = config.get("relation_labeler", {})
        self.relation_labeler = RelationLabeler(
            cos_sim_threshold=labeler_cfg.get("cos_sim_threshold", 0.7),
            crossing_cos_threshold=labeler_cfg.get("crossing_cos_threshold", -0.3),
            closing_rate_threshold=labeler_cfg.get("closing_rate_threshold", 2.0),
            distance_near=labeler_cfg.get("distance_near", 15.0),
            following_lateral_threshold=labeler_cfg.get("following_lateral_threshold", 3.0),
        )

        st_cfg = train_cfg.get("self_training", {})
        self.st_scheduler = SelfTrainingScheduler(
            total_epochs=self.epochs,
            stage1_ratio=st_cfg.get("stage1_ratio", 0.3),
            stage2_ratio=st_cfg.get("stage2_ratio", 0.4),
            confidence_threshold_stage2=st_cfg.get("confidence_threshold", 0.85),
            confidence_threshold_stage3=st_cfg.get("stage3_confidence", 0.7),
        )

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # 学习率调度器
        lr_scheduler_type = train_cfg.get("lr_scheduler", "cosine")
        if lr_scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )

        # TensorBoard writer (可选)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            print("[WARNING] TensorBoard not available, skipping logging.")

        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, dataset):
        """执行完整训练循环。

        Args:
            dataset: TrafficFrameDataset 实例
        """
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # 每次处理一个序列
            shuffle=True,
            num_workers=0, # 必须为 0，防止由于 Pandas DataFrame 跨进程传输导致的卡死
        )

        best_loss = float("inf")

        print(f"[INFO] Training on {self.device} for {self.epochs} epochs")
        print(f"[INFO] Dataset: {len(dataset)} sequences")
        print(f"[INFO] Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.epochs + 1):
            epoch_loss = self._train_epoch(dataloader, epoch)

            # 学习率调度
            self.scheduler.step()

            # TensorBoard 日志
            if self.writer:
                self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                self.writer.add_scalar(
                    "LR", self.optimizer.param_groups[0]["lr"], epoch
                )

            # 保存检查点
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._save_checkpoint(epoch, epoch_loss, is_best=True)

            print(
                f"  Epoch [{epoch}/{self.epochs}] "
                f"Loss: {epoch_loss:.6f} | "
                f"Best: {best_loss:.6f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

        if self.writer:
            self.writer.close()

        print(f"\n[INFO] Training complete. Best loss: {best_loss:.6f}")

    def _train_epoch(self, dataloader, epoch: int) -> float:
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accumulation_steps = 16

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            loss = self._train_step(batch, epoch, accumulation_steps=accumulation_steps)
            if loss is not None:
                total_loss += loss
                num_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")

            # 累加一定步数后再执行计算图裁剪更新
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, torch.Tensor], epoch: int = 1, accumulation_steps: int = 1) -> Optional[float]:
        """单步训练。

        处理一个时间序列，构建图，前向传播，计算对比损失。
        """
        node_features = batch["node_features"].squeeze(0)   # (T, max_N, D)
        num_nodes = batch["num_nodes"].squeeze(0)            # (T,)
        track_ids = batch["track_ids"].squeeze(0)            # (T, max_N)
        raw_positions = batch["raw_positions"].squeeze(0)    # (T, max_N, 2)
        raw_velocities = batch["raw_velocities"].squeeze(0)  # (T, max_N, 2)
        raw_bboxes = batch["raw_bboxes"].squeeze(0)          # (T, max_N, 4)

        T = node_features.shape[0]

        # 重置时序状态
        self.model.reset_temporal_state()
        self.graph_builder.reset()

        sequence_embeddings = []
        all_logits = []
        all_rule_labels = []

        previous_zone_indices = None

        for t in range(T):
            n = num_nodes[t].item()
            if n < 2:
                sequence_embeddings.append(
                    torch.zeros(0, self.model.get_relation_dim(), device=self.device)
                )
                continue

            # 提取当前帧实体数据
            nf_entities = node_features[t, :n].to(self.device)
            tids = track_ids[t, :n].numpy()
            raw_pos = raw_positions[t, :n].numpy()
            raw_vel = raw_velocities[t, :n].numpy()
            raw_bbox = raw_bboxes[t, :n].numpy()

            # BEV 投影
            bev_pos = self.bev_transform.pixel_to_bev(raw_pos)
            bev_vel = self.bev_transform.velocity_to_bev(raw_pos, raw_vel)
            bev_bbox = self.bev_transform.bbox_to_bev_boxes(raw_bbox) # 返回 (N, 4) BEV [x, y, w, h]

            # 1. 建图 (实体-实体) 使用 BEV 坐标
            edge_index_ent, edge_attr_ent = self.graph_builder.build_graph(
                positions=bev_pos,
                velocities=bev_vel,
                bboxes=bev_bbox,
                track_ids=tids,
            )

            # 实体-实体 伪标签
            ent_labels = self.relation_labeler.label_entity_relations(
                bev_positions=bev_pos,
                bev_velocities=bev_vel,
                edge_index=edge_index_ent.numpy(),
            )

            # 2. 区域管理器 (实体-环境)
            current_zone_indices = self.zone_manager.get_zone_index(bev_pos)
            edge_index_zone, edge_attr_zone = self.zone_manager.build_entity_zone_edges(
                entity_bev_pos=bev_pos,
                num_entity_nodes=n,
                zone_node_offset=n,
            )

            # 实体-环境 伪标签
            env_labels = self.relation_labeler.label_environment_relations(
                current_zones=current_zone_indices,
                track_ids=tids,
                previous_zones_dict=previous_zone_indices,
            )
            # zone_edge_labels 需要对应 edge_index_zone (每个实体 2 条边)
            zone_edge_labels = []
            for i in range(n):
                zone_edge_labels.extend([env_labels[i], env_labels[i]])
            
            # 使用 track_id 作为键记录本帧的区域索引，供下一帧对比
            previous_zone_indices = {int(tids[i]): current_zone_indices[i] for i in range(n)}

            # 3. 合并图 (实体节点 + 区域节点)
            nf = torch.cat([nf_entities, self.zone_features], dim=0)
            
            if edge_index_ent.shape[1] > 0 and edge_index_zone.shape[1] > 0:
                edge_index = torch.cat([edge_index_ent, edge_index_zone], dim=1)
                edge_attr = torch.cat([edge_attr_ent, edge_attr_zone], dim=0)
                rule_labels = np.concatenate([ent_labels, zone_edge_labels])
            elif edge_index_ent.shape[1] > 0:
                edge_index = edge_index_ent
                edge_attr = edge_attr_ent
                rule_labels = ent_labels
            else:
                edge_index = edge_index_zone
                edge_attr = edge_attr_zone
                rule_labels = np.array(zone_edge_labels)

            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            rule_labels_tensor = torch.tensor(rule_labels, dtype=torch.long, device=self.device)

            # 保持 tids 长度等于 nf 长度，区域无追踪ID填 -1
            tids_pad = np.concatenate([tids, np.full(self.zone_manager.num_zones, -1)])
            tids_tensor = torch.from_numpy(tids_pad).long().to(self.device)

            # 前向传播
            result = self.model(
                nf, edge_index, edge_attr, track_ids=tids_tensor
            )

            relation_emb = result["relation_embeddings"]
            sequence_embeddings.append(relation_emb)

            if "relation_logits" in result:
                all_logits.append(result["relation_logits"])
                all_rule_labels.append(rule_labels_tensor)

        # 构建对比学习样本对
        try:
            anchors, positives, negatives = self.criterion.compute_contrastive_pairs(
                sequence_embeddings
            )
        except ValueError:
            return None

        if anchors.shape[0] == 0:
            return None

        # 计算对比损失
        self.optimizer.zero_grad()
        contrastive_loss = self.criterion(anchors, positives, negatives)

        # 联合损失
        if self.has_classifier and len(all_logits) > 0:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_rule_labels = torch.cat(all_rule_labels, dim=0)

            # 自训练精炼标签
            refined_labels = self.st_scheduler.refine_labels(
                rule_labels=cat_rule_labels,
                model_logits=cat_logits.detach(), 
                epoch=epoch
            )

            alpha = self.st_scheduler.get_classification_weight(epoch)

            loss_dict = self.multi_task_loss(
                contrastive_loss=contrastive_loss,
                relation_logits=cat_logits,
                pseudo_labels=refined_labels,
                alpha=alpha,
            )
            loss = loss_dict["total"]
        else:
            loss = contrastive_loss

        # 为梯度累加进行缩放
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        # 返回缩放前的 loss 供打印日志使用
        return loss.item()

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存模型检查点。"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载模型检查点。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"[INFO] Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]
