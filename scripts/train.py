"""
训练入口脚本：加载配置 → 构建数据集 → 训练模型
"""

import os
import sys
import argparse
import yaml

# 添加 src 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic_scene_graph.data import TrafficFrameDataset
from traffic_scene_graph.training import Trainer


def main():
    parser = argparse.ArgumentParser(description="训练交通场景图模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备 (cuda / cpu)",
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("  边缘端交通语义场景图 - 训练")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Data:   {config['data']['csv_path']}")
    print("=" * 60)

    # 检查数据文件
    csv_path = config["data"]["csv_path"]
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] 数据文件不存在: {csv_path}")
        print("请先运行: python scripts/generate_sample_data.py")
        sys.exit(1)

    # 构建数据集
    dataset = TrafficFrameDataset(
        csv_path=csv_path,
        frame_width=config["data"].get("frame_width", 1920),
        frame_height=config["data"].get("frame_height", 1080),
        sequence_length=config["training"]["contrastive"].get("positive_window", 3) * 2 + 1,
    )

    print(f"\n[INFO] Dataset loaded: {len(dataset)} sequences, "
          f"{dataset.get_frame_count()} total frames")
    print(f"[INFO] Feature dim: {dataset.get_feature_dim()}")

    # 初始化训练器
    trainer = Trainer(config, device=args.device)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train(dataset)


if __name__ == "__main__":
    main()
