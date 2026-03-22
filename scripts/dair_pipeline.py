import os
import json
import argparse
import random
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# 为了不报错找不到包
try:
    from ultralytics import YOLO
except ImportError:
    pass

import torch
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from traffic_scene_graph.data import TrafficFrameDataset
from traffic_scene_graph.training.trainer import Trainer
from traffic_scene_graph.models import SceneGraphModel

COCO_TO_TSG_CLASS = {
    0: "pedestrian",  # COCO: person
    1: "cyclist",     # COCO: bicycle
    2: "car",         # COCO: car
    3: "cyclist",     # COCO: motorcycle
    5: "bus",         # COCO: bus
    7: "truck",       # COCO: truck
}

class DairPipeline:
    def __init__(self, dair_root: str, output_dir: str):
        self.dair_root = Path(dair_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_csv = self.output_dir / "train_detections.csv"
        self.test_csv = self.output_dir / "test_detections.csv"
        self.config_path = self.output_dir / "pipeline_config.yaml"
        
        # 探测 data_info.json 路径
        self.data_info_path = self._find_data_info()

    def _find_data_info(self):
        """尝试多种结构查找 data_info.json"""
        candidates = [
            self.dair_root / "single-infrastructure-side" / "single-infrastructure-side" / "data_info.json",
            self.dair_root / "single-infrastructure-side" / "data_info.json",
            self.dair_root / "data_info.json"
        ]
        for c in candidates:
            if c.exists():
                return c
        return None


    def _resolve_image_path(self, image_rel_path: str) -> Path:
        """解析 DAIR-V2X 特有的图片路径分离结构"""
        filename = Path(image_rel_path).name
        candidates = [
            # 官方默认的分离式图片包结构
            self.dair_root / "single-infrastructure-side-image" / "single-infrastructure-side-image" / filename,
            self.dair_root / "single-infrastructure-side-image" / "image" / filename,
            self.dair_root / "single-infrastructure-side-image" / filename,
            # 回退到 data_info 所在目录作为基准
            self.data_info_path.parent / image_rel_path,
            self.data_info_path.parent / "image" / filename,
            self.dair_root / image_rel_path
        ]
        for c in candidates:
            if c.exists():
                return c
        # 如果找不到，返回一个预期应该在的位置，用于打印报错
        return self.dair_root / "single-infrastructure-side-image" / "single-infrastructure-side-image" / filename

    def step1_split_dataset(self, split_ratio=0.8):
        """划分 DAIR-V2X 为训练集和测试集（按路口或图片序列划分）"""
        print("=== 步骤 1: 划分 DAIR-V2X 数据集 ===")
        if not self.data_info_path:
            print(f"[ERROR] 在 {self.dair_root} 下找不到 data_info.json！")
            return None, None

        with open(self.data_info_path, "r") as f:
            data_info = json.load(f)


        # 按路口分组，避免同一路口的图片被划分到不同的集合导致数据泄露
        intersections = {}
        for item in data_info:
            loc = item.get("intersection_loc", "unknown")
            if loc not in intersections:
                intersections[loc] = []
            intersections[loc].append(item)

        locs = list(intersections.keys())
        random.seed(42)
        random.shuffle(locs)

        split_idx = int(len(locs) * split_ratio)
        train_locs = set(locs[:split_idx])
        test_locs = set(locs[split_idx:])

        train_items = []
        test_items = []

        for loc, items in intersections.items():
            # 按时间戳排序，对序列提取更有利
            items = sorted(items, key=lambda x: int(x.get("image_timestamp", 0)))
            if loc in train_locs:
                train_items.extend(items)
            else:
                test_items.extend(items)

        print(f"  - 训练集: {len(train_locs)} 个路口, {len(train_items)} 张图像")
        print(f"  - 测试集: {len(test_locs)} 个路口, {len(test_items)} 张图像")

        return train_items, test_items

    def step2_extract_features(self, items, output_csv):
        """运行 YOLO 并在图像序列上跟踪，提取 CSV"""
        print(f"=== 步骤 2: 生成特征 CSV ({output_csv.name}) ===")
        if output_csv.exists():
            try:
                df_check = pd.read_csv(output_csv)
                if not df_check.empty and len(df_check) > 0:
                    print(f"  - 文件已存在且包含 {len(df_check)} 条有效记录，跳过提取: {output_csv}")
                    return
            except Exception:
                pass
            print(f"  - 发现历史无效/空缓存文件 {output_csv}，正在重新处理...")

        print("  - 加载 YOLOv8 模型...")
        yolo_model = YOLO("yolov8n.pt")

        records = []
        frame_idx = 0
        
        # 按照路口分批跟踪，确保 tracker 重置
        intersections = {}
        for item in items:
            loc = item.get("intersection_loc", "unknown")
            if loc not in intersections:
                intersections[loc] = []
            intersections[loc].append(item)

        for loc, group_items in tqdm(intersections.items(), desc="处理路口"):
            # 简单的基于过去的差分测速
            track_history = {}

            # 自动探测图片基础路径
            data_info_dir = self.data_info_path.parent

            # 调试信息: 打印第一个被检查的路径
            if group_items:
                test_path = data_info_dir / group_items[0]["image_path"]
                print(f"[DEBUG] 尝试定位图片. data_info 所在目录: {data_info_dir}")
                print(f"[DEBUG] 预期图片路径示例: {test_path} (exists={test_path.exists()})")
                if not test_path.exists():
                    print(f"[DEBUG] 目录内容汇总: {[p.name for p in list(data_info_dir.glob('*'))[:5]]}")

            for item in group_items:
                image_rel_path = item["image_path"]
                # 动态解析真实的图片绝对路径
                img_path = self._resolve_image_path(image_rel_path)
                    
                if not img_path.exists():
                    if frame_idx % 1000 == 0:
                        print(f"[DEBUG] 找不到图片: {img_path} (原始标注={image_rel_path})")
                    continue


                results = yolo_model.track(str(img_path), persist=True, tracker="bytetrack.yaml", verbose=False)

                if len(results) > 0 and results[0].boxes and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        tid = int(boxes.id[i].item())
                        cls_id = int(boxes.cls[i].item())
                        
                        if cls_id not in COCO_TO_TSG_CLASS:
                            continue
                            
                        cls_name = COCO_TO_TSG_CLASS[cls_id]
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1

                        if tid not in track_history:
                            track_history[tid] = []
                        
                        history = track_history[tid]
                        history.append((frame_idx, cx, cy))
                        if len(history) > 5:
                            history.pop(0)

                        if len(history) >= 2:
                            dt = history[-1][0] - history[0][0]
                            vx = (history[-1][1] - history[0][1]) / dt if dt > 0 else 0.0
                            vy = (history[-1][2] - history[0][2]) / dt if dt > 0 else 0.0
                        else:
                            vx, vy = 0.0, 0.0

                        records.append({
                            "frame_id": frame_idx,
                            "track_id": tid,
                            "class": cls_name,
                            "x": cx, "y": cy, "w": w, "h": h,
                            "vx": vx, "vy": vy
                        })
                frame_idx += 1

        cols = ["frame_id", "track_id", "class", "x", "y", "w", "h", "vx", "vy"]
        df = pd.DataFrame(records, columns=cols)
        df.to_csv(output_csv, index=False)
        print(f"  - 保存了 {len(df)} 条检测记录到 {output_csv}")
        
        if len(df) == 0:
            print(f"[WARNING] 步骤 2 未能检测到任何目标！请检查数据路径或模型是否正常。")


    def step3_train_model(self):
        """训练场景图模型"""
        print("=== 步骤 3: 训练 SceneGraphModel ===")
        # 首先生成一份针对当前管道的配置文件
        base_config_path = Path("configs/default.yaml")
        if not base_config_path.exists():
            print(f"[ERROR] 找不到基础配置 {base_config_path}")
            return
            
        with open(base_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        # 覆盖配置路径
        config["data"]["csv_path"] = str(self.train_csv.absolute())
        config["training"]["epochs"] = 10 # 快速演示
        config["inference"]["checkpoint_path"] = str((self.output_dir / "checkpoints" / "best_model.pt").absolute())
        config["training"]["checkpoint_dir"] = str((self.output_dir / "checkpoints").absolute())
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
            
        print(f"  - 生成管道配置: {self.config_path}")

        if not self.train_csv.exists():
            print("[ERROR] 训练集 CSV 不存在，无法进行训练。")
            return

        dataset = TrafficFrameDataset(
            csv_path=str(self.train_csv),
            frame_width=config["data"].get("frame_width", 1920),
            frame_height=config["data"].get("frame_height", 1080),
            sequence_length=3  # 训练时序
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(config, device=device)
        trainer.train(dataset)

    def step4_inference(self):
        """在测试集上运行推理并输出场景图"""
        print("=== 步骤 4: 推理生成 JSON 场景图 ===")
        
        if not self.config_path.exists() or not self.test_csv.exists():
            print("[ERROR] 缺乏配置或测试集数据！")
            return
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ckpt_path = config["inference"]["checkpoint_path"]
        
        # 覆用已有的推导脚本逻辑或直接调用
        cmd = f"python scripts/inference.py --config {self.config_path} --checkpoint {ckpt_path} --input {self.test_csv} --output-dir {self.output_dir / 'inference_out'}"
        
        print(f"  - 执行推理命令: {cmd}")
        os.system(cmd)
        print(f"  - 推理完成！场景图 JSON 存放在 {self.output_dir / 'inference_out'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAIR-V2X 场景图完整管道跑通脚本")
    parser.add_argument("--dair_root", type=str, required=True, help="DAIR-V2X 数据集根目录路径")
    parser.add_argument("--output_dir", type=str, default="pipeline_output", help="输出文件夹路径")
    parser.add_argument("--skip_tracking", action="store_true", help="如果已有 CSV，则跳过步骤2重跑追踪")
    
    args = parser.parse_args()
    
    pipeline = DairPipeline(args.dair_root, args.output_dir)
    
    # 执行流水线
    train_items, test_items = pipeline.step1_split_dataset(split_ratio=0.8)
    
    if train_items and test_items:
        if not args.skip_tracking:
            pipeline.step2_extract_features(train_items, pipeline.train_csv)
            pipeline.step2_extract_features(test_items, pipeline.test_csv)
        
        pipeline.step3_train_model()
        pipeline.step4_inference()
    else:
        print("[ERROR] 无法分割数据集，管道终止。")
