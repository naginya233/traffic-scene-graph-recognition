"""
video_to_scene_graph.py：视频流前端集成。
输入视频帧，经过 YOLOv8 端到端提取目标，并输入 SceneGraphModel 推理关系，
最后将场景图（风险/交互边）渲染回原视频上并输出。
"""

import os
import sys
import argparse
import cv2
import yaml
import torch
import numpy as np
from collections import deque, defaultdict
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Please install ultralytics: pip install ultralytics opencv-python")
    sys.exit(1)

# 加入源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from traffic_scene_graph.models import SceneGraphModel
from traffic_scene_graph.utils import GraphBuilder, BEVTransform, ZoneManager

# YOLO COCO 类别映射到我们系统的 5 类别
COCO_TO_TSG_CLASS = {
    0: "pedestrian",  # COCO: person
    1: "cyclist",     # COCO: bicycle
    2: "car",         # COCO: car
    3: "cyclist",     # COCO: motorcycle
    5: "bus",         # COCO: bus
    7: "truck",       # COCO: truck
}


class VideoSceneGraphPipeline:
    def __init__(self, config_path: str, output_path: str,
                 video_path: str = None, image_dir: str = None,
                 checkpoint_path: str = None, device: str = None):
        """
        初始化视频/图像序列推理管道。
        """
        self.video_path = video_path
        self.image_dir = image_dir
        self.output_path = output_path
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # 初始化 YOLO (默认 yolov8n，自动下载)
        print("[INFO] Loading YOLOv8 model...")
        self.yolo = YOLO("yolov8n.pt")
        
        # 初始化 SceneGraphModel
        print("[INFO] Loading Scene Graph Model...")
        self.model = SceneGraphModel(self.config).to(self.device)
        self.model.eval()
        
        ckpt = checkpoint_path or self.config["inference"].get("checkpoint_path", None)
        if ckpt and os.path.exists(ckpt):
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Loaded SceneGraph checkpoint from {ckpt}")
        else:
            print(f"[WARNING] No checkpoint found at {ckpt}, using random init!")

        # 建图器
        graph_cfg = self.config.get("graph", {})
        self.graph_builder = GraphBuilder(
            kalman_predict_steps=graph_cfg.get("kalman_predict_steps", 5),
            iou_threshold=graph_cfg.get("iou_threshold", 0.0),
            distance_threshold=graph_cfg.get("distance_threshold", 15.0), # BEV距离
        )

        # BEV 变换
        bev_cfg = self.config.get("bev", {})
        try:
            self.bev_transform = BEVTransform.from_calibration_file(bev_cfg.get("calibration_file", "configs/bev_calibration.json"))
            print("[INFO] BEV Transform loaded.")
        except FileNotFoundError:
            print("[WARNING] BEV calibration file not found. Using default homography.")
            self.bev_transform = BEVTransform()

        # 区域管理器
        zone_cfg = self.config.get("zone", {})
        self.zone_manager = ZoneManager(
            bev_range=zone_cfg.get("bev_range", [0.0, 0.0, 100.0, 100.0]),
            grid_rows=zone_cfg.get("grid_rows", 3),
            grid_cols=zone_cfg.get("grid_cols", 3),
            node_feature_dim=self.config.get("node_feature", {}).get("input_dim", 12)
        )
        self.zone_features = self.zone_manager.generate_zone_features().to(self.device)
        self.class_names_rel = self.config.get("classifier", {}).get("class_names", [])

        # 状态追踪：用于计算 vx, vy
        # track_id -> [(frame_idx, cx, cy), ...]
        self.track_history = defaultdict(list)
        
        # 参数
        self.class_names = self.config["data"]["class_names"]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        self.frame_width = self.config["data"].get("frame_width", 1920)
        self.frame_height = self.config["data"].get("frame_height", 1080)

    def extract_features(self, tracks, frame_idx):
        """
        将 YOLO tracks 转换为模型所需的节点特征矩阵。
        tracks 列表包含: {"track_id": id, "class": cls, "x": cx, "y": cy, "w": w, "h": h}
        """
        n = len(tracks)
        if n == 0:
            return None, None, None, None, None

        # 计算速度矢量
        for t in tracks:
            tid = t["track_id"]
            cx, cy = t["x"], t["y"]
            
            # 加入历史
            self.track_history[tid].append((frame_idx, cx, cy))
            # 保持历史不超过 5 帧
            if len(self.track_history[tid]) > 5:
                self.track_history[tid].pop(0)
                
            history = self.track_history[tid]
            if len(history) >= 2:
                # 简单差分测速 (当前帧与最早一帧的位置差)
                dt = history[-1][0] - history[0][0]
                dx = history[-1][1] - history[0][1]
                dy = history[-1][2] - history[0][2]
                t["vx"] = dx / dt if dt > 0 else 0.0
                t["vy"] = dy / dt if dt > 0 else 0.0
            else:
                t["vx"] = 0.0
                t["vy"] = 0.0

        # 开始构建张量 (N, 12: 5 classes + 4 bbox norms + 2 vel norms + 1 is_zone)
        node_features = np.zeros((n, self.num_classes + 2 + 2 + 2 + 1), dtype=np.float32)
        positions = np.zeros((n, 2), dtype=np.float32)
        velocities = np.zeros((n, 2), dtype=np.float32)
        bboxes = np.zeros((n, 4), dtype=np.float32)
        track_ids = np.zeros(n, dtype=np.int64)

        for i, t in enumerate(tracks):
            # One hot
            cid = self.class_to_idx.get(t["class"], -1)
            if cid >= 0:
                node_features[i, cid] = 1.0
                
            # Coords norm
            node_features[i, self.num_classes] = t["x"] / self.frame_width
            node_features[i, self.num_classes + 1] = t["y"] / self.frame_height
            
            # W/H norm
            node_features[i, self.num_classes + 2] = t["w"] / self.frame_width
            node_features[i, self.num_classes + 3] = t["h"] / self.frame_height
            
            # Velocity norm
            node_features[i, self.num_classes + 4] = t["vx"] / self.frame_width
            node_features[i, self.num_classes + 5] = t["vy"] / self.frame_height

            positions[i] = [t["x"], t["y"]]
            velocities[i] = [t["vx"], t["vy"]]
            bboxes[i] = [t["x"], t["y"], t["w"], t["h"]]
            track_ids[i] = t["track_id"]

        return (
            torch.from_numpy(node_features),
            torch.from_numpy(track_ids),
            positions,
            velocities,
            bboxes
        )

    def get_frames(self):
        """生成器：按序加载视频流帧或图像文件夹中的图像"""
        if self.image_dir and os.path.isdir(self.image_dir):
            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(exts)])
            print(f"[INFO] Found {len(files)} image files in {self.image_dir}")
            for file in files:
                img_path = os.path.join(self.image_dir, file)
                frame = cv2.imread(img_path)
                if frame is not None:
                    yield frame
                else:
                    print(f"[WARNING] Could not read image: {img_path}")
        elif self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {self.video_path}")
                return
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()
        else:
            print("[ERROR] No input specified! Please provide --video or --image_dir.")

    def process_video(self):
        """处理整段视频或图像流"""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        self.model.reset_temporal_state()
        self.graph_builder.reset()

        frame_idx = 0
        print(f"[INFO] Started processing media stream...")
        
        for frame in self.get_frames():
            # 获取画面宽高，初始化录制参数
            if out is None:
                real_height, real_width = frame.shape[:2]
                self.frame_width = real_width
                self.frame_height = real_height
                
                fps = 30
                if self.video_path:
                    cap_tmp = cv2.VideoCapture(self.video_path)
                    if cap_tmp.isOpened():
                        fps = int(cap_tmp.get(cv2.CAP_PROP_FPS))
                        cap_tmp.release()
                
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (real_width, real_height))
                
            # 1. 运行 YOLO 跟踪 (ByteTrack)
            results = self.yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            tracks = []
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
                    
                    tracks.append({
                        "track_id": tid,
                        "class": cls_name,
                        "x": cx, "y": cy, "w": w, "h": h
                    })
                    
            if len(tracks) < 2:
                # 节点不足，无法建图
                out.write(frame)
                frame_idx += 1
                continue

            # 2. 转换特征
            nf, tids, pos, vel, bbox = self.extract_features(tracks, frame_idx)
            n = len(tracks)

            # BEV 投影
            bev_pos = self.bev_transform.pixel_to_bev(pos)
            bev_vel = self.bev_transform.velocity_to_bev(pos, vel)
            bev_bbox = self.bev_transform.bbox_to_bev_footprint(bbox)
            
            # 3. 建图
            edge_index_ent, edge_attr_ent = self.graph_builder.build_graph(bev_pos, bev_vel, bev_bbox, tids.numpy())
            
            edge_index_zone, edge_attr_zone = self.zone_manager.build_entity_zone_edges(
                entity_bev_pos=bev_pos, num_entity_nodes=n, zone_node_offset=n
            )

            nf_full = torch.cat([nf, self.zone_features.cpu()], dim=0).to(self.device)

            if edge_index_ent.shape[1] > 0 and edge_index_zone.shape[1] > 0:
                edge_index = torch.cat([edge_index_ent, edge_index_zone], dim=1)
                edge_attr = torch.cat([edge_attr_ent, edge_attr_zone], dim=0)
            elif edge_index_ent.shape[1] > 0:
                edge_index = edge_index_ent
                edge_attr = edge_attr_ent
            else:
                edge_index = edge_index_zone
                edge_attr = edge_attr_zone

            if edge_index.shape[1] > 0:
                edge_index_dev = edge_index.to(self.device)
                edge_attr_dev = edge_attr.to(self.device)
                tids_pad = np.concatenate([tids.numpy(), np.full(self.zone_manager.num_zones, -1)])
                tids_tensor = torch.from_numpy(tids_pad).long().to(self.device)
                
                # 4. GNN+Transformer 隐式关系提取
                with torch.no_grad():
                    scene_features = self.model.extract_scene_features(
                        nf_full, edge_index_dev, edge_attr_dev, track_ids=tids_tensor
                    )
                
                relation_embeddings = scene_features["relation_embeddings"]
                # 计算 L2 范数表示交互/"异常风险"强度
                norms = relation_embeddings.norm(dim=-1).cpu().numpy()
                edge_idx_np = edge_index.numpy()
                
                labels = None
                if "relation_labels" in scene_features:
                    labels = scene_features["relation_labels"].cpu().numpy()
                
                # 5. 可视化渲染：画框 + 连线
                for idx, t in enumerate(tracks):
                    x, y, w, h = t["x"], t["y"], t["w"], t["h"]
                    p1 = (int(x - w/2), int(y - h/2))
                    p2 = (int(x + w/2), int(y + h/2))
                    # 绿色框画车
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{t['track_id']}", (p1[0], max(0, p1[1] - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                # 画边
                drawn_pairs = set()
                for e in range(edge_idx_np.shape[1]):
                    src = edge_idx_np[0, e]
                    dst = edge_idx_np[1, e]
                    
                    if src >= n or dst >= n:
                        continue # 这里只渲染实体之间的边，避免画面过于脏乱

                    pair = (min(src, dst), max(src, dst))
                    if pair in drawn_pairs:
                        continue
                    drawn_pairs.add(pair)
                    
                    src_pos = (int(pos[src, 0]), int(pos[src, 1]))
                    dst_pos = (int(pos[dst, 0]), int(pos[dst, 1]))
                    
                    strength = norms[e]
                    # 当异常发生（关系向量激增时），连线变红、变粗
                    thickness = 1 + int(strength * 3)
                    color = (0, int(255 - min(strength*20, 255)), int(min(strength*40, 255))) # 青->红渐变
                    
                    cv2.line(frame, src_pos, dst_pos, color, thickness)
                    
                    # 取中点画得分和类别
                    mid = ((src_pos[0]+dst_pos[0])//2, (src_pos[1]+dst_pos[1])//2)
                    
                    text = f"{strength:.2f}"
                    if labels is not None:
                        rel_idx = labels[e]
                        rel_name = self.class_names_rel[rel_idx] if rel_idx < len(self.class_names_rel) else ""
                        text += f" | {rel_name}"
                        
                    cv2.putText(frame, text, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[INFO] Processed {frame_idx} frames...")

        if out is not None:
            out.release()
        print(f"[INFO] Video saved to {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 YOLOv8 端到端视频/图像处理和场景图生成")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--video", type=str, default=None, help="输入视频流（优先级低），例如 raw_video.mp4。若传递此参数，处理视频序列。")
    parser.add_argument("--image_dir", type=str, default=None, help="输入图片序列文件夹（优先级高）。若传递此参数，将按名称排序处理流。")
    parser.add_argument("--output", type=str, default="outputs/scene_graph_output.mp4", help="可视化带有隐式联系异常提示的视频输出路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="SceneGraphModel 核心自学习权重路径（可选）")
    
    args = parser.parse_args()
    
    pipeline = VideoSceneGraphPipeline(
        config_path=args.config,
        output_path=args.output,
        video_path=args.video,
        image_dir=args.image_dir,
        checkpoint_path=args.checkpoint
    )
    pipeline.process_video()
