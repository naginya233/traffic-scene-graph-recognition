"""
推理入口脚本：加载训练好的模型 → 输入新检测帧 → 输出场景图特征
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traffic_scene_graph.data import TrafficFrameDataset
from traffic_scene_graph.models import SceneGraphModel
from traffic_scene_graph.utils import GraphBuilder, BEVTransform, ZoneManager


def main():
    parser = argparse.ArgumentParser(description="交通场景图推理")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型检查点路径 (默认使用配置中的路径)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入 CSV 文件路径 (默认使用配置中的路径)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备",
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = args.input or config["data"]["csv_path"]
    checkpoint_path = args.checkpoint or config["inference"]["checkpoint_path"]
    output_dir = args.output_dir or config["inference"].get("output_dir", "outputs")

    print("=" * 60)
    print("  边缘端交通语义场景图 - 推理")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Input:      {csv_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Device:     {device}")
    print("=" * 60)

    # 加载模型
    model = SceneGraphModel(config).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFO] Loaded model from {checkpoint_path}")
    else:
        print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
        print("[WARNING] Using randomly initialized model (for testing only)")

    model.eval()

    # 构建图构建器
    graph_cfg = config.get("graph", {})
    graph_builder = GraphBuilder(
        kalman_predict_steps=graph_cfg.get("kalman_predict_steps", 5),
        iou_threshold=graph_cfg.get("iou_threshold", 0.0),
        distance_threshold=graph_cfg.get("distance_threshold", 15.0), # BEV距离
    )

    # 初始化 BEV
    bev_cfg = config.get("bev", {})
    try:
        bev_transform = BEVTransform.from_calibration_file(bev_cfg.get("calibration_file", "configs/bev_calibration.json"))
        print("[INFO] BEV Transform loaded.")
    except FileNotFoundError:
        print("[WARNING] BEV calibration file not found. Using default homography.")
        bev_transform = BEVTransform()

    # 初始化区域管理器
    zone_cfg = config.get("zone", {})
    zone_manager = ZoneManager(
        bev_range=zone_cfg.get("bev_range", [0.0, 0.0, 100.0, 100.0]),
        grid_rows=zone_cfg.get("grid_rows", 3),
        grid_cols=zone_cfg.get("grid_cols", 3),
        node_feature_dim=config.get("node_feature", {}).get("input_dim", 12)
    )
    zone_features = zone_manager.generate_zone_features().to(device)
    
    class_names = config.get("classifier", {}).get("class_names", [])

    # 加载数据
    dataset = TrafficFrameDataset(
        csv_path=csv_path,
        frame_width=config["data"].get("frame_width", 1920),
        frame_height=config["data"].get("frame_height", 1080),
        sequence_length=1,
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Processing {len(dataset)} frames...")

    all_results = []

    with torch.no_grad():
        model.reset_temporal_state()
        graph_builder.reset()

        for idx in range(len(dataset)):
            sample = dataset[idx]

            n = sample["num_nodes"][0].item()
            if n < 2:
                continue

            nf = sample["node_features"][0, :n].to(device)
            tids = sample["track_ids"][0, :n].numpy()
            pos = sample["raw_positions"][0, :n].numpy()
            vel = sample["raw_velocities"][0, :n].numpy()
            bbox = sample["raw_bboxes"][0, :n].numpy()

            # BEV 投影
            bev_pos = bev_transform.pixel_to_bev(pos)
            bev_vel = bev_transform.velocity_to_bev(pos, vel)
            bev_bbox = bev_transform.bbox_to_bev_footprint(bbox)

            # 建图
            edge_index_ent, edge_attr_ent = graph_builder.build_graph(
                positions=bev_pos, velocities=bev_vel, bboxes=bev_bbox, track_ids=tids
            )
            
            # 环境节点边
            edge_index_zone, edge_attr_zone = zone_manager.build_entity_zone_edges(
                entity_bev_pos=bev_pos, num_entity_nodes=n, zone_node_offset=n
            )

            # 合并特征和图
            nf_full = torch.cat([nf, zone_features], dim=0)

            if edge_index_ent.shape[1] > 0 and edge_index_zone.shape[1] > 0:
                edge_index = torch.cat([edge_index_ent, edge_index_zone], dim=1)
                edge_attr = torch.cat([edge_attr_ent, edge_attr_zone], dim=0)
            elif edge_index_ent.shape[1] > 0:
                edge_index = edge_index_ent
                edge_attr = edge_attr_ent
            else:
                edge_index = edge_index_zone
                edge_attr = edge_attr_zone

            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            
            tids_pad = np.concatenate([tids, np.full(zone_manager.num_zones, -1)])
            tids_tensor = torch.from_numpy(tids_pad).long().to(device)

            # 推理
            scene_features = model.extract_scene_features(
                nf_full, edge_index, edge_attr, track_ids=tids_tensor
            )

            frame_id = sample["frame_ids"][0].item()

            entity_relations = []
            environment_relations = []
            
            if "relation_labels" in scene_features:
                labels = scene_features["relation_labels"].cpu().numpy()
                confidences = scene_features["relation_confidences"].cpu().numpy()
                edges = edge_index.cpu().numpy()
                
                # 去重集合，为了简化日志
                added_edges = set()

                for e_idx in range(edges.shape[1]):
                    src, dst = edges[0, e_idx], edges[1, e_idx]
                    rel_idx = labels[e_idx]
                    rel_name = class_names[rel_idx] if rel_idx < len(class_names) else "unknown"
                    conf = float(confidences[e_idx])
                    
                    if src < n and dst < n:
                        # 实体-实体 (有向边记录一次即可，如果只记录一对关系)
                        pair = (min(src, dst), max(src, dst))
                        if pair not in added_edges:
                            entity_relations.append({
                                "source_track": int(tids[src]),
                                "target_track": int(tids[dst]),
                                "relation": rel_name,
                                "confidence": round(conf, 4)
                            })
                            added_edges.add(pair)
                    else:
                        # 实体-环境
                        ent_node = src if src < n else dst
                        zone_node = dst if src < n else src
                        zone_idx = zone_node - n
                        zone_name = zone_manager.get_zone_name(zone_idx)
                        
                        pair = (ent_node, zone_node)
                        if pair not in added_edges:
                            environment_relations.append({
                                "track_id": int(tids[ent_node]),
                                "zone": zone_name,
                                "relation": rel_name,
                                "confidence": round(conf, 4)
                            })
                            added_edges.add(pair)

            result = {
                "frame_id": int(frame_id),
                "num_nodes": int(n),
                "num_edges": int(edge_index.shape[1]),
                "entity_relations": entity_relations,
                "environment_relations": environment_relations
            }
            all_results.append(result)

            if idx < 5:
                print(f"  Frame {frame_id}: {n} entities, {len(entity_relations)} ent-ents, {len(environment_relations)} ent-zones")

    # 保存结果摘要
    summary_path = os.path.join(output_dir, "inference_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Inference complete. Processed {len(all_results)} frames")
    print(f"[INFO] Results saved to: {summary_path}")

    # 打印统计
    if all_results:
        avg_edges = np.mean([r["num_edges"] for r in all_results])
        avg_ent_rels = np.mean([len(r["entity_relations"]) for r in all_results])
        print(f"[INFO] Avg edges/frame: {avg_edges:.1f}")
        print(f"[INFO] Avg entity relations: {avg_ent_rels:.1f}")


if __name__ == "__main__":
    main()
