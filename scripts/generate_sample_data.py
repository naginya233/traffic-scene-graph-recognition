"""
生成模拟检测框 CSV 数据。

输出格式: frame_id, track_id, class, x, y, w, h, vx, vy

模拟多辆车在十字路口附近的运动轨迹。
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_sample_data(
    output_path: str = "data/sample/detections.csv",
    num_frames: int = 200,
    num_vehicles: int = 10,
    seed: int = 42,
):
    """生成模拟交通场景检测框数据。

    Args:
        output_path: 输出 CSV 路径
        num_frames: 总帧数
        num_vehicles: 车辆数量
        seed: 随机种子
    """
    np.random.seed(seed)

    classes = ["car", "truck", "bus", "pedestrian", "cyclist"]
    frame_width = 1920
    frame_height = 1080

    records = []

    # 为每辆车生成初始状态
    vehicles = []
    for i in range(num_vehicles):
        cls = np.random.choice(classes, p=[0.5, 0.15, 0.1, 0.15, 0.1])

        # bbox 尺寸根据类别决定
        if cls == "car":
            w, h = np.random.uniform(60, 120), np.random.uniform(40, 80)
        elif cls == "truck":
            w, h = np.random.uniform(100, 180), np.random.uniform(50, 90)
        elif cls == "bus":
            w, h = np.random.uniform(120, 200), np.random.uniform(50, 80)
        elif cls == "pedestrian":
            w, h = np.random.uniform(20, 40), np.random.uniform(40, 80)
        else:  # cyclist
            w, h = np.random.uniform(20, 50), np.random.uniform(30, 70)

        # 初始位置（随机分布在画面中）
        x = np.random.uniform(100, frame_width - 100)
        y = np.random.uniform(100, frame_height - 100)

        # 初始速度（像素/帧）
        speed = np.random.uniform(1, 8)
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

        # 车辆出现的起始帧和持续时间
        start_frame = np.random.randint(0, max(1, num_frames // 3))
        duration = np.random.randint(num_frames // 2, num_frames)

        vehicles.append({
            "track_id": i + 1,
            "class": cls,
            "x": x, "y": y,
            "w": w, "h": h,
            "vx": vx, "vy": vy,
            "start_frame": start_frame,
            "end_frame": min(start_frame + duration, num_frames),
        })

    # 逐帧生成数据
    for frame_id in range(num_frames):
        for v in vehicles:
            if frame_id < v["start_frame"] or frame_id >= v["end_frame"]:
                continue

            # 当前位置
            dt = frame_id - v["start_frame"]
            x = v["x"] + v["vx"] * dt
            y = v["y"] + v["vy"] * dt

            # 添加少量噪声（模拟检测抖动）
            x += np.random.normal(0, 1.5)
            y += np.random.normal(0, 1.5)

            # 边界反弹
            if x < 50 or x > frame_width - 50:
                v["vx"] *= -1
                x = np.clip(x, 50, frame_width - 50)
            if y < 50 or y > frame_height - 50:
                v["vy"] *= -1
                y = np.clip(y, 50, frame_height - 50)

            # 偶尔注入异常事件（急刹车/突然变道）
            if np.random.random() < 0.02:
                v["vx"] += np.random.normal(0, 10)
                v["vy"] += np.random.normal(0, 10)

            # 速度加噪声
            vx = v["vx"] + np.random.normal(0, 0.5)
            vy = v["vy"] + np.random.normal(0, 0.5)

            records.append({
                "frame_id": frame_id,
                "track_id": v["track_id"],
                "class": v["class"],
                "x": round(x, 2),
                "y": round(y, 2),
                "w": round(v["w"], 2),
                "h": round(v["h"], 2),
                "vx": round(vx, 2),
                "vy": round(vy, 2),
            })

    # 保存 CSV
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[INFO] Generated {len(df)} detections across {num_frames} frames")
    print(f"[INFO] Saved to: {output_path}")
    print(f"[INFO] Vehicles: {num_vehicles}")
    print(f"[INFO] Classes distribution:")
    print(df["class"].value_counts().to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟检测框数据")
    parser.add_argument("--output", default="data/sample/detections.csv", help="输出路径")
    parser.add_argument("--frames", type=int, default=200, help="帧数")
    parser.add_argument("--vehicles", type=int, default=10, help="车辆数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    generate_sample_data(args.output, args.frames, args.vehicles, args.seed)
