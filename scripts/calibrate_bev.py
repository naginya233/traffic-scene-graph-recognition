"""
BEV 提取与验证工具。

提供两种模式:
1. --dair: 从 DAIR-V2X 标定数据集自动推导所有场景的 BEV 单应矩阵。
2. --manual: 给定一张图片，交互式 4 点标定。
"""

import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.traffic_scene_graph.utils.bev_transform import BEVTransform


def calibrate_from_dair(dair_root: str, output_dir: str):
    """从 DAIR-V2X 数据集自动生成标定文件。"""
    dair_path = Path(dair_root)
    if not dair_path.exists():
        print(f"Error: {dair_root} does not exist.")
        return

    data_info_path = dair_path / "single-infrastructure-side" / "single-infrastructure-side" / "data_info.json"
    if not data_info_path.exists():
        print(f"Error: data_info.json not found at {data_info_path}")
        return

    with open(data_info_path, "r") as f:
        data_info = json.load(f)

    # 按路口 (intersection_loc) 分组生成代表性的标定
    intersections = {}

    for item in data_info:
        loc = item.get("intersection_loc", "unknown")
        if loc not in intersections:
            intersections[loc] = item

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(intersections)} intersections. Generating BEV calibrations...")

    for loc, item in intersections.items():
        intrinsic_rel = item["calib_camera_intrinsic_path"]
        extrinsic_rel = item["calib_virtuallidar_to_camera_path"]

        intrinsic_path = dair_path / "single-infrastructure-side" / "single-infrastructure-side" / intrinsic_rel
        extrinsic_path = dair_path / "single-infrastructure-side" / "single-infrastructure-side" / extrinsic_rel

        if intrinsic_path.exists() and extrinsic_path.exists():
            try:
                # 假设地面Z = -1.0 米 (基于经验值)
                transform = BEVTransform.from_dair_calibration(
                    intrinsic_path=str(intrinsic_path),
                    extrinsic_path=str(extrinsic_path),
                    ground_z=-1.0 
                )
                
                out_file = out_path / f"bev_calib_dair_{loc}.json"
                transform.save_calibration(str(out_file))
                print(f"  Saved: {out_file.name}")
            except Exception as e:
                print(f"  Error generating calibration for {loc}: {e}")
        else:
            print(f"  Missing calibration files for {loc}")

    print("Done generating from DAIR-V2X.")


# 手动标定交互逻辑
ref_points = []
def click_event(event, x, y, flags, param):
    global ref_points
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append([x, y])
        img = param[0]
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"Pt {len(ref_points)}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Image", img)


def calibrate_manual(image_path: str, output_path: str):
    """交互式四点标定。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    h, w = img.shape[:2]
    img_disp = img.copy()

    print("=== 手动 BEV 标定 ===")
    print("1. 请在图像上按顺时针顺序点击 4 个共面的地面参考点（例如一个矩形区域的四角）。")
    print("2. 选完 4 个点后，系统将要求输入它们在现实世界中的相对坐标（米）。")
    print("按 'q' 提早退出。")

    cv2.imshow("Image", img_disp)
    cv2.setMouseCallback("Image", click_event, [img_disp])

    while len(ref_points) < 4:
        # 给点时间刷新UI
        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    print("\n已记录 4 个点像素坐标:")
    for i, p in enumerate(ref_points):
        print(f"  Pt {i+1}: ({p[0]}, {p[1]})")

    print("\n请输入这 4 个点在地面(BEV)的坐标 (中心点或角点均可, 单位为米):")
    print("格式例如: 0,0  (表示(0米, 0米))")
    
    dst_points = []
    for i in range(4):
        valid = False
        while not valid:
            val = input(f"输入 Pt {i+1} 真实坐标 X,Y (米): ")
            parts = val.split(',')
            if len(parts) == 2:
                try:
                    X = float(parts[0].strip())
                    Y = float(parts[1].strip())
                    dst_points.append([X, Y])
                    valid = True
                except ValueError:
                    print("格式错误。必须是两个逗号分隔的数字。")
            else:
                print("格式错误。必须是两个逗号分隔的数字。")

    print("\n正在生成单应矩阵...")
    transform = BEVTransform(
        src_points=np.array(ref_points),
        dst_points=np.array(dst_points),
        frame_width=w,
        frame_height=h
    )

    transform.save_calibration(output_path)
    print(f"标定参数已保存至 {output_path}")

    # 验证显示
    print("正在显示鸟瞰验证图...")
    warped = transform.warp_frame(img, output_size=(800, 800), bev_range=(min(p[0] for p in dst_points)-10, min(p[1] for p in dst_points)-10, max(p[0] for p in dst_points)+10, max(p[1] for p in dst_points)+10))
    cv2.imshow("Warped Validation", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEV 校准工具")
    parser.add_argument("--dair", type=str, help="DAIR-V2X 根目录，批量处理")
    parser.add_argument("--image", type=str, help="用于手动标定的输入图像路径")
    parser.add_argument("--output", type=str, default="configs/bev_calibration.json", help="输出标定 JSON 文件或目录")
    
    args = parser.parse_args()

    if args.dair:
        calibrate_from_dair(args.dair, args.output)
    elif args.image:
        calibrate_manual(args.image, args.output)
    else:
        print("请指定 --dair [DAIR-V2X root] 或 --image [image_path]")
        parser.print_help()
