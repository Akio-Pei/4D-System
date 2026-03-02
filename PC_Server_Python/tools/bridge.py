import os
import sys
import subprocess
import argparse
import shutil
import json
import glob
import numpy as np
import cv2
import datetime
from tqdm import tqdm

# ================= 配置区域 =================
HEXPLANE_PYTHON = r"D:\Anaconda3\envs\hexplane\python.exe"
HEXPLANE_CODE_DIR = r"D:\CPP\HexPlane"
TARGET_SIZE = 800  # 强制正方形尺寸


# ===========================================

def process_images_internally(source_dir, output_dir):
    vis_dir = os.path.join(source_dir, "visual")
    therm_dir = os.path.join(source_dir, "thermal")
    out_img_dir = os.path.join(output_dir, "images")

    os.makedirs(out_img_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(vis_dir, "*.png")))
    if not files:
        print("❌ 错误：在 visual 文件夹里没找到图片！")
        return False

    print(f"🔥 [1/3] 强制重制 {len(files)} 张图片 ({TARGET_SIZE}x{TARGET_SIZE})...")

    frames = []
    for idx, f_vis in enumerate(tqdm(files)):
        fname = os.path.basename(f_vis)

        # 1. 读取
        img_v = cv2.imread(f_vis, cv2.IMREAD_GRAYSCALE)
        if img_v is None: continue
        H, W = img_v.shape

        # 2. 读取热成像
        t_path = os.path.join(therm_dir, fname)
        if os.path.exists(t_path):
            img_t_raw = cv2.imread(t_path, cv2.IMREAD_UNCHANGED)
            t_norm = cv2.normalize(img_t_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            t_norm = cv2.resize(t_norm, (W, H))
        else:
            t_norm = np.zeros_like(img_v)

        # 3. 合成
        fusion = np.zeros((H, W, 3), dtype=np.uint8)
        fusion[:, :, 0] = img_v
        fusion[:, :, 1] = t_norm
        fusion[:, :, 2] = img_v

        # 4. 强制中心裁剪
        crop_size = min(H, W)
        start_x = (W - crop_size) // 2
        start_y = (H - crop_size) // 2
        square = fusion[start_y:start_y + crop_size, start_x:start_x + crop_size]

        # 5. 强制缩放
        if crop_size != TARGET_SIZE:
            final_img = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        else:
            final_img = square

        # 6. 重命名为标准格式
        standard_name = f"{idx:06d}.png"
        save_path = os.path.join(out_img_dir, standard_name)

        if not cv2.imwrite(save_path, final_img):
            print(f"❌ 写入失败: {save_path}");
            return False

        frames.append({
            "file_path": f"./images/{os.path.splitext(standard_name)[0]}",
            "time": float(idx) / len(files),
            "transform_matrix": np.eye(4).tolist()
        })

    # 生成 JSON
    fl_px = 1280 / (2 * np.tan(np.radians(70) / 2))
    new_fov = 2 * np.arctan(TARGET_SIZE / (2 * fl_px))

    meta = {
        "camera_angle_x": new_fov,
        "fl_x": fl_px, "fl_y": fl_px,
        "cx": TARGET_SIZE / 2, "cy": TARGET_SIZE / 2,
        "w": TARGET_SIZE, "h": TARGET_SIZE,
        "frames": frames
    }

    for s in ["train", "test", "val"]:
        with open(os.path.join(output_dir, f"transforms_{s}.json"), 'w') as f:
            json.dump(meta, f, indent=4)

    return True


def generate_clean_config(output_dir):
    """
    生成配置文件，彻底移除 model 字段
    """
    new_cfg_path = os.path.join(output_dir, "run_config.yaml")
    safe_datadir = output_dir.replace("\\", "/")  # 路径清洗

    print(f"🔥 [2/3] 生成全新配置文件 (无 model 字段)...")

    # 🔥🔥🔥 核心修改：删除了 model: 那两行 🔥🔥🔥
    config_content = f"""
data:
  datadir: "{safe_datadir}"
  datasampler_type: "rays"
  downsample: 2.0

optim:
  n_iters: 3000
  batch_size: 4096
"""
    try:
        with open(new_cfg_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ 配置文件已生成: {new_cfg_path}")
        return new_cfg_path
    except Exception as e:
        print(f"❌ 配置文件生成失败: {e}")
        return None


def run(target_dir):
    target_dir = os.path.abspath(target_dir)
    exp_name = os.path.basename(target_dir)
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    output_dir = os.path.join(target_dir, f"hexplane_run_{timestamp}")

    print(f"🚀 创建环境: {output_dir}")

    # 1. 生成数据
    if not process_images_internally(target_dir, output_dir):
        return

    # 2. 生成全新配置
    clean_config_path = generate_clean_config(output_dir)
    if not clean_config_path:
        return

    print(f"🔥 [3/3] 启动训练...")

    cmd_list = [
        "cmd", "/k",
        HEXPLANE_PYTHON, "main.py",
        f"config={clean_config_path}",
        f"expname={exp_name}_{timestamp}"
    ]

    subprocess.Popen(
        cmd_list,
        cwd=HEXPLANE_CODE_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    print("✅ 窗口已弹出！祝你好运！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    run(args.target)