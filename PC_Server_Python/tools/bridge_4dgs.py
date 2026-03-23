import os
import sys
import subprocess
import argparse
import json
import glob
import shutil
import cv2
import numpy as np

GS_PYTHON = r"D:\Anaconda3\envs\met_4dgs\python.exe"
GS_CODE_DIR = r"D:\CPP\4DGaussians"
CUDA_BIN_PATH = r"D:\NVIDIA\V12.6\bin"


def build_trimodal_fused_data(target_dir):
    img_dest_dir = os.path.join(target_dir, "images")
    if os.path.exists(img_dest_dir): shutil.rmtree(img_dest_dir)
    os.makedirs(img_dest_dir, exist_ok=True)

    img_src = os.path.join(target_dir, "visual")
    files = sorted(glob.glob(os.path.join(img_src, "*.png")))
    frames = []
    angles = [("c", 0.0, 0.0, 0, 0), ("l", -0.05, 0.0, -4, 0), ("r", 0.05, 0.0, 4, 0), ("u", 0.0, 0.05, 0, -4),
              ("d", 0.0, -0.05, 0, 4)]

    print(f"🧠 [纯净显式编码] 启动安全区过滤，彻底消灭蓝底与噪点...")
    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        img_t = cv2.imread(os.path.join(target_dir, "thermal", fname))
        img_e = cv2.imread(os.path.join(target_dir, "event_hq", fname), 0)

        # 🟢 1. 提取最纯净的 Thermal 主体（绝不包含蓝色冷底）
        r_ch, b_ch = img_t[:, :, 2].astype(np.int16), img_t[:, :, 0].astype(np.int16)
        hot_mask = ((r_ch - b_ch > 20) & (r_ch > 100)).astype(np.uint8)

        cnts_t, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_hot_mask = np.zeros_like(hot_mask)
        if cnts_t:
            c_max_t = max(cnts_t, key=cv2.contourArea)
            cv2.drawContours(clean_hot_mask, [c_max_t], -1, 1, -1)

        # 🟢 2. 划定绝对安全区：严防死守，只允许在发热体周围 25 像素内存活 Event！
        safe_roi = cv2.dilate(clean_hot_mask, np.ones((25, 25), np.uint8), iterations=1)

        # 🟢 3. 提取并清洗 Event 边界（彻底切断右侧悬浮噪点的桥梁）
        _, e_mask = cv2.threshold(img_e, 30, 1, cv2.THRESH_BINARY)
        clean_e_mask = cv2.bitwise_and(e_mask, safe_roi)

        # 🟢 4. 显式物理颜色编码 (拒绝蓝底泄漏)
        color_coding_canvas = np.zeros_like(img_t)

        # 内部：只拷贝真正的发热区域颜色（完美避开蓝色背景）
        clean_hot_mask_3c = np.dstack([clean_hot_mask] * 3)
        color_coding_canvas = np.where(clean_hot_mask_3c == 1, img_t, color_coding_canvas)

        # 边缘：将干净的事件信号涂成纯白
        color_coding_canvas[clean_e_mask == 1] = [255, 255, 255]

        # 🟢 5. 生成抗锯齿融合掩码
        e_dilated = cv2.dilate(clean_e_mask, np.ones((5, 5), np.uint8), iterations=1)
        fused_geometry = cv2.bitwise_or(clean_hot_mask, e_dilated)

        # 此时只需极小内核平滑即可，不会再有误连！
        align_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused_geometry = cv2.morphologyEx(fused_geometry, cv2.MORPH_CLOSE, align_kernel)

        final_gaussian_mask = cv2.GaussianBlur(fused_geometry.astype(np.float32), (3, 3), 0)
        final_gaussian_mask_3c = np.dstack([final_gaussian_mask] * 3)

        # 得到绝对干净、边缘白、内部热成像且底噪为纯黑的训练图
        final_img = (color_coding_canvas.astype(np.float32) * final_gaussian_mask_3c).astype(np.uint8)

        for suffix, tx, ty, du, dv in angles:
            M = np.float32([[1, 0, du], [0, 1, dv]])
            warped = cv2.warpAffine(final_img, M, (200, 200), borderValue=(0, 0, 0))
            out_name = f"f_{idx:03d}_{suffix}.png"
            cv2.imwrite(os.path.join(img_dest_dir, out_name), warped)

            c2w = [[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, ty], [0.0, 0.0, 1.0, 1.8], [0.0, 0.0, 0.0, 1.0]]
            frames.append({"file_path": f"./images/{out_name.split('.')[0]}", "time": float(idx) / len(files),
                           "transform_matrix": c2w})

    meta = {"camera_angle_x": 0.78, "fl_x": 242.0, "fl_y": 242.0, "w": 200, "h": 200, "frames": frames}
    for s in ["train", "test", "val"]:
        with open(os.path.join(target_dir, f"transforms_{s}.json"), 'w') as f:
            json.dump(meta, f, indent=4)
    return True


def run_met_4dgs(target_dir):
    target_dir = os.path.abspath(target_dir)
    if not build_trimodal_fused_data(target_dir): return

    model_out_dir = os.path.join(target_dir, "FINAL_CLEAN_MODEL")
    if os.path.exists(model_out_dir): shutil.rmtree(model_out_dir)

    bad_ply = os.path.join(target_dir, "init_MET_4DGS.ply")
    if os.path.exists(bad_ply):
        os.remove(bad_ply)
        print("🗑️ 已清理有毒先验点云，恢复原生生长引擎！")

    env = os.environ.copy()
    if os.path.exists(CUDA_BIN_PATH):
        env["PATH"] = CUDA_BIN_PATH + os.pathsep + env.get("PATH", "")

    cmd = [
        GS_PYTHON, "-u", "train.py",
        "-s", target_dir,
        "-m", model_out_dir,
        "--iterations", "15000",
        "--sh_degree", "0",
        "-r", "1"
    ]

    process = subprocess.Popen(cmd, cwd=GS_CODE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               bufsize=0)
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None: break
        if char: sys.stdout.write(char.decode('utf-8', 'replace'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    run_met_4dgs(parser.parse_args().target)