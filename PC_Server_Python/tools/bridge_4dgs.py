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
    angles = [("c", 0.0, 0.0, 0, 0), ("l", -0.05, 0.0, -4, 0), ("r", 0.05, 0.0, 4, 0), ("u", 0.0, 0.05, 0, -4), ("d", 0.0, -0.05, 0, 4)]

    print(f"🧠 [极致防畸变] 正在为 {len(files)} 帧图片注入像素级视差...")
    for idx, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        img_v = cv2.imread(fpath)
        img_t = cv2.imread(os.path.join(target_dir, "thermal", fname))
        img_e = cv2.imread(os.path.join(target_dir, "event_hq", fname), 0)

        r_ch, b_ch = img_t[:, :, 2].astype(np.int16), img_t[:, :, 0].astype(np.int16)
        hot_mask = cv2.dilate((r_ch > b_ch + 20).astype(np.uint8), np.ones((5, 5)))
        img_v[hot_mask == 0] = [0, 0, 0]
        img_v[img_e > 30] = [0, 255, 0]

        for suffix, tx, ty, du, dv in angles:
            M = np.float32([[1, 0, du], [0, 1, dv]])
            warped = cv2.warpAffine(img_v, M, (200, 200), borderValue=(0, 0, 0))
            out_name = f"f_{idx:03d}_{suffix}.png"
            cv2.imwrite(os.path.join(img_dest_dir, out_name), warped)

            c2w = [[1.0, 0.0, 0.0, tx], [0.0, 1.0, 0.0, ty], [0.0, 0.0, 1.0, 1.8], [0.0, 0.0, 0.0, 1.0]]
            frames.append({"file_path": f"./images/{out_name.split('.')[0]}", "time": float(idx) / len(files), "transform_matrix": c2w})

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

    # 🟢 彻底删除有毒的先验点云，释放 4DGS 引擎原生增殖能力！
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
        "-r", "1"  # 🟢 使用 1 倍原生分辨率，产生巨大屏幕梯度刺激增殖！
    ]

    process = subprocess.Popen(cmd, cwd=GS_CODE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None: break
        if char: sys.stdout.write(char.decode('utf-8', 'replace'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    run_met_4dgs(parser.parse_args().target)