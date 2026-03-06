import os
import sys
import subprocess
import argparse
import json
import glob
import datetime
import numpy as np
import cv2

HEXPLANE_PYTHON = r"D:\Anaconda3\envs\hexplane\python.exe"
HEXPLANE_CODE_DIR = r"D:\CPP\HexPlane"
TARGET_SIZE = 200


def build_metadata(target_dir, output_dir):
    img_src = os.path.join(target_dir, "visual")
    all_files = sorted(glob.glob(os.path.join(img_src, "*.png")))
    if not all_files: return False

    files = all_files[:60] if len(all_files) > 60 else all_files
    print(f"✅ [Data] 准备 {len(files)} 帧连续素材...", flush=True)

    img_dest_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dest_dir, exist_ok=True)

    frames = []
    c2w = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    for idx, fpath in enumerate(files):
        img = cv2.imread(fpath)
        cv2.imwrite(os.path.join(img_dest_dir, f"{idx:06d}.png"), img)
        frames.append({
            "file_path": f"./images/{idx:06d}",
            "time": float(idx) / len(files),
            "transform_matrix": c2w
        })

    orig_fov = 0.7
    fl_px = TARGET_SIZE / (2 * np.tan(orig_fov / 2))

    meta_train = {
        "camera_angle_x": orig_fov, "fl_x": fl_px, "fl_y": fl_px,
        "cx": TARGET_SIZE / 2, "cy": TARGET_SIZE / 2,
        "w": TARGET_SIZE, "h": TARGET_SIZE, "frames": frames
    }

    meta_eval = meta_train.copy()
    meta_eval["frames"] = frames[::12][:5]

    for split, data in [("train", meta_train), ("test", meta_eval), ("val", meta_eval)]:
        with open(os.path.join(output_dir, f"transforms_{split}.json"), 'w') as f:
            json.dump(data, f, indent=4)

    return True


def run(target_dir):
    target_dir = os.path.abspath(target_dir)
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    exp_name = f"HEX_{timestamp}"
    output_dir = os.path.join(target_dir, f"hex_data_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    if not build_metadata(target_dir, output_dir): return

    cfg_path = os.path.join(output_dir, "run_config.yaml").replace('\\', '/')
    config_content = f"""
data:
  datadir: "{output_dir.replace('\\', '/')}"
  dataset_name: "dnerf"
  downsample: 1.0
render_only: False
systems:
  ckpt: null
optim:
  n_iters: 3000      
  batch_size: 1024    
"""
    with open(cfg_path, 'w') as f:
        f.write(config_content)

    print(f"🚀 [Bridge] 开启 4D 隐式场核心训练...", flush=True)

    cmd = [
        HEXPLANE_PYTHON, "-u", "main.py",
        f"config={cfg_path}", f"expname={exp_name}",
        "systems.ckpt=null", "render_only=False"
    ]
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    process = subprocess.Popen(
        cmd, cwd=HEXPLANE_CODE_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
    )
    while True:
        char = process.stdout.read(1)
        if not char and process.poll() is not None: break
        if char:
            char_str = char.decode('utf-8', 'replace')
            if char_str == '\r':
                sys.stdout.write('\n')
            else:
                sys.stdout.write(char_str)
            sys.stdout.flush()

    print(f"\n🚀 [Bridge] 训练完成！正在将神经网络“烘焙”为 4D 光场矩阵 (约需3-5分钟)...", flush=True)

    matrix_frames = []
    V_STEPS = 36
    T_STEPS = 30

    for t_idx in range(T_STEPS):
        t_val = t_idx / max(1, (T_STEPS - 1))
        for v_idx in range(V_STEPS):
            angle = np.radians(v_idx * 10)
            c, s = float(np.cos(angle)), float(np.sin(angle))
            c2w = [
                [c, 0, s, 3.0 * s],
                [0, 1, 0, 0],
                [-s, 0, c, 3.0 * c],
                [0, 0, 0, 1]
            ]
            matrix_frames.append({
                "file_path": f"./images/000000",  # 🔥 致命修复：强行借用存在的第0帧图片骗过 DataLoader，防止其崩溃！
                "time": float(t_val),
                "transform_matrix": c2w
            })

    with open(os.path.join(output_dir, "transforms_test.json"), 'r') as f:
        meta_eval = json.load(f)
    meta_eval["frames"] = matrix_frames
    with open(os.path.join(output_dir, "transforms_test.json"), 'w') as f:
        json.dump(meta_eval, f, indent=4)

    ckpt_rel_path = f"log/{exp_name}/{exp_name}.th"

    render_cmd = [
        HEXPLANE_PYTHON, "-u", "main.py",
        f"config={cfg_path}", f"expname={exp_name}",
        "render_only=True", "render_test=True",
        f"systems.ckpt={ckpt_rel_path}"
    ]

    render_process = subprocess.Popen(
        render_cmd, cwd=HEXPLANE_CODE_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
    )
    while True:
        char = render_process.stdout.read(1)
        if not char and render_process.poll() is not None: break
        if char:
            char_str = char.decode('utf-8', 'replace')
            if char_str == '\r':
                sys.stdout.write('\n')
            else:
                sys.stdout.write(char_str)
            sys.stdout.flush()

    res_dir = os.path.join(HEXPLANE_CODE_DIR, "log", exp_name, "imgs_test_all").replace('\\', '/')
    print(f"\n🏁 [RESULT_PATH]:{res_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    run(parser.parse_args().target)