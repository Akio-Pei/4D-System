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

    # 🔥 满血版 1080 帧矩阵生成逻辑
    print(f"\n🚀 [Bridge] 训练完成！正在将神经网络“烘焙”为 4D 光场矩阵 (共 1080 帧)...", flush=True)

    matrix_frames = []
    V_STEPS = 36  # 每 10 度转一圈，共 36 个视角
    T_STEPS = 30  # 均分 30 个时间步

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
                "file_path": f"./images/000000",
                "time": float(t_val),
                "transform_matrix": c2w
            })

    with open(os.path.join(output_dir, "transforms_test.json"), 'r') as f:
        meta_eval = json.load(f)
    meta_eval["frames"] = matrix_frames
    with open(os.path.join(output_dir, "transforms_test.json"), 'w') as f:
        json.dump(meta_eval, f, indent=4)

    # 🟢 终极补丁：全盘递归搜索权重文件，拒绝盲猜路径！
    log_base_dir = os.path.join(HEXPLANE_CODE_DIR, "log")
    ckpt_files = glob.glob(os.path.join(log_base_dir, "**", "*.th"), recursive=True)

    actual_ckpt_path = None
    for f in ckpt_files:
        if exp_name in f:
            actual_ckpt_path = f
            break

    if not actual_ckpt_path:
        print(f"\n❌ 找不到包含 {exp_name} 的权重文件！请检查 HexPlane 的 log 目录。", flush=True)
        return

    print(f"\n✅ 自动定位到模型权重: {actual_ckpt_path}", flush=True)

    render_cmd = [
        HEXPLANE_PYTHON, "-u", "main.py",
        f"config={cfg_path}", f"expname={exp_name}",
        "render_only=True", "render_test=True",
        f"systems.ckpt={actual_ckpt_path}"
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

    # 🟢 动态锁定图片渲染目录（紧跟模型所在的同级目录），然后烘焙为丝滑 60FPS
    exp_log_dir = os.path.dirname(actual_ckpt_path)
    res_dir = os.path.join(exp_log_dir, "imgs_test_all")

    img_files = sorted(glob.glob(os.path.join(res_dir, "*.png")))
    if img_files:
        print(f"\n🎬 [Video Engine] 正在将 {len(img_files)} 帧序列合成 60FPS MP4 视频...", flush=True)
        first_img = cv2.imread(img_files[0])
        h, w = first_img.shape[:2]

        out_video_path = os.path.join(target_dir, f"hexplane_60fps_{timestamp}.mp4")

        # 使用 MP4V 编码，兼容性极好
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(out_video_path, fourcc, 60.0, (w, h))
        for img_path in img_files:
            out_vid.write(cv2.imread(img_path))
        out_vid.release()
        print(f"✅ 视频烘焙完成！已存至: {out_video_path}", flush=True)
    else:
        print(f"⚠️ 未找到渲染出的图片，检查 {res_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    run(parser.parse_args().target)