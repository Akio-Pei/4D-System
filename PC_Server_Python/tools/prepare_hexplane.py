import os
import cv2
import json
import numpy as np
import argparse
import shutil
from glob import glob
from tqdm import tqdm

# === 强制设定标准尺寸 ===
TARGET_SIZE = 800


def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def apply_alignment(img_therm, params, target_h, target_w):
    tx = int(params.get("x", 0))
    ty = int(params.get("y", 0))
    scale = params.get("scale", 1.0)
    angle = params.get("angle", 0.0)

    th_h, th_w = img_therm.shape[:2]
    new_w = int(th_w * scale)
    new_h = int(th_h * scale)
    t_scaled = cv2.resize(img_therm, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    t_rotated = rotate_image(t_scaled, angle)

    canvas = np.zeros((target_h, target_w), dtype=img_therm.dtype)
    x1 = max(0, tx);
    y1 = max(0, ty)
    x2 = min(target_w, tx + new_w);
    y2 = min(target_h, ty + new_h)
    ox = x1 - tx;
    oy = y1 - ty

    if x2 > x1 and y2 > y1:
        src_h, src_w = t_rotated.shape[:2]
        if oy + (y2 - y1) <= src_h and ox + (x2 - x1) <= src_w:
            canvas[y1:y2, x1:x2] = t_rotated[oy:oy + (y2 - y1), ox:ox + (x2 - x1)]
    return canvas


def process_dataset(source_dir, output_dir):
    vis_dir = os.path.join(source_dir, "visual")
    therm_dir = os.path.join(source_dir, "thermal")
    param_file = os.path.join(source_dir, "align_params.json")

    out_img_dir = os.path.join(output_dir, "images")

    # 🔥 1. 如果新文件夹已存在，强制删除 (新名字一般不会被占用)
    if os.path.exists(output_dir):
        print(f"🧹 删除旧数据: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except Exception as e:
            print(f"⚠️ 无法删除文件夹，请手动删除: {output_dir}")
            return

    os.makedirs(out_img_dir, exist_ok=True)

    align_params = None
    if os.path.exists(param_file):
        try:
            with open(param_file, 'r') as f:
                align_params = json.load(f)
            print(f"✅ 加载对齐参数: {align_params}")
        except:
            pass
    else:
        print("⚠️ 未找到参数，使用强制拉伸")

    files = sorted(glob(os.path.join(vis_dir, "*.png")))
    if not files: print("❌ 没找到数据"); return

    print(f"🔥 处理中... 目标: {output_dir}")
    print(f"📏 强制尺寸: {TARGET_SIZE}x{TARGET_SIZE}")

    frames = []
    for idx, f_vis in enumerate(tqdm(files)):
        fname = os.path.basename(f_vis)

        # 1. 读原图
        img_v = cv2.imread(f_vis, cv2.IMREAD_GRAYSCALE)
        H, W = img_v.shape

        t_path = os.path.join(therm_dir, fname)
        if not os.path.exists(t_path): continue
        img_t_raw = cv2.imread(t_path, cv2.IMREAD_UNCHANGED)

        # 2. 对齐
        if align_params:
            t_aligned_16bit = apply_alignment(img_t_raw, align_params, H, W)
            t_norm = cv2.normalize(t_aligned_16bit, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            t_norm = cv2.resize(cv2.normalize(img_t_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (W, H))

        fusion_full = np.zeros((H, W, 3), dtype=np.uint8)
        fusion_full[:, :, 0] = img_v
        fusion_full[:, :, 1] = t_norm
        fusion_full[:, :, 2] = img_v

        # 3. 🔥【核心】强制裁剪为正方形
        crop_size = min(H, W)
        start_x = (W - crop_size) // 2
        start_y = (H - crop_size) // 2

        square_crop = fusion_full[start_y:start_y + crop_size, start_x:start_x + crop_size]

        if crop_size != TARGET_SIZE:
            final_img = cv2.resize(square_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
        else:
            final_img = square_crop

        # 写入
        cv2.imwrite(os.path.join(out_img_dir, fname), final_img)

        # 4. 写索引
        name_no_ext = os.path.splitext(fname)[0]
        frames.append({
            "file_path": f"./images/{name_no_ext}",
            "time": float(idx) / len(files),
            "transform_matrix": np.eye(4).tolist()
        })

    # 5. 计算 FOV
    orig_w_assumed = 1280
    orig_fov = np.radians(70)
    fl_px = orig_w_assumed / (2 * np.tan(orig_fov / 2))
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

    print(f"✅ 完成！新数据保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    process_dataset(args.input, args.output)