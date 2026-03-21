import os
import cv2
import json
import numpy as np
import argparse
import struct


def generate_met_pointcloud(target_dir):
    print(f"🚀 [GS Init] 启动体积云爆炸算法 (打破点云死锁与假动作)...")

    with open(os.path.join(target_dir, "transforms_train.json"), 'r') as f:
        meta = json.load(f)

    W, H = meta['w'], meta['h']
    fl_x, fl_y = meta['fl_x'], meta['fl_y']
    # 智能计算光学中心，彻底解决 KeyError: 'cx' 崩溃
    cx, cy = W / 2.0, H / 2.0

    c2w = np.array(meta['frames'][0]['transform_matrix'])

    vis_dir = os.path.join(target_dir, "visual")
    therm_dir = os.path.join(target_dir, "thermal")
    evt_dir = os.path.join(target_dir, "event_hq")
    raw_files = sorted([f for f in os.listdir(vis_dir) if f.endswith(".png")])

    # 强行只读取绝对的第一帧
    raw_fname = raw_files[0]

    img_rgb = cv2.cvtColor(cv2.imread(os.path.join(vis_dir, raw_fname)), cv2.COLOR_BGR2RGB)
    img_t_color = cv2.imread(os.path.join(therm_dir, raw_fname))

    r_ch = img_t_color[:, :, 2].astype(np.int16)
    b_ch = img_t_color[:, :, 0].astype(np.int16)
    hot_mask = ((r_ch - b_ch > 20) & (r_ch > 100)).astype(np.uint8)

    img_e = cv2.imread(os.path.join(evt_dir, raw_fname), cv2.IMREAD_GRAYSCALE)
    e_dilated = cv2.dilate(img_e, np.ones((9, 9), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(e_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    e_filled = np.zeros_like(img_e)
    if cnts: cv2.drawContours(e_filled, cnts, -1, 1, -1)

    final_mask = np.logical_and(hot_mask > 0, e_filled > 0).astype(np.uint8)
    final_mask = cv2.dilate(final_mask, np.ones((5, 5), np.uint8), iterations=1)

    t_float = r_ch.astype(np.float32) * final_mask
    t_norm = np.zeros_like(t_float)
    if t_float.max() > 0: t_norm = t_float / 255.0

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = final_mask > 0
    u_valid = u[valid]
    v_valid = v[valid]
    rgb_valid = img_rgb[valid]
    z_norm_valid = t_norm[valid]

    # 🚀 终极杀招：体积云爆炸 (Volumetric Cloud Injection)
    N_REPEATS = 10  # 强行放大 10 倍基数

    u_valid = np.repeat(u_valid, N_REPEATS)
    v_valid = np.repeat(v_valid, N_REPEATS)
    rgb_valid = np.repeat(rgb_valid, N_REPEATS, axis=0)
    z_norm_valid = np.repeat(z_norm_valid, N_REPEATS)

    # 加入 X/Y 像素级微小混沌扰动 (打破绝对死锁)
    u_valid = u_valid.astype(np.float32) + np.random.uniform(-1.0, 1.0, size=u_valid.shape)
    v_valid = v_valid.astype(np.float32) + np.random.uniform(-1.0, 1.0, size=v_valid.shape)

    BASE_DEPTH = 3.0
    RELIEF_THICKNESS = 0.5
    # 加入 Z 轴大尺度混沌扰动 (强制赋予厚度，产生重叠梯度)
    Z_cam = BASE_DEPTH - (z_norm_valid * RELIEF_THICKNESS)
    Z_cam += np.random.normal(0, 0.1, size=Z_cam.shape)

    X_cam = (u_valid - cx) * Z_cam / fl_x
    Y_cam = (v_valid - cy) * Z_cam / fl_y

    pts_camera = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(X_cam)], axis=1)
    pts_world = (c2w @ pts_camera.T).T[:, :3]

    target_num = min(150000, pts_world.shape[0])
    if target_num > 0:
        indices = np.random.choice(pts_world.shape[0], target_num, replace=False)
        pts_world = pts_world[indices]
        colors = rgb_valid[indices]

    out_ply = os.path.join(target_dir, "init_MET_4DGS.ply")
    with open(out_ply, 'wb') as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(pts_world)}\n".encode('utf-8'))
        f.write(
            b"property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(pts_world)):
            p = pts_world[i]
            c = colors[i]
            f.write(struct.pack('<ffffffBBB', p[0], p[1], p[2], 0.0, 0.0, 0.0, c[0], c[1], c[2]))

    print(f"🎉 混沌物理先验注入成功！点云规模暴涨至: {len(pts_world)} 个！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    generate_met_pointcloud(parser.parse_args().target)