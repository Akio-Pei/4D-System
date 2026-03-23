import os, cv2, json, numpy as np, argparse, struct


def generate_met_pointcloud(target_dir):
    print(f"🚀 [GS Init] 启动安全区过滤版的 3D 实心骨架生成...")
    with open(os.path.join(target_dir, "transforms_train.json"), 'r') as f:
        meta = json.load(f)

    W, H = meta['w'], meta['h']
    fl_x, fl_y = meta['fl_x'], meta['fl_y']
    cx, cy = W / 2.0, H / 2.0
    c2w = np.array(meta['frames'][0]['transform_matrix'])

    therm_dir = os.path.join(target_dir, "thermal")
    evt_dir = os.path.join(target_dir, "event_hq")
    raw_fname = sorted([f for f in os.listdir(therm_dir) if f.endswith(".png")])[0]

    img_t_color = cv2.imread(os.path.join(therm_dir, raw_fname))
    img_e = cv2.imread(os.path.join(evt_dir, raw_fname), 0)

    # 1. 🟢 提取最纯净的热力主体
    r_ch, b_ch = img_t_color[:, :, 2].astype(np.int16), img_t_color[:, :, 0].astype(np.int16)
    hot_mask = ((r_ch - b_ch > 20) & (r_ch > 100)).astype(np.uint8)

    cnts_t, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_hot_mask = np.zeros_like(hot_mask)
    if cnts_t:
        c_max_t = max(cnts_t, key=cv2.contourArea)
        cv2.drawContours(clean_hot_mask, [c_max_t], -1, 1, -1)

    # 2. 🟢 安全区过滤 Event
    safe_roi = cv2.dilate(clean_hot_mask, np.ones((25, 25), np.uint8), iterations=1)
    _, e_mask = cv2.threshold(img_e, 30, 1, cv2.THRESH_BINARY)
    clean_e_mask = cv2.bitwise_and(e_mask, safe_roi)

    # 3. 🟢 构造 3D 点云的参考颜色画布（彻底杜绝蓝底）
    color_coding_canvas = np.zeros_like(img_t_color)
    clean_hot_mask_3c = np.dstack([clean_hot_mask] * 3)
    color_coding_canvas = np.where(clean_hot_mask_3c == 1, img_t_color, color_coding_canvas)
    color_coding_canvas[clean_e_mask == 1] = [255, 255, 255]

    # 4. 🟢 生成纯净几何掩码
    e_dilated = cv2.dilate(clean_e_mask, np.ones((5, 5), np.uint8), iterations=1)
    fused_geometry = cv2.bitwise_or(clean_hot_mask, e_dilated)
    align_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fused_geometry = cv2.morphologyEx(fused_geometry, cv2.MORPH_CLOSE, align_kernel)

    # 5. 🟢 提取真实热力深度映射
    t_float = r_ch.astype(np.float32) * clean_hot_mask
    t_norm = np.zeros_like(t_float)
    if t_float.max() > 0: t_norm = t_float / 255.0

    valid_y, valid_x = np.where(fused_geometry > 0)

    # 6. 🟢 采样 50,000 颗完美对齐的种子
    target_num = 50000
    if len(valid_x) > 0:
        rand_idx = np.random.choice(len(valid_x), target_num, replace=True)
        u_base = valid_x[rand_idx].astype(np.float32)
        v_base = valid_y[rand_idx].astype(np.float32)
        z_norm_base = t_norm[valid_y[rand_idx], valid_x[rand_idx]]

        # 直接从纯净画布上提取颜色，完美一致！
        colors = color_coding_canvas[valid_y[rand_idx], valid_x[rand_idx]]

        u_chaos = u_base + np.random.uniform(-1.5, 1.5, target_num)
        v_chaos = v_base + np.random.uniform(-1.5, 1.5, target_num)

        BASE_DEPTH = 1.8
        RELIEF = 0.1
        THICKNESS = 0.08
        depth_noise = np.random.uniform(0, THICKNESS, target_num)
        Z_cam = BASE_DEPTH - (z_norm_base * RELIEF) + depth_noise

        X_cam = (u_chaos - cx) * Z_cam / fl_x
        Y_cam = (v_chaos - cy) * Z_cam / fl_y

        pts_camera = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(X_cam)], axis=1)
        pts_world = (c2w @ pts_camera.T).T[:, :3]
    else:
        pts_world, colors = np.zeros((0, 3)), np.zeros((0, 3))

    out_ply = os.path.join(target_dir, "init_MET_4DGS.ply")
    with open(out_ply, 'wb') as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(pts_world)}\n".encode('utf-8'))
        f.write(
            b"property float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(pts_world)):
            p, c = pts_world[i], colors[i]
            f.write(struct.pack('<ffffffBBB', p[0], p[1], p[2], 0.0, 0.0, 0.0, c[0], c[1], c[2]))

    print(f"🎉 防漏蓝、强对齐的完美 3D seeds 已注入: {len(pts_world)} 个！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--target", required=True)
    generate_met_pointcloud(parser.parse_args().target)