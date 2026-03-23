import os, sys, subprocess, argparse, math, cv2, torch, numpy as np, copy

GS_PYTHON = r"D:\Anaconda3\envs\met_4dgs\python.exe"
GS_CODE_DIR = r"D:\CPP\4DGaussians"
CUDA_BIN_PATH = r"D:\NVIDIA\V12.6\bin"

if sys.executable.lower() != GS_PYTHON.lower():
    subprocess.run([GS_PYTHON, "-u", os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(0)

try:
    if os.path.exists(CUDA_BIN_PATH): os.add_dll_directory(CUDA_BIN_PATH)
    sys.path.append(GS_CODE_DIR);
    os.chdir(GS_CODE_DIR)
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
except Exception as e:
    sys.exit(1)


class InteractiveCamWrapper:
    def __init__(self, base_cam):
        self.cam = copy.deepcopy(base_cam)
        self.orig_w2c = self.cam.world_view_transform.detach().cpu().numpy().T
        self.proj = self.cam.projection_matrix.detach().cpu().numpy().T
        self.reset()

    def reset(self):
        self.pan_x, self.pan_y, self.zoom_z = 0.0, 0.0, 0.0
        self.orbit_yaw, self.orbit_pitch = 0.0, 0.0
        # 🟢 终极居中对焦！因为点云的 BASE_DEPTH 设置在 1.8，这里我们将旋转中心死死锁在 1.8！
        self.pivot_dist = 1.8
        self.update()

    def pan(self, dx, dy):
        # 右键平移手感优化，方向符合直觉
        self.pan_x -= dx * 0.002;
        self.pan_y -= dy * 0.002;
        self.update()

    def orbit(self, dx, dy):
        # 左键旋转手感优化
        self.orbit_yaw += dx * 0.01;
        self.orbit_pitch -= dy * 0.01;
        self.orbit_pitch = max(-math.pi / 2, min(math.pi / 2, self.orbit_pitch));
        self.update()

    def zoom(self, dz):
        self.zoom_z -= dz * 0.2;
        self.update()

    def update(self):
        D = self.pivot_dist
        T_to_origin = np.eye(4, dtype=np.float32);
        T_to_origin[2, 3] = -D
        cy, sy = math.cos(self.orbit_yaw), math.sin(self.orbit_yaw)
        R_yaw = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=np.float32)
        cp, sp = math.cos(self.orbit_pitch), math.sin(self.orbit_pitch)
        R_pitch = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype=np.float32)
        T_from_origin = np.eye(4, dtype=np.float32);
        T_from_origin[2, 3] = D
        T_pan = np.eye(4, dtype=np.float32);
        T_pan[0, 3] = self.pan_x;
        T_pan[1, 3] = self.pan_y;
        T_pan[2, 3] = self.zoom_z

        M_local = T_pan @ T_from_origin @ R_pitch @ R_yaw @ T_to_origin
        w2c = M_local @ self.orig_w2c

        self.cam.world_view_transform = torch.tensor(w2c.T, dtype=torch.float32, device="cuda")
        self.cam.projection_matrix = torch.tensor(self.proj.T, dtype=torch.float32, device="cuda")
        self.cam.full_proj_transform = (
            self.cam.world_view_transform.unsqueeze(0).bmm(self.cam.projection_matrix.unsqueeze(0))).squeeze(0)
        self.cam.camera_center = self.cam.world_view_transform.inverse()[3, :3]


def get_real_model_path(p):
    p = os.path.normpath(p)
    if os.path.basename(p) == "point_cloud": return os.path.dirname(p)
    if "iteration_" in os.path.basename(p): return os.path.dirname(os.path.dirname(p))
    if os.path.exists(os.path.join(p, "FINAL_CLEAN_MODEL", "point_cloud")): return os.path.join(p, "FINAL_CLEAN_MODEL")
    return p


def get_real_source_path(model_path):
    s = model_path
    if "FINAL_CLEAN_MODEL" in s: s = os.path.dirname(s)
    if not os.path.exists(os.path.join(s, "transforms_train.json")): s = os.path.dirname(s)
    return s


def interactive_viewer(raw_model_path):
    print("⏳ 正在读取 4D 神经网络配置并载入显存...")
    model_path = get_real_model_path(raw_model_path)
    source_path = get_real_source_path(model_path)

    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    args = parser.parse_args([])
    args.model_path = model_path
    args.source_path = source_path
    args.sh_degree = 0

    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            from argparse import Namespace
            saved_args = eval(f.read(), {'Namespace': Namespace, 'torch': torch, 'np': np})
            for k, v in vars(saved_args).items():
                if k not in ["source_path", "model_path", "sh_degree"]: setattr(args, k, v)

    gaussians = GaussianModel(args.sh_degree, hp.extract(args))
    scene = Scene(lp.extract(args), gaussians, load_iteration=-1, shuffle=False)

    train_cams = scene.getTrainCameras()
    wrapper = InteractiveCamWrapper(train_cams[0])

    win_name = "MET-4DGS Viewer (Orbital Native POV)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 800)

    mouse_state = {"L_drag": False, "R_drag": False, "x": 0, "y": 0}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state.update({"L_drag": True, "x": x, "y": y})
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state["L_drag"] = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse_state.update({"R_drag": True, "x": x, "y": y})
        elif event == cv2.EVENT_RBUTTONUP:
            mouse_state["R_drag"] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_state["L_drag"]:
                wrapper.orbit(x - mouse_state["x"], y - mouse_state["y"])
                mouse_state.update({"x": x, "y": y})
            elif mouse_state["R_drag"]:
                wrapper.pan(x - mouse_state["x"], y - mouse_state["y"])
                mouse_state.update({"x": x, "y": y})
        elif event == cv2.EVENT_MOUSEWHEEL:
            wrapper.zoom(-1.0 if flags > 0 else 1.0)

    cv2.setMouseCallback(win_name, mouse_cb)

    def update_time(val):
        t_val = val / 100.0
        # 🟢 强行写入所有 4D 变量，保证时间轴必生效
        wrapper.cam.time = t_val
        wrapper.cam.timestamp = t_val
        wrapper.cam.fid = t_val
        wrapper.cam.sys_time = t_val
        if hasattr(wrapper.cam, 'time_tensor'):
            wrapper.cam.time_tensor = torch.tensor([t_val], dtype=torch.float32, device="cuda")

    cv2.createTrackbar("Time (t)", win_name, 0, 100, update_time)

    print("\n✅ 载入完成！模型已完美居中！")
    print("🖱️ [左键拖动]：3D旋转 | 🖱️ [右键拖动]：平移 | 🖱️ [滚轮]：缩放 | ⌨️ [R]：重置视角")

    with torch.no_grad():
        while cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1:
            img = (render(wrapper.cam, gaussians, pp.extract(args), torch.tensor([0.15, 0.15, 0.18], device="cuda"))[
                       "render"].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imshow(win_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key in (ord('r'), ord('R')):
                wrapper.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--model_path", required=True)
    interactive_viewer(parser.parse_args().model_path)