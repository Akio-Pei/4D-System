import os, sys, subprocess, argparse, math, cv2, torch, numpy as np

GS_PYTHON = r"D:\Anaconda3\envs\met_4dgs\python.exe"
GS_CODE_DIR = r"D:\CPP\4DGaussians"
CUDA_BIN_PATH = r"D:\NVIDIA\V12.6\bin"

if sys.executable.lower() != GS_PYTHON.lower():
    subprocess.run([GS_PYTHON, "-u", os.path.abspath(__file__)] + sys.argv[1:])
    sys.exit(0)

try:
    if os.path.exists(CUDA_BIN_PATH):
        os.add_dll_directory(CUDA_BIN_PATH)
        os.environ["PATH"] = CUDA_BIN_PATH + os.pathsep + os.environ.get("PATH", "")
    sys.path.append(GS_CODE_DIR);
    os.chdir(GS_CODE_DIR)
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from arguments import ModelParams, PipelineParams, OptimizationParams
except Exception as e:
    sys.exit(1)


# 🟢 终极形态：原生轨道相机
class OrbitalNativeCamera:
    def __init__(self, base_cam):
        self.image_width = base_cam.image_width
        self.image_height = base_cam.image_height
        self.FoVy = base_cam.FoVy
        self.FoVx = base_cam.FoVx
        self.znear = 0.001
        self.zfar = 10000.0  # 极大视野深度，防裁切
        self.time = 0.0

        # 抓取原生训练相机矩阵 (保证初始视角 100% 完美，绝不瞎跑)
        self.orig_w2c = base_cam.world_view_transform.detach().cpu().numpy().T
        self.proj = base_cam.projection_matrix.detach().cpu().numpy().T

        self.reset()

    def reset(self):
        self.pan_x, self.pan_y = 0.0, 0.0
        self.zoom_z = 0.0
        self.orbit_yaw = 0.0
        self.orbit_pitch = 0.0
        self.pivot_dist = 1.8  # 数据集默认相机距离模型 Z=1.8，这就是完美轴心！
        self.update()

    def pan(self, dx, dy):
        self.pan_x -= dx * 0.002
        self.pan_y += dy * 0.002
        self.update()

    def orbit(self, dx, dy):
        self.orbit_yaw -= dx * 0.01
        self.orbit_pitch -= dy * 0.01
        self.orbit_pitch = max(-math.pi / 2, min(math.pi / 2, self.orbit_pitch))
        self.update()

    def zoom(self, dz):
        self.zoom_z -= dz * 0.2
        self.update()

    def update(self):
        # 核心数学：在局部坐标系下构建轨道旋转矩阵，围绕距离相机 1.8m 的中心旋转
        D = self.pivot_dist

        T_to_origin = np.eye(4, dtype=np.float32)
        T_to_origin[2, 3] = -D

        cy, sy = math.cos(self.orbit_yaw), math.sin(self.orbit_yaw)
        R_yaw = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        cp, sp = math.cos(self.orbit_pitch), math.sin(self.orbit_pitch)
        R_pitch = np.array([
            [1, 0, 0, 0],
            [0, cp, -sp, 0],
            [0, sp, cp, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        T_from_origin = np.eye(4, dtype=np.float32)
        T_from_origin[2, 3] = D

        T_pan = np.eye(4, dtype=np.float32)
        T_pan[0, 3] = self.pan_x
        T_pan[1, 3] = self.pan_y
        T_pan[2, 3] = self.zoom_z

        # 矩阵连乘：平移 + 轨道推回 + 俯仰角 + 偏航角 + 轨道拉近
        M_local = T_pan @ T_from_origin @ R_pitch @ R_yaw @ T_to_origin
        w2c = M_local @ self.orig_w2c

        self.world_view_transform = torch.tensor(w2c.T, dtype=torch.float32, device="cuda")
        self.projection_matrix = torch.tensor(self.proj.T, dtype=torch.float32, device="cuda")
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def interactive_viewer(model_path):
    model_path = os.path.normpath(model_path)
    if "iteration_" in model_path or "point_cloud" in model_path: model_path = os.path.dirname(
        model_path) if "point_cloud" in model_path else os.path.dirname(os.path.dirname(model_path))
    if not os.path.exists(os.path.join(model_path, "point_cloud")):
        alt_path = os.path.join(model_path, "FINAL_CLEAN_MODEL")
        if os.path.exists(os.path.join(alt_path, "point_cloud")): model_path = alt_path

    parser = argparse.ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    args = parser.parse_args([]);
    args.model_path = model_path

    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            try:
                from argparse import Namespace;
                saved_args = eval(f.read(), {'Namespace': Namespace, 'torch': torch, 'np': np})
                for k, v in vars(saved_args).items(): setattr(args, k, v)
            except Exception:
                pass

    default_args = {"net_width": 64, "timebase_pe": 4, "grid_pe": 4, "multires": [1, 2], "defor_depth": 1,
                    "posebase_pe": 10, "scale_rotation_pe": 2, "opacity_pe": 2, "timenet_width": 64,
                    "timenet_output": 32, "bounds": 1.6, "no_dx": False, "no_grid": False, "no_ds": False,
                    "no_dr": False, "no_do": False, "no_dshs": False, "empty_voxel": False, "static_mlp": False,
                    "apply_rotation": False}
    for k, v in default_args.items():
        if not hasattr(args, k): setattr(args, k, v)
    if not hasattr(args, "kplanes_config"): setattr(args, "kplanes_config",
                                                    {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                                                     'output_coordinate_dim': 32, 'resolution': [64, 64, 64, 25]})

    gaussians = GaussianModel(args.sh_degree, args)
    scene = Scene(lp.extract(args), gaussians, load_iteration=-1, shuffle=False)

    train_cams = scene.getTrainCameras()
    if not train_cams:
        print("❌ 致命错误：找不到训练场景的相机！")
        return

    cam = OrbitalNativeCamera(train_cams[0])

    win_name = "MET-4DGS Viewer (Orbital Native POV)"
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL);
    cv2.resizeWindow(win_name, 800, 800)
    mouse_state = {"L_drag": False, "x": 0, "y": 0}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state.update({"L_drag": True, "x": x, "y": y})
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state["L_drag"] = False
        elif event == cv2.EVENT_MOUSEMOVE and mouse_state["L_drag"]:
            dx, dy = x - mouse_state["x"], y - mouse_state["y"]
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                cam.orbit(dx, dy)
            else:
                cam.pan(dx, dy)
            mouse_state.update({"x": x, "y": y})
        elif event == cv2.EVENT_MOUSEWHEEL:
            cam.zoom(-1.0 if flags > 0 else 1.0)

    cv2.setMouseCallback(win_name, mouse_cb)
    cv2.createTrackbar("Time (t)", win_name, 0, 100, lambda val: setattr(cam, 'time', val / 100.0))

    print("\n🚀 终极轨道视角已开启！")
    print("🖱️ [鼠标左键拖动]：画面平移 (Pan)")
    print("⌨️ [Shift + 左键拖动]：轨道 3D 旋转 (Orbit 围绕中心手势)")
    print("🖱️ [鼠标滚轮滚动]：推进/拉远 (Zoom)")
    print("⌨️ [R 键]：一键重置为初始完美视角")

    with torch.no_grad():
        while cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1:
            img = (render(cam, gaussians, pp.extract(args), torch.tensor([0.15, 0.15, 0.18], device="cuda"))[
                       "render"].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imshow(win_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key in (ord('r'), ord('R')):
                cam.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--model_path", required=True)
    interactive_viewer(parser.parse_args().model_path)