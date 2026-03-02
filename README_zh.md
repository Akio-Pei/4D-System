# Thermo-EventGS (TE-GS)：基于热红外与事件流融合的单目 4D 场景重建系统

![Status](https://img.shields.io/badge/状态-积极开发中-brightgreen)
![Hardware](https://img.shields.io/badge/硬件-树莓派_4B-blue)
![Framework](https://img.shields.io/badge/框架-PyTorch%20%7C%20OpenCV-orange)

> **TE-GS** 是一个专为极弱光/无纹理环境设计的超低成本、高鲁棒性单目 4D 动态场景重建系统。

通过绕开昂贵的事件相机 (DVS) 硬件和高开销的通用 API，本项目利用 Linux 原生的 V4L2 驱动，在边缘计算设备（树莓派 4B）上实现了 120FPS 无损视频流的极致拉取。我们利用对数光强变化实时模拟高频事件流，并将其与热成像管线深度耦合，从而为后端的 4D 高斯溅射 (4DGS) 与 HexPlane 表征提供强大的跨模态时空约束。

## 🔬 核心理论创新 (Novelty)

传统的 4DGS 极度依赖光照完美的 RGB 视频。为了攻克极暗环境，本项目提出了一种全新的**跨模态异步时空解耦机制**：

1. **基于 HexPlane 的异步时空解耦 (Asynchronous Spatio-Temporal Decoupling)：** 热红外相机的低帧率 (30FPS) 会在极速运动下产生严重拖影，而模拟事件流 (120FPS) 能够以微秒级精度捕捉边缘运动。我们在 HexPlane 的 6 个正交平面中对这两种异步信号进行了解耦融合：
   * **空间平面 (XY, YZ, ZX)：** 主要由 30Hz 热红外数据监督，为 3D 高斯球赋予基础的温度光度属性和粗糙的体积几何。
   * **时间平面 (XT, YT, ZT)：** 由 120Hz 的高频事件脉冲死死锚定，作为时间轴上的“高频节拍器”，彻底消除热红外的动态模糊。
2. **事件引导的热边缘锐化损失 ($\mathcal{L}_{event}$)：** 热成像缺乏高频纹理。我们利用对数域差分 $L = \ln(I + 0.01)$ 提取出极其敏锐的物理运动边缘。通过引入全新的对比度最大化损失函数，强迫高斯球在有“事件流”触发的区域密集收敛，用高频事件脉冲将模糊的热量团“雕刻”出极度锐利的物理边界。
3. **零成本的软硬件协同闭环：** 证明了复杂的事件-热红外融合无需昂贵的 DVS 硬件。仅依托树莓派、全局快门相机 (OV9281) 以及自研的 C++ V4L2 底层驱动，即可在极低功耗下完成边缘侧的闭环验证。

## 🏗️ 系统架构与代码结构

本项目分为“边缘端采集 (C++/Python)”与“云端重建 (CUDA/PyTorch)”两大部分：

```text
├── algorithms/
│   ├── alignment.py       # 仿射变换与多模态空间配准
│   ├── event_sim.py       # 对数光强事件模拟 (VID2E 近似实现)
│   └── vignetting.py      # 多项式暗角与平场校正
├── core/
│   ├── detector.py        # 基于卡尔曼滤波的 ROI 追踪与形态学处理
│   └── sync_engine.py     # 120FPS 与 30FPS 异步视频流的多线程同步引擎
├── hardware_drivers/      # 面向 OV9281 的原生 V4L2 C++ 驱动实现
└── 4D_Reconstruction/     # 自定义 HexPlane / 4DGS 训练管线 (研发中)
```

## 🚀 快速开始 (边缘端部署)

### 1. 硬件要求
* **计算平台：** 树莓派 4B (推荐 4GB/8GB 内存版本)
* **传感器：** OV9281 (接入 USB 3.0), Tiny1-C 热成像仪 (接入 USB 2.0)
* **供电系统：** 12V 航模电池 + UBEC 5V/3A 降压稳压模块

### 2. 环境配置
```bash
# 安装 V4L2 底层工具链与 OpenCV 依赖
sudo apt-get install v4l-utils libopencv-dev python3-opencv
pip install numpy pyqt6
```

### 3. 运行多模态融合引擎
确保两台相机均已正确物理连接。`sync_engine.py` 将自动接管底层异步多线程并完成时间戳对齐。
```bash
python main.py
```

## 🤝 待办事项与研究路线图 (To-Do)

- [x] 硬件原型设计与自定义 V4L2 底层驱动开发。
- [x] 边缘端 120FPS 实时对数域事件流模拟。
- [x] 热红外与可见光视频流的时空严格对齐 (卡尔曼滤波 + 形态学处理)。
- [ ] **当前攻坚：** 解决 HexPlane 表征中自定义 CUDA 算子（如 `grid_encoder`）的编译与环境冲突报错问题。
- [ ] **算法创新：** 在 PyTorch 后端实现基于事件引导的 $\mathcal{L}_{event}$ 边缘锐化损失函数。

---
**作者:** 裴家豪 (Jiahao PEI)  
**联系方式:** 18638256716@163.com | [GitHub 主页链接]