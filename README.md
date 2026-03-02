# Thermo-EventGS (TE-GS): Monocular 4D Scene Reconstruction via Thermal-Event Fusion

![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry_Pi_4B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20OpenCV-orange)

> **TE-GS** is an ultra-low-cost, highly robust monocular 4D dynamic scene reconstruction system designed for extreme low-light environments. 

By bypassing expensive DVS sensors and high-overhead APIs, this project leverages native Linux V4L2 drivers to achieve zero-loss 120FPS video streaming on edge devices (Raspberry Pi 4B). We simulate high-frequency event streams via log-intensity variations and tightly couple them with thermal imaging pipelines to constrain 4D Gaussian Splatting / HexPlane representations.

## 🔬 Core Innovations (Novelty)

Unlike traditional 4DGS that relies on perfect RGB lighting, our system tackles extreme dark environments by introducing a novel **Asynchronous Spatio-Temporal Decoupling** mechanism:

1. **Asynchronous Spatio-Temporal Decoupling in HexPlane:** Thermal cameras suffer from low framerates (30FPS) causing severe motion blur, while our simulated event stream captures microsecond edge movements at 120FPS. We decouple these modalities within the 6 orthogonal planes of the HexPlane architecture:
   * **Spatial Planes (XY, YZ, ZX):** Primarily supervised by the 30Hz thermal data to establish the base temperature photometry and coarse volume geometry.
   * **Temporal Planes (XT, YT, ZT):** Strictly anchored by the 120Hz high-frequency event spikes, acting as a "temporal pacemaker" to eliminate thermal dynamic blurring.
2. **Event-Guided Thermal Sharpening Loss ($\mathcal{L}_{event}$):** Thermal imaging is notoriously textureless. We extract sharp physical edges using log-domain difference $L = \ln(I + 0.01)$. We introduce a novel contrast maximization loss that forces the 3D Gaussians to densely align and sharpen precisely where event spikes are triggered, effectively "sculpting" sharp boundaries out of blurry thermal blobs.
3. **Zero-Cost Edge Hardware-in-the-Loop:** We prove that complex event-thermal fusion does not require $10,000+ DVS hardware. We achieved it entirely on a Raspberry Pi using a global shutter camera (OV9281) and custom C++ V4L2 drivers.

## 🏗️ System Architecture & Code Structure

The project is divided into Edge Acquisition (C++/Python) and Cloud Reconstruction (CUDA/PyTorch).

```text
├── algorithms/
│   ├── alignment.py       # Affine transformation & multi-modal spatial registration
│   ├── event_sim.py       # Log-intensity event simulation (VID2E approximation)
│   └── vignetting.py      # Polynomial flat-field correction
├── core/
│   ├── detector.py        # Kalman-filtered ROI tracking & morphological operations
│   └── sync_engine.py     # Multi-thread synchronization for 120FPS & 30FPS streams
├── hardware_drivers/      # Native V4L2 C++ implementation for OV9281
└── 4D_Reconstruction/     # Custom HexPlane / 4DGS pipeline (Active Development)
```

## 🚀 Quick Start (Edge Deployment)

### 1. Hardware Requirements
* **Compute:** Raspberry Pi 4B (4GB/8GB)
* **Sensors:** OV9281 (connected to USB 3.0), Tiny1-C Thermal (connected to USB 2.0)
* **Power:** 12V Battery with UBEC 5V/3A step-down module.

### 2. Environment Setup
```bash
# Install V4L2 and OpenCV dependencies
sudo apt-get install v4l-utils libopencv-dev python3-opencv
pip install numpy pyqt6
```

### 3. Run the Sensor Fusion Engine
Ensure both cameras are connected. The `sync_engine.py` will automatically handle the asynchronous multi-threading and buffer alignment.
```bash
python main.py
```

## 🤝 To-Do & Research Roadmap

- [x] Hardware prototyping and custom V4L2 driver development.
- [x] Edge-side real-time log-domain event simulation at 120FPS.
- [x] Thermal and visual spatial-temporal alignment.
- [ ] **Current Focus:** Resolving custom CUDA kernel compilation (e.g., `grid_encoder`) for the HexPlane representation.
- [ ] **Algorithm Update:** Implementing the $\mathcal{L}_{event}$ sharpening loss function in the PyTorch backend.

---
**Author:** Jiahao PEI  
**Contact:** 18638256716@163.com | [GitHub Link]