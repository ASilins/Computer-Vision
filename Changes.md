# CenterPoint (Optimized Fork): FastPillars, ResNet & Env-Driven Deployment

This repository is a heavily optimized fork of the official [CenterPoint](https://github.com/tianweiy/CenterPoint) 3D object detection framework. 

While the original CenterPoint established SOTA accuracy using 3D Sparse Convolutions, it is prohibitively expensive for resource-constrained edge devices (e.g., 12GB VRAM GPUs). This fork introduces critical architectural modernizations, PyTorch 2.0 backend optimizations, and a new **Environment Variable (`.env`) deployment system** to enable lightweight, high-performance training and inference.

## 🚀 Key Architectural Upgrades

* **FastPillars (MAPE) Encoder:** Replaced standard PointPillars encoding with Max-and-Attention Pillar Encoding. This preserves critical 3D geometric details during the 2D pseudo-image projection phase.
* **Deep Residual Neck (ResNet-18):** Replaced the shallow convolutional neck with a deep `PillarRes18BackBone8x` to increase the receptive field, drastically improving the detection of elongated vehicles (e.g., buses, trailers) and small objects (bicycles).

## ⚙️ The `.env` Configuration System (New)

To simplify deployment and avoid hardcoding paths or GPU IDs across multiple scripts, this repository has been updated to support global parameter management via a `.env` file. 

Create a `.env` file in the root directory to define your training and hardware constraints globally:

```env

# Hardware Specifics
GPU_SAMPLES=2
GPU_WORKERS=4
SELECTED_GPUS=1

# Training Settings
TOTAL_EPOCHS=20
```
