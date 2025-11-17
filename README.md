# 🌐 OCSplat — Robust 3D Gaussian Splatting with Clutter-Aware Reconstruction

OCSplat provides a complete pipeline for **robust 3D Gaussian Splatting reconstruction** in realistic outdoor and indoor scenes.  
It includes:

- COLMAP preprocessing  
- SAM-based rigid/clutter separation  
- Feature extraction  
- Robust Gaussian Splatting optimization  
- Visualization and evaluation tools  

This project integrates **gsplat**, **Segment Anything**, **pycolmap**, and several custom modules to provide stable reconstruction even under clutter, occlusion, or complex structures.

---

# 🚀 Installation Guide

Follow the steps below to set up your environment.

---

## 1️⃣ Conda Environment

```bash
conda create -n ocsplat python=3.10
conda activate ocsplat
