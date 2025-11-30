# 🌐 OCSplats: Observation Completeness Quantification and Label Noise Separation in 3DGS

# 🚀 Installation Guide

Follow the steps below to set up your environment.

[PAPER](https://openaccess.thecvf.com/content/ICCV2025/html/Ling_OCSplats_Observation_Completeness_Quantification_and_Label_Noise_Separation_in_3DGS_ICCV_2025_paper.html) 

[Data(on-the-go)](https://drive.google.com/file/d/1GA25m2l-y6-k3GhLpWVr_q_82_WcM-Gj/view?usp=sharing)

## 1 Conda Environment

```bash
conda create -n ocsplat python=3.10
conda activate ocsplat
```

## 2 Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## 3 Install OCSplat (local project)
Make sure you are inside the project directory:
```bash
python setup.py install
```

## 4 Install Required Dependencies
```bash
pip install jaxtyping ninja
pip install "numpy<2"
pip install "opencv-python<=4.8.1.78"
pip install imageio
pip install nerfview==0.0.2
pip install torchmetrics
pip install mediapy
pip install tensorboard
pip install scikit-learn
```

## 5 Install Segment Anything (SAM)
```bash
cd segment-anything
pip install -e .
cd ..
```
🔔 Note:
Download the SAM weights and update the file path in SAM_block.py at line 20.


## 6 Install pycolmap
```bash
cd pycolmap
python3 -m pip install -e .
cd ..
```


# 🛠️ Full Reconstruction Pipeline
Below is the complete OCSplat workflow, from COLMAP preprocessing to final robust Gaussian Splatting reconstruction.

## 1 COLMAP Sparse Reconstruction
Ensure that COLMAP is installed on your Ubuntu environment.
Then run:
```bash
bash local_colmap_and_resize.sh /path/to/dataset/
```
Your dataset folder must contain:
```bash
images/       # input RGB images
```
After reconstruction, COLMAP will produce:
```bash
sparse/0/
```

## 2 Feature Extraction & Rigid/Clutter Segmentation

Run the SAM-based preprocessing script:
```bash
python SAM_block.py
```
Before running, modify the dataset path at line 327 of SAM_block.py:
```bash
path = "/home/lh/all_datasets/RobustScene/testsocsplat/spot/"
```
The folder must include:
```bash
images/
sparse/0/
```

## 3 Robust Gaussian Splatting Reconstruction
Finally, run the main training script:
```bash
python examples/mytrainer2.py \
    --data_dir /home/lh/all_datasets/RobustScene/testsocsplat/spot/ \
    --data_factor 8 \
    --result_dir /home/lh/all_datasets/RobustScene/testsocsplat/spottest/ \
    --loss_type robust \
    --semantics \
    --no-cluster \
    --train_keyword "clutter" \
    --test_keyword "extra"
```

## 🔧 Parameter Description
Parameter	Description

--data_dir	Dataset folder containing images + COLMAP sparse/0

--result_dir	Output directory of reconstruction results

--data_factor	Downsample factor (default = 8 except patio dataset)

## 📌 On first run: gsplat will compile CUDA kernels, so initial startup may take extra time.


# 🙏 Acknowledgements
This project builds upon the excellent work of: https://github.com/lilygoli/SpotLessSplats

# 🙏 Citation
```bash
@inproceedings{ling2025ocsplats,
  title={OCSplats: Observation Completeness Quantification and Label Noise Separation in 3DGS},
  author={Ling, Han and Xu, Xian and Sun, Yinghui and Sun, Quansen},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={25680--25689},
  year={2025}
}
```
