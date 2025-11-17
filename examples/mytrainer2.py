
'''
	CUDA_HOME=/usr/local/cuda-11.8/;CUDA_VISIBLE_DEVICES=3 python mytrainer.py \
		--data_dir="/home/lh/all_datasets/RobustScene/on-the-go/spot/" \
		--data_factor 8 \
		--result_dir="/home/lh/SpotLessSplats/examples/results_mode1/spot" \
		--loss_type robust --semantics --no-cluster --train_keyword "clutter" --test_keyword "extra" --abmodel 1 --cgr
'''
import json
import math
import os
import time
from dataclasses import dataclass, field
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
import viser
import nerfview
from datasets.colmap import Dataset, Parser, ClutterDataset, SemanticParser
from datasets.traj import generate_interpolated_path, get_ordered_poses
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from scipy.stats import gamma
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    get_positional_encodings,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
    SpotLessModule,
    BayesModule,
    ssimv1,
)

from gsplat.rendering import rasterization


def plot_gmm_histogram(hist, means, covariances, weights):
    # 绘制直方图
    plt.bar(range(len(hist)), hist.numpy(), alpha=0.6, color='gray', label='Histogram')

    # 绘制拟合的高斯分布
    x_vals = np.linspace(0, len(hist) - 1, 1000)
    y_vals = np.zeros_like(x_vals)
    for i in range(len(means)):
        y_vals += weights[i] * (1 / np.sqrt(2 * np.pi * covariances[i])) * np.exp(
            -0.5 * ((x_vals - means[i]) ** 2 / covariances[i]))

    plt.plot(x_vals, y_vals, color='red', label='Fitted Gaussian Mixture')

    plt.title('Histogram with GMM Fit')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# TODO 这里要找到边缘部分并抑制这部分的强度，边缘的误差是不可避免的，因为即使没有任何遮挡，误差也会在边缘累积
def otsu_threshold(hist):
    # 归一化直方图以获得概率分布
    hist = hist.float()
    total_pixels = hist.sum()
    hist = hist / total_pixels
    # 计算灰度级索引
    levels = torch.arange(hist.size(0)).float()

    # 累积分布函数和累积平均值
    cumulative_prob = torch.cumsum(hist, dim=0)
    cumulative_mean = torch.cumsum(hist * levels, dim=0)
    # 全局平均值
    global_mean = cumulative_mean[-1]
    # 类间方差公式
    inter_class_variance = (global_mean * cumulative_prob - cumulative_mean) ** 2 / (
            cumulative_prob * (1 - cumulative_prob) + 1e-8
    )
    # 找到最大类间方差对应的阈值
    max_variance, threshold = torch.max(inter_class_variance[:-1], dim=0)  # 忽略最后一个值

    # 计算前景和背景的占比
    background_ratio = cumulative_prob[threshold]  # 背景累积概率
    # 前景和背景范围
    background_hist = hist[:threshold + 1]
    # 计算背景峰值位置
    background_peak = torch.argmax(background_hist).item()

    # 计算每类中心位置
    class_0_center = (torch.arange(threshold) * hist[:threshold]).sum() / hist[:threshold].sum()
    class_1_center = (torch.arange(threshold, len(hist)) * hist[threshold:]).sum() / hist[threshold:].sum()

    threshold = (threshold).float() / 1000.0  # 中值
    background_c = float(class_0_center) / 1000.0  # 背景中值
    foreground_c = float(class_1_center) / 1000.0  # 前景峰值
    background_peak = float(background_peak) / 1000.0  # 背景峰值
    # 计算前景部分的均值和方差少量

    return threshold.item(), background_c, foreground_c, background_peak


def get_wlmask(rgb_pred):
    rgb_pred = rgb_pred.numpy()[0] / 255.0
    scale = (rgb_pred.shape[0] / 500) * (rgb_pred.shape[1] / 500)
    rgb_pred_gray = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(rgb_pred_gray, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(rgb_pred_gray, cv2.CV_64F, 0, 1, ksize=7)
    # 只统计梯度明显区域的方向
    val = np.sqrt(sobelx * sobelx + sobely * sobely)

    masku = (val > 50).astype(np.float64)
    maskus = cv2.blur(masku, (int(7 * scale), int(7 * scale)))
    maskus = torch.from_numpy(maskus)
    '''
    masku = val.astype(np.float64).clip(0,50)
    maskus2 = cv2.blur(masku, (int(7*scale), int(7*scale)))/50.0
    maskus2 = torch.from_numpy(maskus2)
    '''
    return maskus

gaussian_blur = transforms.GaussianBlur(kernel_size=9, sigma=(3.0, 3.0))

def get_wlmask_pytorch(rgb_pred):
    # 确保输入在 GPU 上
    rgb_pred = rgb_pred[0] / 255.0  # 归一化

    # 计算缩放比例
    scale = (rgb_pred.shape[0] / 500) * (rgb_pred.shape[1] / 500)

    # 转换为灰度图 (0.299 * R + 0.587 * G + 0.114 * B)
    rgb_pred_gray = 0.299 * rgb_pred[:,:,0] + 0.587 * rgb_pred[:,:,1] + 0.114 * rgb_pred[:,:,2]
    rgb_pred_gray=rgb_pred_gray.unsqueeze(0).unsqueeze(0)
    # 定义 7x7 Sobel 核
    sobel_x = torch.tensor([
        [-2, -1, 0, 1, 2],
        [-3, -2, 0, 2, 3],
        [-4, -3, 0, 3, 4],
        [-3, -2, 0, 2, 3],
        [-2, -1, 0, 1, 2],
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(rgb_pred.device)  # (1, 1, 7, 7)

    sobel_y = sobel_x.transpose(2, 3)  # 转置为 y 方向的 Sobel 核

    # 计算 Sobel 梯度
    # 使用 Sobel 核计算梯度
    padded_tensor = F.pad(rgb_pred_gray, pad=(2,2, 2, 2), mode='reflect')
    grad_x = F.conv2d(padded_tensor, sobel_x)
    grad_y = F.conv2d(padded_tensor, sobel_y)
    # 梯度幅值
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # 创建掩码
    mask = (gradient_magnitude).float().clip(0,5)/5.0
    blurred_mask = gaussian_blur(mask)[0,0]
    #plt.imshow(blurred_mask.detach().cpu().numpy())
    #plt.show()
    return blurred_mask

# 这个计算基于深度引导的平滑正则
def get_wlmaskd(depth):
    rgb_pred = depth.numpy()[0]
    scale = (rgb_pred.shape[0] / 500) * (rgb_pred.shape[1] / 500)
    rgb_pred_gray = rgb_pred

    sobelx = cv2.Sobel(rgb_pred_gray, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(rgb_pred_gray, cv2.CV_64F, 0, 1, ksize=7)
    # 只统计梯度明显区域的方向
    val = np.sqrt(sobelx * sobelx + sobely * sobely)
    masku = (val > 50).astype(np.float64)
    maskus = cv2.blur(masku, (int(7 * scale), int(7 * scale)))
    maskus = torch.from_numpy(maskus)
    return maskus


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 8
    # Normalize the axis and world view
    normalize: bool = True
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Train and test image name keywords
    train_keyword: str = "clutter"
    test_keyword: str = "extra"
    # Enable semantic feature based training
    semantics: bool = True
    # Enable clustering of semantic features
    cluster: bool = False
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.0
    # Loss types: l1, robust
    loss_type: str = "robust"
    # Robust loss percentile for threshold
    robust_percentile: float = 0.7
    # enable alpha scheduling
    schedule: bool = True
    # alpha sampling schedule rate (higher more robust)
    schedule_beta: float = -3e-3
    # Thresholds for mlp mask supervision
    lower_bound: float = 0.4
    upper_bound: float = 0.6
    # bin size for the error hist for robust threshold
    bin_size: int = 1000

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 1.0

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 300000
    # Refine GSs every this steps
    refine_every: int = 100
    # Reset SH specular coefficients once
    reset_sh: int = 8002
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Use utilization-based pruning (UBP) for compression: xection 4.2.3 https://arxiv.org/pdf/2406.20055
    ubp: bool = False

    cgr: bool = False
    ifshow: bool = False
    # Threshold for UBP
    ubp_thresh: float = 1e-14
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    abmodel: int = 0

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)


def create_splats_with_optimizers(
        parser: Parser,
        init_type: str = "sfm",
        init_num_pts: int = 100_000,
        init_extent: float = 3.0,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        batch_size: int = 1,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, torch.optim.Optimizer]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    N = points.shape[0]
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = [
        (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size), "name": name}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    ]
    return splats, optimizers


def load_feat_sam(path, feat_rate, file_id):
    feat_path = path + '/' + f'features_sam' + '/' + file_id + '.npy'
    feat = np.load(feat_path)
    return feat


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"
        self.impath = 'lala'
        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.meanbad = 0
        self.meangood = 0
        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders" + str(cfg.lower_bound) + '_' + str(cfg.upper_bound)
        os.makedirs(self.render_dir, exist_ok=True)
        self.masks_dir = f"{cfg.data_dir}Moving_masks"
        os.makedirs(self.masks_dir, exist_ok=True)
        self.txt_path = f"{cfg.result_dir}/results.txt"
        if os.path.exists(self.txt_path):
            print('txt exits')
        else:
            with open(self.txt_path, "w") as file:
                file.write("Start!" + cfg.result_dir + "\n")
        # 初始化梯度算子
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()  # 转到GPU
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()  # 转到GPU

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.semantics:
            self.parser = SemanticParser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize,
                load_keyword=cfg.train_keyword,
                cluster=cfg.cluster,
            )
        else:
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize,
                test_every=cfg.test_every,
            )
        self.trainset = ClutterDataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            train_keyword=cfg.train_keyword,
            test_keyword=cfg.test_keyword,
            semantics=cfg.semantics,
        )
        self.valset = ClutterDataset(
            self.parser,
            split="test",
            train_keyword=cfg.train_keyword,
            test_keyword=cfg.test_keyword,
            semantics=False,
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
        )
        print("Model initialized. Number of GS:", len(self.splats["means3d"]))

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
        self.spotless_optimizers = []
        self.mlp_spotless = cfg.semantics and not cfg.cluster
        if self.mlp_spotless:
            # currently using positional encoding of order 20 (4*20 = 80)
            self.spotless_module = SpotLessModule(
                num_classes=1, num_features=self.trainset[0]["Dinofeatures"].shape[2] + 81
            ).cuda()
            self.Bayes_module = BayesModule(
                num_features=self.trainset[0]["sam_features"].shape[1] + self.trainset[0]["Dinofeatures"].shape[2] + 81
            ).cuda()

            # todo 设置优化器
            self.spotless_optimizers = [
                torch.optim.Adam(
                    self.spotless_module.parameters(),
                    lr=1e-3,
                )
            ]
            self.Bayes_optimizers = [
                torch.optim.Adam(
                    self.Bayes_module.parameters(),
                    lr=1e-3,
                )
            ]

            self.spotless_loss = lambda p, minimum, maximum: torch.mean(
                torch.nn.ReLU()(p - minimum) + torch.nn.ReLU()(maximum - p)
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means3d"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "radii": torch.zeros(n_gauss, device=self.device),
            "gcr": torch.zeros(n_gauss, device=self.device),  # 用来记录观测频率
            "mean": torch.zeros((n_gauss, 3), device=self.device),  # 用来记录位姿均值
            "fc": torch.zeros((n_gauss, 3), device=self.device),  # 用来记录位姿方差
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
            "isfar": torch.zeros(n_gauss, device=self.device, dtype=torch.bool),
            "numkit": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
            "hist_err": torch.zeros((cfg.bin_size,)),
            "avg_err": 1.0,
            "lower_err": 0.0,
            "upper_err": 1.0,
            "sqrgrad": torch.zeros(n_gauss, device=self.device),
        }

    def compute_depth_loss(self, image):
        """
        计算输入图像在 x 和 y 方向上的梯度
        :param image: 输入图像, Tensor, 形状 [B, C, H, W]，需在 GPU 上
        :return: dx, dy, 分别为 x 和 y 方向的梯度
        """
        # 确保图像为单通道（灰度图）或者多通道的 [B, 1, H, W]
        if image.ndim == 3:
            image = image.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        # 在 x 方向计算梯度
        grad_x = F.conv2d(image, self.sobel_x, padding=1, groups=image.shape[1])
        # 在 y 方向计算梯度
        grad_y = F.conv2d(image, self.sobel_y, padding=1, groups=image.shape[1])
        grad_loss = torch.abs(grad_x) + torch.abs(grad_x)
        return grad_loss

    def rasterize_cgr(  # 这个是专门用来渲染观测完备性
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # numkitu = self.running_stats['numkit'] > 0
        means = self.splats["means3d"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        gcrTC = self.running_stats['gcr'].unsqueeze(-1)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=gcrTC,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            ubp=False,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        return render_colors

    def rasterize_splats(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means3d"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            ubp=False,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
        )
        # alphashow = render_alphas[0,:,:,0].detach().cpu().numpy()
        # plt.imshow(alphashow)
        # plt.show()

        return render_colors, render_alphas, info

    # 总之就是找到内点
    def robust_mask(
            self, error_per_pixel: torch.Tensor, loss_threshold: float
    ) -> torch.Tensor:
        epsilon = 1e-3
        # error_per_pixel = error_per_pixel.norm(axis=-1, keepdims=True)#首先对三个通道求均值
        error_per_pixel = error_per_pixel.squeeze(-1).unsqueeze(0)
        is_inlier_pixel = (error_per_pixel < loss_threshold).float()  # 如果残差很小，那就是内点
        window_size = 3
        channel = 1
        window = torch.ones((1, 1, window_size, window_size), dtype=torch.float) / (
                window_size * window_size
        )
        if error_per_pixel.is_cuda:
            window = window.cuda(error_per_pixel.get_device())
        window = window.type_as(error_per_pixel)
        has_inlier_neighbors = F.conv2d(
            is_inlier_pixel, window, padding=window_size // 2, groups=channel
        )
        has_inlier_neighbors = (has_inlier_neighbors > 0.5).float()  # 用一个3*3的窗口膨胀一下
        is_inlier_pixel = ((has_inlier_neighbors + is_inlier_pixel) > epsilon).float()
        pred_mask = is_inlier_pixel.squeeze(0).unsqueeze(-1)
        return pred_mask

    def robust_cluster_mask(self, inlier_sf, semantics):
        inlier_sf = inlier_sf.squeeze(-1).unsqueeze(0)
        cluster_size = torch.sum(
            semantics, axis=[-1, -2], keepdims=True, dtype=torch.float
        )
        inlier_cluster_size = torch.sum(
            inlier_sf * semantics, axis=[-1, -2], keepdims=True, dtype=torch.float
        )
        cluster_inlier_percentage = (inlier_cluster_size / cluster_size).float()
        is_inlier_cluster = (cluster_inlier_percentage > 0.5).float()
        inlier_sf = torch.sum(
            semantics * is_inlier_cluster, axis=1, keepdims=True, dtype=torch.float
        )
        pred_mask = inlier_sf.squeeze(0).unsqueeze(-1)
        return pred_mask

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.json", "w") as f:
            json.dump(vars(cfg), f)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        train_len1 = int(self.trainset.indices.__len__())

        max_steps = train_len1*150
        if max_steps>30000:
            max_steps = 30000
        init_step = 0
        cfg.eval_steps.append(max_steps)
        cfg.save_steps.append(max_steps)
        schedulers = [
            # means3d has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers[0], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )




        pbar = tqdm.tqdm(range(init_step, max_steps), ncols=120)
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            ConfSFM = data['Conf_SFM'].to(device)
            BlockSAM = data['Block_SAM'].to(device)
            self.impath = data['image_path']
            self.impathid = data['image_id']
            self.RT = data["camtoworld"]
            # 对于正向块我TM直接学习！，反向块只准在非正向块中学
            # 块填充正则化后面再加！
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)
            bymask = get_wlmask_pytorch(data["image"].to(device))

            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+ED",
            )

            render_cgr = self.rasterize_cgr(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=None,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )

            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            info["means2d"].retain_grad()  # used for running stats
            train_len = int(self.trainset.indices.__len__())
            rgb_pred_mask = None
            zid = image_ids / train_len

            # em没错，我他妈的还要把这个观测不确定性显示出来。

            # loss
            if cfg.loss_type == "l1":
                rgbloss = F.l1_loss(colors, pixels)
                Bayesloss = 0
                Bayesloss_all = 0
            else:
                # todo robust loss，这里计算残差损失,话说回来如果使用贝叶斯置信度的话，就只能通过二值化掩膜间接影响，还有就是删除的时候其实应该所有点都参与
                # todo 第一个共识是，这个贝叶斯置信度这个鬼玩意不能直接干扰分裂，极端的前置系数会导致梯度爆炸
                #
                error_per_pixel = torch.abs(colors - pixels)
                # 简单根据这个大津法做一个二值化输入进去
                if cfg.semantics:
                    dinof = data["Dinofeatures"].to(device).permute((0, 3, 1, 2))
                    sf = data["sam_features"].to(device)[0]
                    # sf = data["semantics"].to(device)
                    error_per_pixel_en = pixels.clone().detach()[0].permute(2, 0, 1)
                    if cfg.cluster:
                        # cluster the semantic feature and mask based on cluster voting
                        sf = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="nearest",
                        )(sf).squeeze(0)
                        pred_mask = self.robust_cluster_mask(pred_mask, semantics=sf)
                    else:
                        # use spotless mlp to predict the mask
                        sf = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="bilinear",
                        )(sf).squeeze(0)

                        pos_enc = get_positional_encodings(  # 搞一个位置编码
                            colors.shape[1], colors.shape[2], 20
                        ).permute((2, 0, 1))
                        sf = torch.cat([sf, pos_enc], dim=0)

                        dinof = nn.Upsample(
                            size=(colors.shape[1], colors.shape[2]),
                            mode="bilinear",
                        )(dinof).squeeze(0)
                        zenc = zid * torch.ones_like(pos_enc[0:1, :, :])
                        sfu = torch.cat([dinof, sf, zenc], dim=0).permute(1, 2, 0)
                        dinof = torch.cat([dinof, pos_enc], dim=0)
                        dinof = dinof.permute(1, 2, 0)

                        self.Bayes_module.train()
                        Bayes_up = self.Bayes_module(sfu)
                        # 贝叶斯置信度
                        confGS = 2 * nn.Sigmoid()(0.2 * Bayes_up).unsqueeze(0)

                        dinofb = torch.cat([dinof, confGS[0].detach() * 0.5], dim=-1)
                        self.spotless_module.eval()
                        pred_mask_up = self.spotless_module(dinofb)
                        pred_mask = pred_mask_up.clone().detach().reshape(
                            1, colors.shape[1], colors.shape[2], 1
                        )
                        #合理的一点是距离越远的越应该，抑制复杂边缘
                        render_cgrused = render_cgr[0, :, :, 0]
                        render_cgrused = (0.3-render_cgrused.clip(0,0.3))*bymask*0
                        qzmask_pixel = (1 - render_cgrused * 2.5).unsqueeze(0).unsqueeze(-1).float()
                        qzmask_conf = (1 - render_cgrused * 3.0).unsqueeze(0).unsqueeze(-1).float()
                        pixsline = (error_per_pixel.detach().norm(dim=-1,keepdim=True) * qzmask_pixel * 0.4 + qzmask_conf * confGS.detach() * 0.5 * 0.6)
                        if cfg.ifshow and step>6010:
                            render_cgrusedss = render_cgr[0, :, :, 0].detach().cpu().numpy()
                            plt.imshow(render_cgrusedss,cmap='gray')
                            plt.show()
                            pixelss = pixels[0].detach().cpu().numpy()
                            plt.imshow(pixelss)
                            plt.show()
                            pixslines = pixsline[0,:,:,0].detach().cpu().numpy()
                            plt.imshow(pixslines,cmap='jet')
                            plt.show()
                            bymasks = bymask.detach().cpu().numpy()
                            plt.imshow(bymasks,cmap='plasma')
                            plt.show()
                            #pixsliness1 = (error_per_pixel.detach().norm(dim=-1,keepdim=True))[0,:,:,0].detach().cpu().numpy()
                            #plt.imshow(pixsliness1,cmap='jet')
                            #plt.show()
                            #pixsliness2 = (confGS.detach())[0,:,:,0].detach().cpu().numpy()
                            #plt.imshow(pixsliness2,cmap='jet')
                            #plt.show()
                            pixslinea = (error_per_pixel.detach().norm(dim=-1,keepdim=True) * 0.5 +  confGS.detach() * 0.5*0.5)[0,:,:,0].detach().cpu().numpy()
                            plt.imshow(pixslinea, cmap='jet')
                            plt.show()
                            qzmask_confs =qzmask_conf[0,:,:,0].detach().cpu().numpy()
                            plt.imshow(qzmask_confs, cmap='gray')
                            plt.show()
                            print('lalala')
                        lower_mask = self.robust_mask(
                            pixsline, self.running_stats["lower_err"]
                        )
                        upper_mask = self.robust_mask(
                            pixsline, self.running_stats["upper_err"]
                        )

                # TODO --------------------------- 块强制平滑 ------------------------------------------------------------
                if step > train_len1*60:
                    Blocknum = BlockSAM.max()

                    for i in range(int(Blocknum)):
                        j = i + 1
                        mask_now = BlockSAM[0] == j
                        if mask_now.sum() > 10:
                            ifnotconf = ConfSFM[0][mask_now].mean() < 10  # 这个块还得是坏的

                            Outline_rate = (pred_mask[0, :, :, 0][mask_now] < 0.5).sum() / mask_now.sum()
                            if Outline_rate > 0.65 and ifnotconf:
                                pred_mask[0, :, :, 0][mask_now] = 0  # pred_mask[0, :, :, 0][mask_now].mean()

                log_pred_mask = pred_mask.clone().detach()
                if cfg.schedule:
                    # schedule sampling of the mask based on alpha
                    alpha = np.exp(cfg.schedule_beta * np.floor((1 + step) / 1.5))
                    pred_mask = torch.bernoulli(  # 伯努利分布二值化，一开始都学，后面逐渐相信网络结果
                        torch.clip(
                            alpha + (1 - alpha) * pred_mask.clone().detach(),
                            min=0.0,
                            max=1.0,
                        )
                    )
                rgbloss = (pred_mask.clone().detach() * error_per_pixel).mean()
                confGSPF = confGS * confGS
                Bayesloss_all = ((error_per_pixel.detach()) / (2 * confGSPF) + torch.log(confGS))
            ssimloss = 1.0 - self.ssim(pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2))  # ssim项不管吗？？？还真不管
            pred_maskssim = pred_mask.clone().detach()
            ssimlossv2 = ssimv1((pixels * pred_maskssim).permute(0, 3, 1, 2),
                                (colors * pred_maskssim).permute(0, 3, 1, 2), womean=True).permute(0, 2, 3, 1)
            ssimlossv2 = (ssimlossv2 * pred_maskssim).mean()
            loss = rgbloss * (0.7) + ssimlossv2 * 0.3
            lossconf = Bayesloss_all.mean()
            gradloss = torch.tensor([0])

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            loss.backward()
            lossconf.backward()
            if self.mlp_spotless:  # 这里是训练分类器
                self.spotless_module.train()
                # 强制学习好的部分
                # 如果块内黑的很多，那就强制全黑

                spot_loss = self.spotless_loss(
                    pred_mask_up.flatten(), upper_mask.flatten(), lower_mask.flatten()
                )  # 前面的教学0，后面的教学1
                # reg = 0.5 * self.spotless_module.get_regularizer()#这个是平滑正则化，有啥子用？？？
                spot_loss = spot_loss  # + reg
                spot_loss.backward()

            # Pass the error histogram for capturing error statistics
            # 把所有点的损失撒到0，1之间的一万个区间里，这个ERR是每个区间的计数
            pixsline = torch.abs(colors - pixels).detach().cpu()
            pixsline = torch.norm(pixsline, dim=-1,keepdim=True)
            confline= confGS.detach().cpu()
            pixsline = (pixsline*qzmask_pixel.cpu()*0.4+qzmask_conf.cpu()*confline*0.5*0.6)
            info["err"],binlist = torch.histogram(
                pixsline,
                bins=cfg.bin_size,
                range=(0.0, 1.0),
            )
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if 1:
                desc += f"depth loss={gradloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)
            # 记录参数
            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/rgbloss", rgbloss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means3d"]), step
                )
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            self.update_running_stats(info, c2w=camtoworlds)
            self.running_stats['isfar']=info['depths'][0]<info['depths'][0].mean()*0.2
            cfg.refine_every = train_len1
            cfg.refine_stop_iter = train_len1*75
            # update running stats for prunning & growing
            if step < cfg.refine_stop_iter:

                if step > cfg.refine_start_iter and step % cfg.refine_every == 0:
                    grads = self.running_stats["grad2d"] / self.running_stats[
                        "count"
                    ].clamp_min(1)

                    # grow GSs
                    is_grad_high = grads >= cfg.grow_grad2d
                    is_small = (
                            torch.exp(self.splats["scales"]).max(dim=-1).values
                            <= cfg.grow_scale3d * self.scene_scale
                    )

                    is_dupli = is_grad_high & is_small  # 是否复制？ 梯度又高，大小又小的高斯体
                    n_dupli = is_dupli.sum().item()
                    self.refine_duplicate(is_dupli)

                    is_split = is_grad_high & ~is_small  # 是否分裂？ 梯度又高，大小又大的高斯体
                    is_split = torch.cat(
                        [
                            is_split,
                            # new GSs added by duplication will not be split
                            torch.zeros(n_dupli, device=device, dtype=torch.bool),
                        ]
                    )
                    n_split = is_split.sum().item()
                    self.refine_split(is_split)
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # prune GSs 是否裁剪？ 梯度又高，大小又小的高斯体

                    is_prune = torch.sigmoid(self.splats["opacities"]) < cfg.prune_opa
                    '''

                    '''
                    # 删除那些观察次数小于三次，并且参与度过低的体素
                    # have_see = self.running_stats["numkit"] >100
                    if cfg.cgr:
                        is_too_big = (
                                torch.exp(self.splats["scales"]).max(dim=-1).values
                                > cfg.prune_scale3d * self.scene_scale
                        )
                        is_prune = is_prune | is_too_big
                        not_see = self.running_stats["count"] < 3
                        not_learn = self.running_stats["gcr"] < 0.1
                        is_prune = is_prune | ( not_see & not_learn & self.running_stats['isfar'])
                    # 在观测次数小于10次的点集合中，剔除这些点中位置梯度极高的部分

                    # todo 这里要加一个
                    n_prune = is_prune.sum().item()
                    self.refine_keep(~is_prune)
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(self.splats['means3d'])} GSs."
                    )

                    # reset running stats，重置计数器和梯度
                    self.running_stats["grad2d"].zero_()
                    self.running_stats["radii"].zero_()
                    self.running_stats["count"].zero_()
                    # self.running_stats["numkit"].zero_()
                if step % cfg.reset_every == 0 and cfg.loss_type != "robust":
                    self.reset_opa(cfg.prune_opa * 2.0)
                if step == cfg.reset_sh and cfg.loss_type == "robust":
                    self.reset_sh()
            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )
            # 欠约束的漂浮物，因为缺乏约束而产生的漂浮物
            # 参数改完了开始更改优化器
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:  # 这个玩意还带姿态优化的吗？？？
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.spotless_optimizers:  # 这个玩意你还他妈的单独拉出来跟新
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.Bayes_optimizers:  # todo 更新贝叶斯损失
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # TODO ------------------------------Save the mask image-----------------------------------

            if step > max_steps - train_len1 and cfg.semantics:
                st_interval = time.time()
                rgb_pred_mask = (
                    (log_pred_mask > 0.5).repeat(1, 1, 1, 3).clone().detach()
                ).clip(0,1)
                rgb_pred_mask_for_save = (
                    (log_pred_mask).repeat(1, 1, 1, 3).clone().detach()
                )
                confGSs = confGS.repeat(1, 1, 1, 3).detach()
                pixsline = torch.abs(colors - pixels).detach()
                pixsline = torch.norm(pixsline, dim=-1, keepdim=True)
                pixsline = (qzmask_conf * confGSs * 0.5 * 0.6 + qzmask_pixel * pixsline * 0.4)

                render_cgr = (render_cgr).repeat(1, 1, 1, 3).clone().detach().clip(0,1)
                canvas = (
                    torch.cat([pixels, rgb_pred_mask, pixsline, colors,render_cgr], dim=2)
                        .squeeze(0)
                        .cpu()
                        .detach()
                        .numpy()
                )
                imname = data['image_path'][0].split('/')[-1].split('.')[0]
                # imname = image_ids.cpu().detach().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{imname}.png",
                    (canvas * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    f"{self.masks_dir}/{imname}.png",
                    (rgb_pred_mask_for_save.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8),
                )
                global_tic += time.time() - st_interval

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024 ** 3
                stats = {
                    "mem": mem,
                    "ellapsed_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means3d"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                torch.save(
                    {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    },
                    f"{self.ckpt_dir}/ckpt_{step}.pt",
                )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                psnr, ssim, lpips = self.eval(step)
                with open(self.txt_path, "a") as file:
                    writein = 'STEP:' + str(step) + ',' + f"PSNR: {psnr:.3f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.3f} \n"
                    file.write(writein)

                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                        num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    # 把观测位姿均值和观测方差放在这个里面
    def update_running_stats(self, info: Dict, c2w=None):
        """Update running stats."""
        cfg = self.cfg

        # normalize grads to [-1, 1] screen space
        if cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()  # 当前的梯度，关于像素位置
        grads[..., 0] *= info["width"] / 2.0 * cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * cfg.batch_size

        self.running_stats["hist_err"] = (
            0.95 * self.running_stats["hist_err"] + info["err"]
        )
        thresholdOTUS, bOTUS, fOTUS, bpeak = otsu_threshold(self.running_stats["hist_err"])
        self.running_stats["lower_err"] = bOTUS*0.8+thresholdOTUS*0.2
        self.running_stats["upper_err"] = bOTUS*0.25+thresholdOTUS*0.75

        if cfg.packed:
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]TMD 看投影半径大于0的区域
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.running_stats["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.running_stats["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

            # 开始计算位姿方差
            sel2 = grads.norm(dim=-1) > 0.0000000001
            updatemask = (sel & sel2)[0]
            firstupdate_mask = self.running_stats["numkit"] == 0  # 第一次被记录的点
            firstupdate_mask = firstupdate_mask & updatemask
            notfrist_mask = updatemask ^ firstupdate_mask  # 非第一次被记录的点
            beta_np = (1 / self.running_stats["numkit"]).unsqueeze(-1).clone().detach()

            # [C, N]TMD 看投影半径大于0的区域
            gs_idks = torch.where(sel2)[1]
            self.running_stats["numkit"].index_add_(
                0, gs_idks, torch.ones_like(gs_idks).int()
            )
            beta_np1 = (1 / self.running_stats["numkit"]).unsqueeze(-1).clone().detach()
            if c2w != None:
                T_now = c2w[0, :3, 3].unsqueeze(0)
                # 第一次被记录的点
                self.running_stats["mean"][firstupdate_mask] = T_now[0]
                self.running_stats["fc"][firstupdate_mask] = 0
                # 非第一次被记录的点
                if notfrist_mask.sum() > 0:
                    last_mean = self.running_stats["mean"][notfrist_mask].clone().detach()
                    self.running_stats["mean"][notfrist_mask] = last_mean + beta_np1[notfrist_mask] * (
                                T_now - last_mean)
                    self.running_stats["fc"][notfrist_mask] = (1 - beta_np[notfrist_mask]) * self.running_stats["fc"][
                        notfrist_mask] + beta_np1[notfrist_mask] * (T_now - last_mean) * (T_now - last_mean)

            self.running_stats["radii"] = info["radii"][0]

            gu = self.running_stats["fc"].norm(dim=-1)
            depthwonan = torch.nan_to_num(info["depths"][0], nan=1)

            gu[~sel2[0]] = 0
            gc = gu  #/(depthwonan+depthwonan.mean()*2)
            gc = torch.nan_to_num(gc, nan=0.0)
            gc = gc.clip(0, 1)
            self.running_stats["gcr"] = self.running_stats["gcr"] * 0.99 + gc * 0.01  # 类似于100次左右的均值


    @torch.no_grad()
    def reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.splats["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def reset_sh(self, value: float = 0.001):
        """Utility function to reset SH specular coefficients."""
        colors = torch.clamp(
            self.splats["shN"], max=torch.abs(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "shN":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(colors)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[param_group["name"]] = p_new
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        """Utility function to grow GSs."""
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.splats["scales"][sel])  # [N, 3]
        quats = F.normalize(self.splats["quats"][sel], dim=-1)  # [N, 4]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if v is None or k.find("err") != -1:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            if k == "sqrgrad":
                v_new = torch.ones_like(
                    v_new
                )  # the new ones are assumed to have high utilization in the start
            self.running_stats[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
        """Unility function to duplicate GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            if k == "sqrgrad":
                self.running_stats[k] = torch.cat(
                    (v, torch.ones_like(v[sel]))
                )  # new ones are assumed to have high utilization
            else:
                self.running_stats[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            if k.find("err") != -1:
                continue
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{self.render_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
            )

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means3d'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means3d"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()
        return psnr.item(), ssim.item(), lpips.item()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        camtoworlds = get_ordered_poses(self.parser.camtoworlds)

        camtoworlds = generate_interpolated_path(
            camtoworlds[::20].copy(), 40, spline_degree=1, smoothness=0.3
        )  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]
        camtoworlds = camtoworlds * np.reshape([1.1, 1.1, 1, 1], (1, 4, 1))

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i: i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=0 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.gif", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.gif")

    @torch.no_grad()
    def _viewer_render_fn(
            self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
