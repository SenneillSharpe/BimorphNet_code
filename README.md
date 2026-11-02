# BiMorphNet

**Bi-stream Cross-modal Morphology-aware Hierarchical Network for Thin-Structure Segmentation**

BiMorphNet 面向光学与 SAR 双模态遥感影像中的线性农田防护林（Linear Farmland Shelterbelts, LFS）二值语义分割，结合跨模态交互、通道依赖重标定、层级特征传递与细节引导重建，兼顾目标判别与细窄结构恢复。

本仓库当前提供网络实现、配置以及损失函数、评价指标和权重等辅助工具。当前版本不包含数据加载器、完整训练/测试入口、数据集或预训练权重。

## Model overview

| 组件 | 名称 | 代码入口 |
| --- | --- | --- |
| 双分支编码 | Independent optical/SAR residual encoders | `encoder.rgb_backbone`、`encoder.sar_backbone` |
| RCGI | Reciprocal Cross-modal Gated Interaction | `RCGI`；`encoder.low_rcgi`、`mid1_rcgi`、`mid2_rcgi`、`high_rcgi` |
| KDRM | Kernelized Dependency-Relevance Modulation | `Model/KDRM.py`；`decoder.kdrm` |
| HCFR | Hierarchical Cross-modal Feature Relay | `decoder.hcfr_relay`；`hcfr_fc1`—`hcfr_fc4` 及对应投影 |
| HDGR | Hierarchical Detail-Guided Reconstruction | `decoder.hdgr_reconstruct` |

HDGR 组合级联 FADE、同尺度门控融合以及末端 DUpsampling，保留 FADE 和 DUpsampling 的原方法名称。默认骨干为 `resnet101`，解码通道数为 256；双分支编码器参数独立。`pretrained=True` 当前不受支持，模型初始化不会自动下载 ImageNet 权重。

## Environment

### Locally verified environment

以下为 **2026-11-02 实测运行当前代码的本机环境**，用于说明已验证的运行条件，并非所有历史训练实验的环境追溯记录。

| 项目 | 版本或配置 |
| --- | --- |
| 操作系统 | Windows，系统 build `10.0.26200`，64 位 |
| Python | `3.10.16`（Anaconda） |
| PyTorch | `2.8.0.dev20250320+cu128` |
| PyTorch CUDA 构建版本 | `12.8` |
| cuDNN | `torch.backends.cudnn.version() = 90701`（9.7.1） |
| GPU | NVIDIA GeForce RTX 5070 Ti，16 GB 显存级别 |
| 系统内存 | 64 GB 级别 |
| NumPy | `1.25.0` |
| Pillow | `11.2.1` |

本机 PyTorch 为开发版（nightly），版本号应完整记录，不能简写为稳定版 `2.8.0`。`torch.version.cuda` 表示 PyTorch 的 CUDA 构建版本，不代表本机独立安装的 CUDA Toolkit 版本。

### Dependencies

| 范围 | 依赖 |
| --- | --- |
| 网络前向计算、FADE、DUpsampling、KDRM | `torch` |
| 评价指标、随机种子、CAM 等工具 | `numpy` |
| CAM 图像保存 | `Pillow`，导入名为 `PIL` |

Python 建议使用 **3.10**，与当前验证环境一致。当前代码包含 `int | None` 等类型标注，请勿直接沿用早期参考仓库的 Python 3.7/3.8 环境。

当前 `FADE.py` 已包含纯 PyTorch 的 CARAFE 运算和初始化实现，**不需要安装 MMCV 或编译额外 CUDA 扩展**。当前骨干网络由本仓库实现，不依赖 `torchvision`、`timm`、`mmsegmentation` 或 `resnest` 包；无需为本模型额外安装 `torchaudio`。

本机另装有 `torchvision 0.22.0.dev20250321+cu128`、`tqdm 4.67.1`、`tifffile 2025.2.18`、OpenCV `4.10.0` 和 `rasterio 1.4.4`。这些不是当前网络导入及前向计算的必需依赖；如后续加入数据读取、训练或 GeoTIFF 制图入口，再按对应脚本安装。

## Installation

在 Conda 环境中安装依赖，然后将工作目录切换到本仓库根目录，即包含 `Model/`、`configs/` 和 `Utils/` 的目录。

```shell
conda create -n bimorphnet python=3.10 -y
conda activate bimorphnet
```

### GPU installation option

以下使用 PyTorch 官方提供的稳定版 CUDA 12.8 wheel，作为新建环境的安装方案：

```shell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.25.0 Pillow==11.2.1
```

**该稳定版方案与上方实测 nightly 环境不同，本次未新建环境验证。** 安装后请执行下面的环境检查与 Quick start。GPU 运行需要兼容所选 PyTorch CUDA 构建版本的 NVIDIA 驱动；安装选项参见 [PyTorch 官方版本页面](https://pytorch.org/get-started/previous-versions/)。

### CPU installation option

无需 GPU 时，可将上述 PyTorch 安装命令替换为：

```shell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

随后同样安装 NumPy 和 Pillow。CPU 适合接口检查与小规模调试，完整模型运行速度取决于硬件。

### Check the environment

```shell
python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('cuDNN:', torch.backends.cudnn.version())"
```

## Repository structure

```text
LFS/
├── README.md
├── Model/
│   ├── BiMorphNet.py       # 主模型、RCGI、HCFR 与 HDGR 调用路径
│   ├── KDRM.py            # 依赖度与任务相关性通道重标定
│   ├── FADE.py            # FADE、级联上采样与同尺度门控融合
│   ├── Dupsample.py       # DUpsampling 实现
│   └── __init__.py
├── configs/
│   ├── train_config.py    # 数据路径与训练配置容器
│   └── __init__.py
└── Utils/                 # 损失、指标、权重、EMA、调度器与 CAM 工具
```

`configs/train_config.py` 中的参数由外部训练代码读取；直接实例化 `BiMorphNet` 不会自动读取该配置。目前配置仍保留工作项目中的设置，例如 SGD 和 300 epochs，不能直接视为已核定的论文复现配置。

## Quick start

在仓库根目录运行下面的 Python 示例，或保存为本地脚本后运行：

```python
import torch
from Model.BiMorphNet import BiMorphNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiMorphNet(num_classes=1).to(device).eval()

# 随机张量仅检查接口；真实影像需按训练时的方法预处理和归一化。
rgb = torch.randn(1, 3, 224, 224, device=device)
sar = torch.randn(1, 2, 224, 224, device=device)

with torch.inference_mode():
    logits = model(rgb, sar)
    probability = torch.sigmoid(logits)
    prediction = (probability >= 0.25).to(torch.uint8)

print("Logits:", tuple(logits.shape))
print("Prediction:", tuple(prediction.shape))
```

预期两项输出尺寸均为 `(1, 1, 224, 224)`。

| 张量 | 格式 | 含义 |
| --- | --- | --- |
| `rgb` | `[B, 3, H, W]`，浮点型 | 按 R、G、B 排列的光学影像 |
| `sar` | `[B, 2, H, W]`，浮点型 | 按 VV、VH 排列的 SAR 影像 |
| `logits` | `[B, 1, H, W]` | 未经 sigmoid 的前景预测分数 |

两路输入必须具有相同的 batch size、空间尺寸和地理对齐关系。`0.25` 是当前配置保留的二值化阈值；随机初始化模型的输出仅用于运行检查，不代表实际分割能力。

## Loading trained weights

```python
from pathlib import Path
from Utils.checkpoint import load_model_weights

# 延续 Quick start 中的 model 和 device；请自行提供对应训练权重。
load_model_weights(
    Path("Model_save/BiMorphNet_best.pth"),
    model,
    device,
    use_ema=False,
)
model.eval()
```

加载器支持 `model_state_dict` 格式的 checkpoint 或纯模型参数字典。网络支持将旧版本的 MCAM、CDM 和跨层投影成员路径映射至当前名称；这只解决命名兼容，仍要求骨干网络、通道数及模块配置与权重匹配。

## Verification

在上方实测环境中已通过默认 `resnet101` 模型的 CUDA 前向检查：RGB 输入为 `(1, 3, 224, 224)`、SAR 输入为 `(1, 2, 224, 224)`，输出为有限值的 `(1, 1, 224, 224)` 张量。

此检查验证当前代码的运行接口，不代替完整训练、数据集评估或论文精度复现，也不表示已验证其他操作系统或新建的稳定版环境。

## Acknowledgements and references

网络中的 FADE 和 DUpsampling 保留原方法名称。相关方法与实现参考：

- [FADE 官方代码与论文说明](https://github.com/poppinace/fade)：Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling，ECCV 2022。原仓库使用的环境与本仓库不同，以本 README 的环境和当前代码依赖为准。
- [DUpsampling 原论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Decoders_Matter_for_Semantic_Segmentation_Data-Dependent_Decoding_Enables_Flexible_Feature_CVPR_2019_paper.pdf)：Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation，CVPR 2019。

本 README 的环境、安装和用法组织参考了 [FADE](https://github.com/poppinace/fade) 与 [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) 的开源说明结构。CMX 在此仅作为文档组织参考，不是本仓库的软件依赖。
