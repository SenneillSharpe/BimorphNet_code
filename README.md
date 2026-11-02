# BiMorphNet

**Bi-stream Cross-modal Morphology-aware Hierarchical Network for Thin-Structure Segmentation**

BiMorphNet 用于光学与 SAR 遥感影像中的线性农田防护林（Linear Farmland Shelterbelts, LFS）二值语义分割。本仓库提供模型、实验配置、损失函数、评价指标及权重管理工具，暂未提供数据加载器、完整训练与测试入口、数据集和训练权重。

## 模型组成

| 组件 | 名称 | 代码入口 |
| --- | --- | --- |
| 双分支编码 | Independent optical/SAR residual encoders | `encoder.rgb_backbone`、`encoder.sar_backbone` |
| RCGI | Reciprocal Cross-modal Gated Interaction | `encoder.low_rcgi`、`mid1_rcgi`、`mid2_rcgi`、`high_rcgi` |
| KDRM | Kernelized Dependency-Relevance Modulation | `Model/KDRM.py`；`decoder.kdrm` |
| HCFR | Hierarchical Cross-modal Feature Relay | `decoder.hcfr_relay` 及四级特征投影 |
| HDGR | Hierarchical Detail-Guided Reconstruction | `decoder.hdgr_reconstruct` |

默认骨干为两条参数独立的 ResNet-101，解码通道数为256。HDGR 结合级联 FADE、同尺度门控融合和末端 DUpsampling。模型不会自动下载 ImageNet 权重，当前不支持 `pretrained=True`。

## 测试环境

以下环境于2026-11-02通过默认模型的 GPU 前向测试；未据此推定历史训练时所有软件包的版本。

| 项目 | 版本或配置 |
| --- | --- |
| 操作系统 | Windows 64位，build `10.0.26200` |
| Python | `3.10.16` |
| PyTorch | `2.8.0.dev20250320+cu128`（nightly） |
| PyTorch CUDA 构建版本 | `12.8` |
| cuDNN | `9.7.1` |
| GPU | NVIDIA GeForce RTX 5070 Ti，16 GB VRAM |
| 系统内存 | 64 GB |
| NumPy | `1.25.0` |
| Pillow | `11.2.1` |

网络依赖 PyTorch；评价指标和部分工具使用 NumPy，CAM 图像保存使用 Pillow。FADE 包含纯 PyTorch CARAFE 实现，无需安装 MMCV 或额外编译 CUDA 扩展。当前模型不依赖 torchvision、timm 或 mmsegmentation。

## 安装

```shell
git clone https://github.com/SenneillSharpe/BimorphNet_code.git
cd BimorphNet_code
conda create -n bimorphnet python=3.10 -y
conda activate bimorphnet
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.25.0 Pillow==11.2.1
```

上述安装命令使用稳定版 PyTorch 2.8.0，与测试环境中的 nightly 构建不同，尚未进行独立环境验证。其他 CUDA 或 CPU 安装选项参见 [PyTorch 官方版本页面](https://pytorch.org/get-started/previous-versions/)。

## 实验配置

`configs/train_config.py` 以原主实验 `D4C_finally_01_best.pth` 保存的配置为基础，当前发布版本将最大训练轮数设为150、二值化阈值设为0.5，其余训练配置沿用主实验。路径以当前仓库为根目录，不保留原计算机的绝对路径。`D4C_finally_01` 是 BiMorphNet 的历史代码名称。

| 参数 | 当前发布设置 |
| --- | --- |
| 输入 | RGB 3通道，SAR 2通道（VV、VH） |
| 图像尺寸 | `224 × 224` |
| 训练 / 验证 batch size | `32 / 2` |
| DataLoader workers | `0` |
| 优化器 | AdamW |
| 初始学习率 | `1e-4` |
| Weight decay | `1e-3` |
| AdamW betas / epsilon | `(0.9, 0.999)` / `1e-8` |
| 最大训练轮数 | `150` |
| 学习率调度 | 10轮 warm-up，随后 cosine decay，最低 `1e-6` |
| AMP / 梯度累积步数 | 开启 / `1` |
| 梯度裁剪最大范数 | `1.0` |
| EMA | 开启，decay为 `0.999` |
| 验证权重 | 普通模型权重，`validate_with_ema=False` |
| 随机种子 | `42` |
| 二值化阈值 | `0.5` |
| 模型选择 | 最大化验证集 `composite_score` |
| 早停 | 第81轮起监控，patience为20，min_delta为0 |

原主实验使用300轮上限和0.25阈值，第106轮保存最佳模型，第126轮触发早停。当前150轮上限和0.5阈值是发布配置的调整，尚未据此重新训练或评估；最佳模型按综合分数选择，而非按单一 mIoU 选择。配置中的 `momentum` 是共享接口的保留字段，AdamW 使用 betas。

综合分数的权重为：Precision 0.10、Recall 0.15、F1 0.20、前景 IoU 0.20、mIoU 0.15、clDice 0.15、loss 0.03、Betti-0 error 0.01、Betti-1 error 0.01；loss及Betti误差先转换为越高越好的分数再参与加权。

主实验使用 `Utils/loss.py` 中的 `CoVFocalDiceCLDiceLoss`，组合 Focal、Dice 和 soft-clDice。前10轮权重为 `0.3 / 0.5 / 0.2`，随后采用 CoV 动态加权。实现默认使用 Focal alpha 0.25、gamma 2.0、Dice smooth 1.0、软骨架迭代3次、CoV momentum 0.95；三项权重分别约束于 `[0.25, 0.60]`、`[0.25, 0.60]`、`[0.05, 0.30]`。历史日志可核实初始权重及动态变化，但 checkpoint 未单独保存完整的损失构造参数。

配置文件是参数容器。直接实例化 `BiMorphNet` 不会自动读取该文件；接入自己的训练循环时，需显式使用配置构建数据加载器、优化器、调度器和损失函数。

## 目录结构

```text
BimorphNet_code/
├── README.md
├── Model/
│   ├── BiMorphNet.py
│   ├── KDRM.py
│   ├── FADE.py
│   ├── Dupsample.py
│   └── __init__.py
├── configs/
│   ├── train_config.py
│   └── __init__.py
└── Utils/
```

## 快速运行

在仓库根目录运行以下 Python 示例：

```python
import torch
from Model.BiMorphNet import BiMorphNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiMorphNet(num_classes=1).to(device).eval()

rgb = torch.randn(1, 3, 224, 224, device=device)
sar = torch.randn(1, 2, 224, 224, device=device)

with torch.inference_mode():
    logits = model(rgb, sar)
    prediction = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)

print("Logits:", tuple(logits.shape))
print("Prediction:", tuple(prediction.shape))
```

预期两项输出尺寸均为 `(1, 1, 224, 224)`。模型输出为未经 sigmoid 的 logits；随机输入和随机初始化仅用于检查接口。真实推理需要对应训练权重，以及与训练一致的预处理和归一化。

光学输入按 R、G、B 排列，SAR输入按 VV、VH 排列，两路浮点张量必须具有相同的 batch size、空间尺寸和地理对齐关系。

## 加载权重

```python
from pathlib import Path
from Utils.checkpoint import load_model_weights

# 使用上例中的 model 和 device，自行提供模型权重。
load_model_weights(
    Path("Model_save/BiMorphNet_best.pth"),
    model,
    device,
    use_ema=False,
)
model.eval()
```

支持包含 `model_state_dict` 的 checkpoint 和纯参数字典。历史 MCAM、CDM及跨层投影的成员名称会在加载时映射为当前名称；骨干、通道数和模块配置仍需与权重一致。

## 致谢

- [FADE](https://github.com/poppinace/fade)：Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling，ECCV 2022。
- [DUpsampling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Decoders_Matter_for_Semantic_Segmentation_Data-Dependent_Decoding_Enables_Flexible_Feature_CVPR_2019_paper.pdf)：Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation，CVPR 2019。
