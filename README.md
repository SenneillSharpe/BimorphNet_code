# BiMorphNet

**Bi-stream Cross-modal Morphology-aware Hierarchical Network for Thin-Structure Segmentation**

BiMorphNet 用于光学与 SAR 遥感影像中的线性农田防护林（Linear Farmland Shelterbelts, LFS）二值语义分割。本仓库提供模型、实验配置、损失函数、评价指标及权重管理工具。

## 模型组成

| 组件 | 名称 | 代码入口 |
| --- | --- | --- |
| 双分支编码 | Independent optical/SAR residual encoders | `encoder.rgb_backbone`、`encoder.sar_backbone` |
| RCGI | Reciprocal Cross-modal Gated Interaction | `encoder.low_rcgi`、`mid1_rcgi`、`mid2_rcgi`、`high_rcgi` |
| KDRM | Kernelized Dependency-Relevance Modulation | `Model/KDRM.py`；`decoder.kdrm` |
| HCFR | Hierarchical Cross-modal Feature Relay | `decoder.hcfr_relay` 及四级特征投影 |
| HDGR | Hierarchical Detail-Guided Reconstruction | `decoder.hdgr_reconstruct` |

## 安装

```shell
git clone https://github.com/SenneillSharpe/BimorphNet_code.git
cd BimorphNet_code
conda create -n bimorphnet python=3.10 -y
conda activate bimorphnet
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==1.25.0 Pillow==11.2.1
```

安装命令采用 PyTorch 2.8.0 的 CUDA 12.8 构建。其他 CUDA 或 CPU 安装选项参见 [PyTorch 官方版本页面](https://pytorch.org/get-started/previous-versions/)。

## 实验配置

当前配置见 `configs/train_config.py`：最大训练轮数为150，二值化阈值为0.5，优化器为 AdamW。数据、模型权重和输出路径均以仓库根目录为基准。

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

最佳模型按验证集综合分数选择。`momentum` 是共享配置中的保留字段，AdamW 使用 betas。

当前150轮上限和0.5阈值尚未用于重新训练或评估。

综合分数的权重为：Precision 0.10、Recall 0.15、F1 0.20、前景 IoU 0.20、mIoU 0.15、clDice 0.15、loss 0.03、Betti-0 error 0.01、Betti-1 error 0.01；loss 及 Betti 误差先转换为越高越好的分数再参与加权。

组合损失由 `Utils/loss.py` 中的 `CoVFocalDiceCLDiceLoss` 实现，组合 Focal、Dice 和 soft-clDice。前10轮权重为 `0.3 / 0.5 / 0.2`，随后采用 CoV 动态加权。实现默认使用 Focal alpha 0.25、gamma 2.0、Dice smooth 1.0、软骨架迭代3次、CoV momentum 0.95；三项权重分别约束于 `[0.25, 0.60]`、`[0.25, 0.60]`、`[0.05, 0.30]`。

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

## 致谢

- [FADE](https://github.com/poppinace/fade)：Fusing the Assets of Decoder and Encoder for Task-Agnostic Upsampling，ECCV 2022。
- [DUpsampling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Decoders_Matter_for_Semantic_Segmentation_Data-Dependent_Decoding_Enables_Flexible_Feature_CVPR_2019_paper.pdf)：Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation，CVPR 2019。
