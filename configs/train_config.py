from dataclasses import asdict, dataclass, field
from pathlib import Path


# 项目根目录：自动定位到当前仓库目录，下面所有默认路径都基于它拼接。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 数据总目录：训练集、验证集、测试集路径统一从这里继续拼接。
DATA_ROOT = PROJECT_ROOT / "Data"


@dataclass
class TrainConfig:
    # 路径配置：供外部训练、验证和测试代码读取。
    project_root: Path = PROJECT_ROOT  # 项目根目录；通常不需要手动修改。
    train_rgb_dir: Path = DATA_ROOT / "train" / "rgb"  # 训练集 RGB 图像目录。
    train_sar_dir: Path = DATA_ROOT / "train" / "sar"  # 训练集 SAR 图像目录，通常为 VV/VH 双通道。
    train_label_dir: Path = DATA_ROOT / "train" / "label"  # 训练集标签目录。
    val_rgb_dir: Path = DATA_ROOT / "val" / "rgb"  # 验证集 RGB 图像目录。
    val_sar_dir: Path = DATA_ROOT / "val" / "sar"  # 验证集 SAR 图像目录。
    val_label_dir: Path = DATA_ROOT / "val" / "label"  # 验证集标签目录，用于计算验证指标。
    test_rgb_dir: Path = DATA_ROOT / "test" / "rgb"  # 测试集 RGB 图像目录，用于预测和独立评估。
    test_sar_dir: Path = DATA_ROOT / "test" / "sar"  # 测试集 SAR 图像目录，用于预测和独立评估。
    test_label_dir: Path = DATA_ROOT / "test" / "label"  # 测试集标签目录，用于评估已保存模型。
    model_save_dir: Path = PROJECT_ROOT / "Model_save"  # 模型权重保存目录，保存 latest.pth、best.pth 等文件。
    result_dir: Path = PROJECT_ROOT / "result"  # 预测结果输出目录。
    log_dir: Path = PROJECT_ROOT / "logs"  # 训练日志输出目录。

    # 数据和基础训练配置。
    image_size: int = 224  # 输入图像尺寸；RGB、SAR 和标签会统一缩放到该大小。
    input_mode: str = "rgb_sar"  # 输入模态；rgb_sar 表示同时使用 RGB 和 SAR，rgb/sar 表示只使用单一模态。
    epochs: int = 300  # 最大训练轮数。
    batch_size: int = 32  # 训练批大小；显存不足时优先调小该值。
    val_batch_size: int = 2  # 验证批大小；主实验使用 2。
    num_workers: int = 0  # DataLoader 数据读取进程数；Windows/IDE 运行时 0 最稳定。

    # BiMorphNet 主实验：与 D4C_finally_01_best.pth 保存的配置一致。
    optimizer_name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    momentum: float = 0.9  # 保留共享接口字段；AdamW 使用 betas，不使用此值。

    # 学习率调度配置。
    scheduler_name: str = "warmup_cosine"  # 学习率调度策略；warmup_cosine 表示预热后余弦衰减。
    warmup_epochs: int = 10  # 学习率预热轮数，训练初期逐步升高学习率。
    min_learning_rate: float = 1e-6  # 学习率衰减到后期时保留的最小学习率。
    # 早停配置。
    early_stopping_patience: int = 20  # 验证指标连续多少轮没有明显提升后停止训练。
    early_stopping_min_delta: float = 0.0  # 判定验证指标提升所需的最小变化量。
    early_stopping_start_epoch: int = 81  # 从第几轮开始启用早停检测；之前只训练、验证和保存权重，不提前停止。
    composite_metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "precision": 0.10,  # 精确率，越高越好；控制误检。
            "recall": 0.15,  # 召回率，越高越好；控制漏检。
            "f1": 0.20,  # F1 分数，越高越好；平衡 precision 和 recall。
            "target_iou": 0.20,  # 目标类 IoU，越高越好；重点评价防护林前景分割质量。
            "miou": 0.15,  # 平均 IoU，越高越好；兼顾目标类和背景类。
            "cldice": 0.15,  # clDice 拓扑连通性指标，越高越好；适合细长连通目标。
            "loss": 0.03,  # 验证损失，越低越好；训练脚本会转换成越高越好的 loss_score。
            "betti0_error": 0.01,  # 前景连通分量误差，越低越好；训练脚本会转换成 betti0_score。
            "betti1_error": 0.01,  # 孔洞数量误差，越低越好；训练脚本会转换成 betti1_score。
        }
    )  # 综合监控指标权重，用于计算 composite_score 并判断 best.pth 与早停。

    # 训练稳定性和显存优化配置。
    gradient_accumulation_steps: int = 1  # 梯度累积步数；显存不足但想模拟更大 batch 时可调大。
    use_amp: bool = True  # 是否使用自动混合精度训练，可降低显存占用并加速 GPU 训练。
    gradient_clip_norm: float = 1.0  # 梯度裁剪阈值，用于限制梯度过大导致的训练不稳定。

    # EMA 配置。
    use_ema: bool = True  # 是否启用模型参数指数滑动平均，常用于提升验证和测试稳定性。
    ema_decay: float = 0.999  # EMA 衰减系数，越接近 1 表示历史参数保留越多。
    validate_with_ema: bool = False  # 验证阶段是否使用 EMA 权重进行评估。

    # 评估和运行环境配置。
    threshold: float = 0.25  # 二值分割阈值，预测概率大于该值的像素判为目标类。
    seed: int = 42  # 随机种子，用于尽量保证训练结果可复现。
    device: str = "cuda"  # 训练设备；cuda 表示优先使用 GPU，训练脚本会在不可用时回退到 CPU。
    best_metric_name: str = "composite_score"  # 最佳模型保存和早停依据，当前使用多指标加权综合分数。

    def to_checkpoint_dict(self) -> dict:
        # 将配置转换为可保存到 checkpoint 的字典；Path 类型需要转成字符串。
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def make_config() -> TrainConfig:
    # 外部训练、预测和评估代码可通过该函数创建配置对象。
    return TrainConfig()
