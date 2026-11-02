import math

import torch


def warmup_cosine_factor(epoch: int, total_epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    if total_epochs <= 0:
        return 1.0
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_epochs = max(0, warmup_epochs)

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return max((epoch + 1) / warmup_epochs, min_lr_ratio)

    if total_epochs <= warmup_epochs:
        return 1.0

    progress = min(max((epoch - warmup_epochs) / (total_epochs - warmup_epochs), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ===== SGD + poly scheduler 参考代码（当前仅作为注释，不会自动生效）=====
# 若后续将 CNN/ResNet/HRNet 风格模型切换为 SGD，并希望复现语义分割论文中常见的 poly LR，
# 可以取消下面 poly_factor() 的注释，并在 build_scheduler() 中取消 poly 分支的注释。
#
# def poly_factor(epoch: int, total_epochs: int, power: float, min_lr_ratio: float) -> float:
#     if total_epochs <= 0:
#         return 1.0
#     min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
#     progress = min(max(epoch / total_epochs, 0.0), 1.0)
#     factor = (1.0 - progress) ** power
#     return max(factor, min_lr_ratio)
#
#
def build_scheduler(config, optimizer: torch.optim.Optimizer):
    scheduler_name = config.scheduler_name.lower()
    if scheduler_name in {"none", "off", "disabled"}:
        return None
    # ===== SGD + poly scheduler 参考分支（当前仅作为注释，不会自动生效）=====
    # 启用步骤：
    # 1. 在 configs/train_config.py 中设置 scheduler_name = "poly"。
    # 2. 取消上方 poly_factor() 和下方 if scheduler_name == "poly" 分支的注释。
    # 3. 如需显式配置 power，可在 TrainConfig 中增加 poly_power: float = 0.9；
    #    若不增加，下面 getattr(config, "poly_power", 0.9) 会默认使用 0.9。
    #
    # if scheduler_name == "poly":
    #     min_lr_ratio = config.min_learning_rate / config.learning_rate if config.learning_rate > 0 else 0.0
    #     return torch.optim.lr_scheduler.LambdaLR(
    #         optimizer,
    #         lr_lambda=lambda epoch: poly_factor(
    #             epoch=epoch,
    #             total_epochs=config.epochs,
    #             power=getattr(config, "poly_power", 0.9),
    #             min_lr_ratio=min_lr_ratio,
    #         ),
    #     )
    #
    if scheduler_name != "warmup_cosine":
        raise ValueError("Unsupported scheduler_name: " f"{config.scheduler_name!r}. Choose from: warmup_cosine, none.")

    min_lr_ratio = config.min_learning_rate / config.learning_rate if config.learning_rate > 0 else 0.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine_factor(
            epoch=epoch,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )
