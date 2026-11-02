import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """按 PyTorch 常用 reduction 规则汇总逐样本损失。"""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of: mean, sum, none.")


def _to_probabilities(inputs: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """把模型输出转换为 [0, 1] 概率图；训练脚本默认传入 logits。"""
    return torch.sigmoid(inputs) if from_logits else inputs


def _validate_same_shape(inputs: torch.Tensor, labels: torch.Tensor) -> None:
    """损失函数要求预测和标签具有相同的 N,C,H,W 或 N,C,D,H,W 形状。"""
    if inputs.shape != labels.shape:
        raise ValueError(f"inputs and labels must have the same shape, got {inputs.shape} and {labels.shape}.")


class BinarySegmentationLoss(nn.Module):
    """二分类语义分割损失函数。

    模型输出应为未经过 sigmoid 的 logits，标签通常为 0/1 浮点张量。
    pos_weight 可用于提高前景类权重，缓解前景像素较少造成的类别不平衡。
    """

    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        # pos_weight=None 时使用普通 BCEWithLogitsLoss；否则提高正样本权重。
        weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """根据模型 logits 和真实标签计算损失值。"""
        if self.loss.pos_weight is not None:
            # 确保权重张量和模型输出位于同一个设备，避免 CPU/GPU 不一致报错。
            self.loss.pos_weight = self.loss.pos_weight.to(logits.device)
        return self.loss(logits, labels)


class BinaryFocalLoss(nn.Module):
    """二分类 Focal Loss。

    适用于前景像素很少、普通 BCE 容易被大量背景像素主导的二分类分割任务。
    模型输入仍然应为未经过 sigmoid 的 logits，内部会用稳定的
    binary_cross_entropy_with_logits 计算基础损失。
    """

    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be None or a value in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.register_buffer("pos_weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """根据 logits 和 0/1 标签计算 Focal Loss。"""
        labels = labels.float()
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )

        probabilities = torch.sigmoid(logits)
        pt = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_weight = self.alpha * labels + (1.0 - self.alpha) * (1.0 - labels)
            focal_weight = alpha_weight * focal_weight

        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryDiceLoss(nn.Module):
    """二分类 Dice Loss。

    主要参考 V-Net 中的 Dice coefficient objective：
    通过预测概率图和真实标签的重叠程度计算损失，适合前景/背景像素严重不平衡的分割任务。
    默认使用 V-Net 风格的平方分母；若要作为 soft-clDice 的区域项，可关闭 squared_denominator。
    """

    def __init__(
        self,
        smooth: float = 1.0,
        eps: float = 1e-7,
        from_logits: bool = True,
        squared_denominator: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        if smooth < 0:
            raise ValueError("smooth must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")

        self.smooth = smooth
        self.eps = eps
        self.from_logits = from_logits
        self.squared_denominator = squared_denominator
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """根据 logits/概率图和 0/1 标签计算 Dice Loss。"""
        _validate_same_shape(inputs, labels)
        probabilities = _to_probabilities(inputs, self.from_logits)
        labels = labels.to(device=inputs.device, dtype=probabilities.dtype)
        reduce_dims = tuple(range(1, probabilities.ndim))

        intersection = torch.sum(probabilities * labels, dim=reduce_dims)
        if self.squared_denominator:
            denominator = torch.sum(probabilities.pow(2), dim=reduce_dims) + torch.sum(labels.pow(2), dim=reduce_dims)
        else:
            denominator = torch.sum(probabilities, dim=reduce_dims) + torch.sum(labels, dim=reduce_dims)

        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth + self.eps)
        return _reduce_loss(1.0 - dice_score, self.reduction)


class SoftSkeletonize(nn.Module):
    """soft-clDice 使用的可微软骨架提取。

    参考 soft-clDice 论文中的 soft-skeletonization：用 min/max pooling 近似形态学腐蚀、
    膨胀和开运算，从而在训练中保留连通性信息。支持 2D N,C,H,W 和 3D N,C,D,H,W。
    """

    def __init__(self, iterations: int = 3):
        super().__init__()
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")
        self.iterations = int(iterations)

    def _soft_erode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            erode_h = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
            erode_w = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
            return torch.minimum(erode_h, erode_w)
        if image.ndim == 5:
            erode_d = -F.max_pool3d(-image, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            erode_h = -F.max_pool3d(-image, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
            erode_w = -F.max_pool3d(-image, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
            return torch.minimum(torch.minimum(erode_d, erode_h), erode_w)
        raise ValueError(f"SoftSkeletonize expects 4D or 5D tensors, got {image.ndim}D.")

    def _soft_dilate(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)
        if image.ndim == 5:
            return F.max_pool3d(image, kernel_size=3, stride=1, padding=1)
        raise ValueError(f"SoftSkeletonize expects 4D or 5D tensors, got {image.ndim}D.")

    def _soft_open(self, image: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(image))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """返回输入概率图的软骨架图。"""
        image = image.clamp(0.0, 1.0)
        opened = self._soft_open(image)
        skeleton = F.relu(image - opened)

        for _ in range(self.iterations):
            image = self._soft_erode(image)
            opened = self._soft_open(image)
            delta = F.relu(image - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton.clamp(0.0, 1.0)


class SoftCLDiceLoss(nn.Module):
    """二分类 soft-clDice Loss。

    参考 Shit et al. 的 soft-clDice：分别计算预测软骨架落在标签内的拓扑精度
    和标签软骨架落在预测内的拓扑召回，再用二者的调和平均约束连通性。
    该损失更适合道路、河网、血管等细长或管状目标。
    """

    def __init__(
        self,
        iter_: int = 3,
        smooth: float = 1.0,
        eps: float = 1e-7,
        from_logits: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        if smooth < 0:
            raise ValueError("smooth must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")

        self.smooth = smooth
        self.eps = eps
        self.from_logits = from_logits
        self.reduction = reduction
        self.skeletonize = SoftSkeletonize(iterations=iter_)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """根据 logits/概率图和 0/1 标签计算 soft-clDice Loss。"""
        _validate_same_shape(inputs, labels)
        probabilities = _to_probabilities(inputs, self.from_logits)
        labels = labels.to(device=inputs.device, dtype=probabilities.dtype).clamp(0.0, 1.0)
        probabilities = probabilities.clamp(0.0, 1.0)

        pred_skeleton = self.skeletonize(probabilities)
        label_skeleton = self.skeletonize(labels)
        reduce_dims = tuple(range(1, probabilities.ndim))

        tprec = (torch.sum(pred_skeleton * labels, dim=reduce_dims) + self.smooth) / (
            torch.sum(pred_skeleton, dim=reduce_dims) + self.smooth + self.eps
        )
        tsens = (torch.sum(label_skeleton * probabilities, dim=reduce_dims) + self.smooth) / (
            torch.sum(label_skeleton, dim=reduce_dims) + self.smooth + self.eps
        )
        cldice_score = (2.0 * tprec * tsens) / (tprec + tsens + self.eps)
        return _reduce_loss(1.0 - cldice_score, self.reduction)


class SoftDiceCLDiceLoss(nn.Module):
    """soft Dice + soft-clDice 组合损失。

    soft-clDice 原论文建议将区域重叠项和拓扑项组合：
    (1 - alpha) * DiceLoss + alpha * clDiceLoss。alpha 越大，越强调连通性。
    """

    def __init__(
        self,
        alpha: float = 0.5,
        iter_: int = 3,
        smooth: float = 1.0,
        eps: float = 1e-7,
        from_logits: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")

        self.alpha = alpha
        self.reduction = reduction
        self.dice_loss = BinaryDiceLoss(
            smooth=smooth,
            eps=eps,
            from_logits=from_logits,
            squared_denominator=False,
            reduction="none",
        )
        self.cldice_loss = SoftCLDiceLoss(
            iter_=iter_,
            smooth=smooth,
            eps=eps,
            from_logits=from_logits,
            reduction="none",
        )

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算 Dice 区域项和 clDice 拓扑项的加权和。"""
        loss = (1.0 - self.alpha) * self.dice_loss(inputs, labels) + self.alpha * self.cldice_loss(inputs, labels)
        return _reduce_loss(loss, self.reduction)


class FocalDiceCLDiceLoss(nn.Module):
    """Focal Loss + Dice Loss + soft-clDice Loss 组合损失。

    Focal Loss 主要处理前景/背景类别不平衡；
    Dice Loss 主要约束目标区域重叠；
    soft-clDice Loss 主要约束细长目标的连通性。
    """

    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        cldice_weight: float = 0.2,
        focal_alpha: float | None = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
        dice_smooth: float = 1.0,
        cldice_iter: int = 3,
        from_logits: bool = True,
    ):
        super().__init__()
        if focal_weight < 0 or dice_weight < 0 or cldice_weight < 0:
            raise ValueError("loss weights must be non-negative.")
        if not from_logits:
            raise ValueError("FocalDiceCLDiceLoss expects logits because BinaryFocalLoss is logits-based.")

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight

        self.focal_loss = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight,
            reduction="mean",
        )
        self.dice_loss = BinaryDiceLoss(
            smooth=dice_smooth,
            from_logits=from_logits,
            squared_denominator=False,
            reduction="mean",
        )
        self.cldice_loss = SoftCLDiceLoss(
            iter_=cldice_iter,
            from_logits=from_logits,
            reduction="mean",
        )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算三类损失的加权和。"""
        focal = self.focal_loss(logits, labels)
        dice = self.dice_loss(logits, labels)
        cldice = self.cldice_loss(logits, labels)
        return (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.cldice_weight * cldice
        )

    def get_current_weights(self) -> dict[str, float]:
        """返回当前静态权重，供训练日志记录。"""
        return {
            "focal": float(self.focal_weight),
            "dice": float(self.dice_weight),
            "cldice": float(self.cldice_weight),
        }


class CoVFocalDiceCLDiceLoss(nn.Module):
    """基于 CoV-Weighting 的 Focal + Dice + soft-clDice 动态组合损失。

    warmup 阶段使用初始静态权重；超过 warmup epoch 后，根据三项损失
    loss ratio 的变异系数动态调整权重，并用 min/max ratio 限制权重范围。
    """

    loss_names = ("focal", "dice", "cldice")

    def __init__(
        self,
        initial_weights: Mapping[str, float] | None = None,
        cov_warmup_epochs: int = 10,
        cov_momentum: float = 0.95,
        min_weight_ratios: Mapping[str, float] | None = None,
        max_weight_ratios: Mapping[str, float] | None = None,
        total_weight_scale: float | None = None,
        focal_alpha: float | None = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
        dice_smooth: float = 1.0,
        cldice_iter: int = 3,
        from_logits: bool = True,
        eps: float = 1e-7,
    ):
        super().__init__()
        if cov_warmup_epochs < 0:
            raise ValueError("cov_warmup_epochs must be non-negative.")
        if not 0.0 <= cov_momentum < 1.0:
            raise ValueError("cov_momentum must be in [0, 1).")
        if not from_logits:
            raise ValueError("CoVFocalDiceCLDiceLoss expects logits because BinaryFocalLoss is logits-based.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        initial = self._mapping_to_tensor(
            initial_weights,
            defaults={"focal": 0.3, "dice": 0.5, "cldice": 0.2},
            field_name="initial_weights",
        )
        if torch.any(initial < 0) or initial.sum() <= 0:
            raise ValueError("initial_weights must be non-negative and have a positive sum.")

        min_ratios = self._mapping_to_tensor(
            min_weight_ratios,
            defaults={"focal": 0.25, "dice": 0.25, "cldice": 0.05},
            field_name="min_weight_ratios",
        )
        max_ratios = self._mapping_to_tensor(
            max_weight_ratios,
            defaults={"focal": 0.60, "dice": 0.60, "cldice": 0.30},
            field_name="max_weight_ratios",
        )
        if torch.any(min_ratios < 0) or torch.any(max_ratios > 1) or torch.any(min_ratios > max_ratios):
            raise ValueError("weight ratio thresholds must satisfy 0 <= min <= max <= 1.")
        if min_ratios.sum() > 1.0 + eps or max_ratios.sum() < 1.0 - eps:
            raise ValueError("weight ratio thresholds must allow ratios that sum to 1.")

        self.eps = float(eps)
        self.cov_warmup_epochs = int(cov_warmup_epochs)
        self.cov_momentum = float(cov_momentum)
        self.total_weight_scale = float(initial.sum().item() if total_weight_scale is None else total_weight_scale)
        self.current_epoch = 0

        initial_ratios = initial / initial.sum().clamp_min(self.eps)
        initial_ratios = self._project_ratios(initial_ratios, min_ratios, max_ratios)
        self.register_buffer("min_weight_ratios", min_ratios)
        self.register_buffer("max_weight_ratios", max_ratios)
        self.register_buffer("initial_weight_ratios", initial_ratios)
        self.register_buffer("current_weight_ratios", initial_ratios.clone())
        self.register_buffer("current_weights", initial_ratios * self.total_weight_scale)
        self.register_buffer("loss_ema", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("ratio_ema", torch.ones(3, dtype=torch.float32))
        self.register_buffer("ratio_square_ema", torch.ones(3, dtype=torch.float32))
        self.register_buffer("cov_initialized", torch.tensor(False))

        self.focal_loss = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight,
            reduction="mean",
        )
        self.dice_loss = BinaryDiceLoss(
            smooth=dice_smooth,
            from_logits=from_logits,
            squared_denominator=False,
            reduction="mean",
        )
        self.cldice_loss = SoftCLDiceLoss(
            iter_=cldice_iter,
            from_logits=from_logits,
            reduction="mean",
        )

    @classmethod
    def _mapping_to_tensor(
        cls,
        values: Mapping[str, float] | None,
        defaults: Mapping[str, float],
        field_name: str,
    ) -> torch.Tensor:
        merged = dict(defaults)
        if values is not None:
            unknown = sorted(set(values) - set(cls.loss_names))
            if unknown:
                raise ValueError(f"{field_name} contains unsupported keys: {unknown}.")
            merged.update(values)
        return torch.tensor([float(merged[name]) for name in cls.loss_names], dtype=torch.float32)

    def _project_ratios(
        self,
        ratios: torch.Tensor,
        min_ratios: torch.Tensor | None = None,
        max_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        min_values = self.min_weight_ratios if min_ratios is None else min_ratios
        max_values = self.max_weight_ratios if max_ratios is None else max_ratios
        projected = ratios.float().clamp(min=min_values, max=max_values)

        for _ in range(10):
            total = projected.sum()
            if torch.abs(total - 1.0) <= self.eps:
                break
            if total > 1.0:
                capacity = (projected - min_values).clamp_min(0.0)
                if capacity.sum() <= self.eps:
                    break
                projected = torch.maximum(projected - (total - 1.0) * capacity / capacity.sum(), min_values)
            else:
                capacity = (max_values - projected).clamp_min(0.0)
                if capacity.sum() <= self.eps:
                    break
                projected = torch.minimum(projected + (1.0 - total) * capacity / capacity.sum(), max_values)
        return projected / projected.sum().clamp_min(self.eps)

    def set_epoch(self, epoch: int) -> None:
        """由训练循环设置当前 epoch，用于判断是否结束 warmup。"""
        self.current_epoch = int(epoch)

    def _update_dynamic_weights(self, component_losses: torch.Tensor) -> None:
        values = component_losses.detach().to(device=self.current_weights.device, dtype=torch.float32).clamp_min(self.eps)
        if not bool(self.cov_initialized.item()):
            self.loss_ema.copy_(values)
            self.ratio_ema.fill_(1.0)
            self.ratio_square_ema.fill_(1.0)
            self.cov_initialized.fill_(True)
            return

        ratio = values / self.loss_ema.clamp_min(self.eps)
        momentum = self.cov_momentum
        self.loss_ema.mul_(momentum).add_(values, alpha=1.0 - momentum)
        self.ratio_ema.mul_(momentum).add_(ratio, alpha=1.0 - momentum)
        self.ratio_square_ema.mul_(momentum).add_(ratio.pow(2), alpha=1.0 - momentum)

        ratio_variance = (self.ratio_square_ema - self.ratio_ema.pow(2)).clamp_min(0.0)
        cov = torch.sqrt(ratio_variance) / self.ratio_ema.abs().clamp_min(self.eps)
        if cov.sum() <= self.eps:
            next_ratios = self.initial_weight_ratios
        else:
            next_ratios = cov / cov.sum().clamp_min(self.eps)

        next_ratios = self._project_ratios(next_ratios)
        self.current_weight_ratios.copy_(next_ratios)
        self.current_weights.copy_(next_ratios * self.total_weight_scale)

    def get_current_weight_ratios(self) -> dict[str, float]:
        """返回当前归一化权重比例，三项之和为 1。"""
        values = self.current_weight_ratios.detach().cpu().tolist()
        return {name: round(float(value), 6) for name, value in zip(self.loss_names, values)}

    def get_current_weights(self) -> dict[str, float]:
        """返回当前实际损失权重，供训练日志记录。"""
        values = self.current_weights.detach().cpu().tolist()
        return {name: round(float(value), 6) for name, value in zip(self.loss_names, values)}

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算三项损失，并在训练 warmup 结束后按 CoV 动态更新权重。"""
        focal = self.focal_loss(logits, labels)
        dice = self.dice_loss(logits, labels)
        cldice = self.cldice_loss(logits, labels)
        component_losses = torch.stack([focal, dice, cldice])

        if self.training and self.current_epoch > self.cov_warmup_epochs:
            self._update_dynamic_weights(component_losses)

        weights = self.current_weights.to(device=logits.device, dtype=component_losses.dtype)
        return weights[0] * focal + weights[1] * dice + weights[2] * cldice
