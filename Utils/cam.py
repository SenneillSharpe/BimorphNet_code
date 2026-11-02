from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


class GradCAMForSegmentation:
    """面向二分类语义分割模型的 Grad-CAM 计算工具。

    当前项目模型采用双输入前向接口：`model(rgb, sar)`。
    rgb 的形状为 `(B, 3, H, W)`，sar 的形状为 `(B, 2, H, W)`；
    模型输出为未经过 sigmoid 的 logits，形状应为 `(B, 1, H, W)`。
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str) -> None:
        """初始化 CAM 生成器，并在目标层上注册 forward/backward hook。

        Args:
            model: 已构建并加载好权重的 PyTorch 分割模型。
            target_layer_name: 目标层名称，支持完整名称或唯一后缀名称，
                例如 `decoder`、`encoder.aspp`、`rgb_backbone.layer4`。
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = self._find_target_layer(target_layer_name)
        self.feature_map: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self.forward_handle: torch.utils.hooks.RemovableHandle | None = None
        self.backward_handle: torch.utils.hooks.RemovableHandle | None = None
        self._register_hooks()

    def _find_target_layer(self, target_layer_name: str) -> torch.nn.Module:
        """根据字符串名称在模型中查找目标层。

        优先精确匹配 `named_modules()` 中的名称；如果精确匹配失败，
        再允许唯一后缀匹配，方便使用 `aspp`、`rgb_backbone.layer4`
        这类不包含顶层前缀的名称。
        """
        if not target_layer_name:
            raise ValueError("target_layer_name 不能为空。")

        named_modules = dict(self.model.named_modules())
        if target_layer_name in named_modules:
            return named_modules[target_layer_name]

        suffix_matches = [
            (name, module)
            for name, module in named_modules.items()
            if name and (name.endswith(f".{target_layer_name}") or name == target_layer_name)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0][1]
        if len(suffix_matches) > 1:
            candidates = ", ".join(name for name, _ in suffix_matches)
            raise ValueError(
                f"目标层名称 {target_layer_name!r} 匹配到多个模块：{candidates}。"
                "请使用更完整的层名称。"
            )

        available = ", ".join(name for name in named_modules if name)
        raise ValueError(
            f"在模型中找不到目标层 {target_layer_name!r}。"
            f"可用层名称包括：{available}"
        )

    def _register_hooks(self) -> None:
        """在目标层注册 hook，用于保存 feature map 和 gradient。"""
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    ) -> None:
        """forward hook：保存目标层前向传播输出的特征图。"""
        del module, inputs
        self.feature_map = self._extract_tensor(output, "目标层 forward 输出")

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        """backward hook：保存目标层输出对应的反向梯度。"""
        del module, grad_input
        self.gradient = self._extract_tensor(grad_output, "目标层 backward 梯度")

    @staticmethod
    def _extract_tensor(
        value: torch.Tensor | tuple[torch.Tensor | None, ...] | list[torch.Tensor | None],
        context: str,
    ) -> torch.Tensor:
        """从 hook 的输出中提取第一个 4D Tensor。

        常规卷积层直接返回 Tensor；少数模块可能返回 tuple/list。
        Grad-CAM 需要 `(B, C, H, W)` 形式的特征图或梯度。
        """
        tensors: Iterable[torch.Tensor | None]
        if isinstance(value, torch.Tensor):
            tensors = (value,)
        elif isinstance(value, (tuple, list)):
            tensors = value
        else:
            raise TypeError(f"{context} 的类型不受支持：{type(value)!r}。")

        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() != 4:
                    raise ValueError(f"{context} 必须是 4D Tensor，实际形状为 {tuple(tensor.shape)}。")
                return tensor
        raise ValueError(f"{context} 中没有可用于 Grad-CAM 的 Tensor。")

    @staticmethod
    def _prepare_target_mask(
        target_mask: torch.Tensor | np.ndarray,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """整理 target_mask，使其与概率图具有相同 batch 和空间尺寸。

        输入可为 `(B, H, W)` 或 `(B, 1, H, W)`；如果空间尺寸不同，
        使用最近邻插值缩放，避免改变二值区域含义。
        """
        if isinstance(target_mask, np.ndarray):
            mask = torch.from_numpy(target_mask)
        elif isinstance(target_mask, torch.Tensor):
            mask = target_mask
        else:
            raise TypeError("target_mask 必须是 torch.Tensor 或 numpy.ndarray。")

        mask = mask.to(device=probabilities.device, dtype=probabilities.dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(
                "target_mask 的形状必须是 (B,H,W) 或 (B,1,H,W)，"
                f"实际形状为 {tuple(mask.shape)}。"
            )
        if mask.size(0) != probabilities.size(0):
            raise ValueError(
                f"target_mask batch 大小必须等于概率图 batch，"
                f"实际为 {mask.size(0)} 和 {probabilities.size(0)}。"
            )
        if mask.size(1) != 1:
            raise ValueError(f"target_mask 通道数必须为 1，实际为 {mask.size(1)}。")
        if mask.shape[-2:] != probabilities.shape[-2:]:
            mask = F.interpolate(mask, size=probabilities.shape[-2:], mode="nearest")
        return (mask > 0).to(dtype=probabilities.dtype)

    def _build_backward_target(
        self,
        probabilities: torch.Tensor,
        target_mask: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        """构建反向传播目标标量。

        target_mask 为空时，使用整张前景概率图的均值；
        target_mask 不为空时，只对 mask 标记的目标区域求均值。
        """
        if target_mask is None:
            return probabilities.mean()

        mask = self._prepare_target_mask(target_mask, probabilities)
        mask_sum = mask.sum()
        if mask_sum.item() <= 0:
            raise ValueError("target_mask 中没有有效目标区域，无法计算区域平均概率。")
        return (probabilities * mask).sum() / mask_sum

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        """将每个样本的 CAM 独立归一化到 0-1。"""
        batch_size = cam.size(0)
        flat = cam.view(batch_size, -1)
        cam_min = flat.min(dim=1).values.view(batch_size, 1, 1, 1)
        cam_max = flat.max(dim=1).values.view(batch_size, 1, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def _clear_cached_tensors(self) -> None:
        """清空上一次 CAM 计算保存的 feature map 和 gradient。"""
        self.feature_map = None
        self.gradient = None

    def _zero_model_grad(self) -> None:
        """清理模型梯度，避免 CAM 反向传播影响后续推理或训练。"""
        self.model.zero_grad(set_to_none=True)

    def generate(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
        target_mask: torch.Tensor | np.ndarray | None = None,
    ) -> np.ndarray:
        """生成二分类语义分割 Grad-CAM 热力图。

        Args:
            rgb: RGB 输入，形状为 `(B, 3, H, W)`。
            sar: SAR/VV-VH 输入，形状为 `(B, 2, H, W)`。
            target_mask: 可选目标区域 mask，形状为 `(B,H,W)` 或 `(B,1,H,W)`。

        Returns:
            numpy.ndarray: 归一化后的 CAM，形状为 `(B, H, W)`，数值范围为 0-1。
        """
        if rgb.dim() != 4 or sar.dim() != 4:
            raise ValueError("rgb 和 sar 都必须是 4D Tensor，形状分别为 (B,3,H,W) 和 (B,2,H,W)。")
        if rgb.size(0) != sar.size(0) or rgb.shape[-2:] != sar.shape[-2:]:
            raise ValueError(
                "rgb 和 sar 必须具有相同 batch 大小和空间尺寸，"
                f"实际形状为 {tuple(rgb.shape)} 和 {tuple(sar.shape)}。"
            )

        self._clear_cached_tensors()
        self._zero_model_grad()

        try:
            logits = self.model(rgb, sar)
            if logits.dim() != 4 or logits.size(1) != 1:
                raise ValueError(
                    "模型输出 logits 必须是形状为 (B,1,H,W) 的 4D Tensor，"
                    f"实际形状为 {tuple(logits.shape)}。"
                )

            probabilities = torch.sigmoid(logits)
            backward_target = self._build_backward_target(probabilities, target_mask)
            backward_target.backward()

            if self.feature_map is None:
                raise RuntimeError("未捕获到目标层 feature map，请确认目标层参与了本次前向传播。")
            if self.gradient is None:
                raise RuntimeError("未捕获到目标层 gradient，请确认目标层参与了反向传播。")

            weights = self.gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.feature_map).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            cam = self._normalize_cam(cam)
            return cam.squeeze(1).detach().cpu().numpy().astype(np.float32)
        finally:
            self._zero_model_grad()
            self._clear_cached_tensors()

    def remove_hooks(self) -> None:
        """移除已注册的 hook，释放引用，避免重复注册或内存占用。"""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle is not None:
            self.backward_handle.remove()
            self.backward_handle = None
