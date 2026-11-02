from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_numpy(array: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    """将 torch.Tensor 或 numpy.ndarray 统一转换为 CPU 上的 numpy 数组。"""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    raise TypeError(f"{name} 必须是 torch.Tensor 或 numpy.ndarray，实际类型为 {type(array)!r}。")


def _normalize_float_image(image: np.ndarray) -> np.ndarray:
    """将浮点图像整理到 0-1 范围，便于后续转换为 uint8。"""
    image = image.astype(np.float32, copy=False)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

    image_min = float(image.min())
    image_max = float(image.max())
    if image_min >= 0.0 and image_max <= 1.0:
        return image
    if image_min >= 0.0 and image_max <= 255.0:
        return image / 255.0
    if image_max <= image_min:
        return np.zeros_like(image, dtype=np.float32)
    return (image - image_min) / (image_max - image_min)


def _enhance_rgb_for_display(
    image: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    gamma: float = 0.8,
) -> np.ndarray:
    """对显示用 RGB 做分位数拉伸和 gamma 增亮，不改变模型输入。"""
    if not 0.0 <= percentile_low < percentile_high <= 100.0:
        raise ValueError(
            "percentile_low 和 percentile_high 必须满足 0 <= low < high <= 100，"
            f"实际为 {percentile_low} 和 {percentile_high}。"
        )
    if gamma <= 0:
        raise ValueError(f"gamma 必须大于 0，实际为 {gamma}。")

    enhanced = np.empty_like(image, dtype=np.float32)
    for channel_index in range(image.shape[2]):
        channel = image[:, :, channel_index]
        low = float(np.percentile(channel, percentile_low))
        high = float(np.percentile(channel, percentile_high))
        if high <= low:
            enhanced[:, :, channel_index] = np.clip(channel, 0.0, 1.0)
            continue
        stretched = (channel - low) / (high - low)
        enhanced[:, :, channel_index] = np.clip(stretched, 0.0, 1.0)

    return np.power(enhanced, gamma).astype(np.float32, copy=False)


def rgb_tensor_to_uint8(
    rgb_tensor: torch.Tensor | np.ndarray,
    enhance: bool = False,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    gamma: float = 0.8,
) -> np.ndarray:
    """将 RGB tensor 转为可保存的 uint8 RGB 图像。

    支持输入形状 `(3,H,W)` 或 `(H,W,3)`；常见 0-1 浮点图会乘以 255，
    0-255 图像会直接裁剪到合法范围，其他数值范围会按当前图像做 min-max 拉伸。
    """
    rgb = _to_numpy(rgb_tensor, "rgb_tensor")
    if rgb.ndim != 3:
        raise ValueError(f"rgb_tensor 必须是 3D 数组，实际形状为 {tuple(rgb.shape)}。")

    if rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    elif rgb.shape[-1] != 3:
        raise ValueError(
            "rgb_tensor 的形状必须是 (3,H,W) 或 (H,W,3)，"
            f"实际形状为 {tuple(rgb.shape)}。"
        )

    rgb = _normalize_float_image(rgb)
    if enhance:
        rgb = _enhance_rgb_for_display(
            rgb,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            gamma=gamma,
        )
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _prepare_cam(cam: torch.Tensor | np.ndarray) -> np.ndarray:
    """将 CAM 整理为 0-1 范围的二维数组。"""
    cam_array = _to_numpy(cam, "cam").astype(np.float32, copy=False)
    cam_array = np.squeeze(cam_array)
    if cam_array.ndim != 2:
        raise ValueError(f"cam 必须可压缩为二维数组 (H,W)，实际形状为 {tuple(cam_array.shape)}。")
    cam_array = np.nan_to_num(cam_array, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(cam_array, 0.0, 1.0)


def cam_to_grayscale_uint8(cam: torch.Tensor | np.ndarray) -> np.ndarray:
    """将 0-1 CAM 转为 uint8 灰度图。"""
    cam_array = _prepare_cam(cam)
    return np.clip(np.rint(cam_array * 255.0), 0, 255).astype(np.uint8)


def _fallback_jet_colormap(cam_gray: np.ndarray) -> np.ndarray:
    """使用 numpy 实现简单的蓝-黄-红颜色映射，作为 cv2 不可用时的 fallback。"""
    value = cam_gray.astype(np.float32) / 255.0
    heatmap = np.zeros((*value.shape, 3), dtype=np.float32)

    low = value <= 0.5
    high = ~low

    low_ratio = value[low] / 0.5
    heatmap[low, 0] = low_ratio * 255.0
    heatmap[low, 1] = low_ratio * 255.0
    heatmap[low, 2] = (1.0 - low_ratio) * 255.0

    high_ratio = (value[high] - 0.5) / 0.5
    heatmap[high, 0] = 255.0
    heatmap[high, 1] = (1.0 - high_ratio) * 255.0
    heatmap[high, 2] = 0.0

    return np.clip(np.rint(heatmap), 0, 255).astype(np.uint8)


def cam_to_heatmap(cam: torch.Tensor | np.ndarray) -> np.ndarray:
    """将 0-1 CAM 转为彩色 RGB 热力图。

    如果当前环境安装了 cv2，则优先使用 `cv2.COLORMAP_JET`；
    如果没有 cv2，则使用 numpy 实现的蓝-黄-红映射。
    """
    cam_gray = cam_to_grayscale_uint8(cam)
    try:
        import cv2  # type: ignore

        heatmap_bgr = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        return _fallback_jet_colormap(cam_gray)


def overlay_cam_on_rgb(
    rgb: torch.Tensor | np.ndarray,
    cam: torch.Tensor | np.ndarray,
    alpha: float = 0.45,
    rgb_enhance: bool = False,
    rgb_percentile_low: float = 2.0,
    rgb_percentile_high: float = 98.0,
    rgb_gamma: float = 0.8,
) -> np.ndarray:
    """将 CAM 热力图叠加到 RGB 图像上。

    Args:
        rgb: RGB 图像，形状为 `(3,H,W)`、`(H,W,3)` 或已转换好的 uint8 RGB。
        cam: 0-1 CAM，形状为 `(H,W)` 或可 squeeze 成 `(H,W)`。
        alpha: 热力图权重，0 表示只保留 RGB，1 表示只保留热力图。
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 0-1 范围内，实际为 {alpha}。")

    rgb_uint8 = rgb_tensor_to_uint8(
        rgb,
        enhance=rgb_enhance,
        percentile_low=rgb_percentile_low,
        percentile_high=rgb_percentile_high,
        gamma=rgb_gamma,
    )
    heatmap = cam_to_heatmap(cam)
    if heatmap.shape[:2] != rgb_uint8.shape[:2]:
        raise ValueError(
            "cam 与 rgb 的空间尺寸必须一致，"
            f"实际为 cam={heatmap.shape[:2]}，rgb={rgb_uint8.shape[:2]}。"
        )

    overlay = (1.0 - alpha) * rgb_uint8.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def prediction_to_uint8(prediction: torch.Tensor | np.ndarray) -> np.ndarray:
    """将二值预测图转为 0/255 的 uint8 灰度图。

    支持 `(H,W)` 或 `(1,H,W)`；输入值可以是 0/1，也可以是 0/255。
    """
    pred = _to_numpy(prediction, "prediction")
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise ValueError(f"prediction 必须可压缩为二维数组 (H,W)，实际形状为 {tuple(pred.shape)}。")
    pred = np.nan_to_num(pred.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    return (pred > 0).astype(np.uint8) * 255


def _save_rgb(path: Path, image: np.ndarray) -> None:
    """保存 RGB 图像。"""
    Image.fromarray(image, mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    """保存单通道灰度图像。"""
    Image.fromarray(image, mode="L").save(path)


def save_cam_outputs(
    output_dir: str | Path,
    sample_name: str,
    rgb_tensor: torch.Tensor | np.ndarray,
    cam: torch.Tensor | np.ndarray,
    prediction: torch.Tensor | np.ndarray | None = None,
    label: torch.Tensor | np.ndarray | None = None,
    alpha: float = 0.45,
    rgb_enhance: bool = False,
    rgb_percentile_low: float = 2.0,
    rgb_percentile_high: float = 98.0,
    rgb_gamma: float = 0.8,
) -> dict[str, Path]:
    """保存 CAM 可视化结果。

    会保存原始 RGB 图、CAM 灰度图、CAM 彩色热力图、CAM 与 RGB 叠加图；
    如果传入 prediction 或 label，则额外保存二值预测图或标签图。文件统一保存为 png。

    Returns:
        dict[str, Path]: 每类输出对应的保存路径，便于上层脚本记录或打印。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rgb_uint8 = rgb_tensor_to_uint8(
        rgb_tensor,
        enhance=rgb_enhance,
        percentile_low=rgb_percentile_low,
        percentile_high=rgb_percentile_high,
        gamma=rgb_gamma,
    )
    cam_gray = cam_to_grayscale_uint8(cam)
    heatmap = cam_to_heatmap(cam)
    overlay = overlay_cam_on_rgb(rgb_uint8, cam, alpha=alpha)

    if cam_gray.shape != rgb_uint8.shape[:2]:
        raise ValueError(
            "cam 与 rgb 的空间尺寸必须一致，"
            f"实际为 cam={cam_gray.shape}，rgb={rgb_uint8.shape[:2]}。"
        )

    paths: dict[str, Path] = {
        "rgb": output_path / f"{sample_name}_rgb.png",
        "cam": output_path / f"{sample_name}_cam.png",
        "heatmap": output_path / f"{sample_name}_heatmap.png",
        "overlay": output_path / f"{sample_name}_overlay.png",
    }

    _save_rgb(paths["rgb"], rgb_uint8)
    _save_gray(paths["cam"], cam_gray)
    _save_rgb(paths["heatmap"], heatmap)
    _save_rgb(paths["overlay"], overlay)

    if prediction is not None:
        pred_uint8 = prediction_to_uint8(prediction)
        if pred_uint8.shape != rgb_uint8.shape[:2]:
            raise ValueError(
                "prediction 与 rgb 的空间尺寸必须一致，"
                f"实际为 prediction={pred_uint8.shape}，rgb={rgb_uint8.shape[:2]}。"
            )
        paths["prediction"] = output_path / f"{sample_name}_prediction.png"
        _save_gray(paths["prediction"], pred_uint8)

    if label is not None:
        label_uint8 = prediction_to_uint8(label)
        if label_uint8.shape != rgb_uint8.shape[:2]:
            raise ValueError(
                "label 与 rgb 的空间尺寸必须一致，"
                f"实际为 label={label_uint8.shape}，rgb={rgb_uint8.shape[:2]}。"
            )
        paths["label"] = output_path / f"{sample_name}_label.png"
        _save_gray(paths["label"], label_uint8)

    return paths


__all__ = [
    "cam_to_grayscale_uint8",
    "cam_to_heatmap",
    "overlay_cam_on_rgb",
    "prediction_to_uint8",
    "rgb_tensor_to_uint8",
    "save_cam_outputs",
]
