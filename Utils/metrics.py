import numpy as np
import torch

from configs.train_config import TrainConfig
from Utils.loss import SoftSkeletonize


def binary_segmentation_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float | None = None) -> dict:
    """计算二分类语义分割常用评价指标。

    输入 logits 会先经过 sigmoid 转成概率，再按照 threshold 转成 0/1 预测图。
    不显式传入 threshold 时，默认使用 TrainConfig.threshold，避免与训练配置不一致。
    返回 precision、recall、f1、目标类 IoU、背景类 IoU、mIoU、clDice、betti0_error 和 betti1_error。
    """
    if threshold is None:
        threshold = TrainConfig.threshold

    probs = torch.sigmoid(logits)
    pred_mask = (probs >= threshold).detach().float().cpu()
    truth_mask = (labels.detach() >= 0.5).float().cpu()
    preds = pred_mask.numpy().astype(np.uint8).reshape(-1)
    truth = truth_mask.numpy().astype(np.uint8).reshape(-1)

    # 根据预测结果和真实标签统计混淆矩阵中的 TP、TN、FP、FN。
    tp = int(np.sum((preds == 1) & (truth == 1)))
    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))
    eps = 1e-7

    # 根据混淆矩阵计算目标类、背景类和整体二分类分割指标。
    target_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    # 农田防护林前景占比较低，accuracy 容易被大量背景像素主导，训练评价中暂不计算。
    # accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    cldice = cldice_metric(pred_mask, truth_mask, iterations=3, smooth=1.0, eps=eps)
    betti_metrics = betti_number_error(pred_mask, truth_mask)

    metrics = {
        # "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_iou": target_iou,
        "background_iou": background_iou,
        "miou": (target_iou + background_iou) / 2.0,
        "cldice": cldice,
    }
    metrics.update(betti_metrics)
    return metrics


def cldice_metric(
    pred_mask: torch.Tensor,
    truth_mask: torch.Tensor,
    iterations: int = 3,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> float:
    """计算 clDice 拓扑指标，适合评价农田防护林这类细长连通目标。

    pred_mask 和 truth_mask 应为 0/1 二值掩膜。该实现对标 soft-clDice 论文中的
    topology precision 和 topology sensitivity：预测骨架落在真值区域内的比例、
    真值骨架落在预测区域内的比例，以及二者的调和平均。
    """
    if pred_mask.shape != truth_mask.shape:
        raise ValueError(f"pred_mask and truth_mask must have the same shape, got {pred_mask.shape} and {truth_mask.shape}.")

    pred_mask = pred_mask.float().clamp(0.0, 1.0)
    truth_mask = truth_mask.float().clamp(0.0, 1.0)
    skeletonize = SoftSkeletonize(iterations=iterations)
    pred_skeleton = skeletonize(pred_mask)
    truth_skeleton = skeletonize(truth_mask)

    tprec = (torch.sum(pred_skeleton * truth_mask) + smooth) / (torch.sum(pred_skeleton) + smooth + eps)
    tsens = (torch.sum(truth_skeleton * pred_mask) + smooth) / (torch.sum(truth_skeleton) + smooth + eps)
    score = (2.0 * tprec * tsens) / (tprec + tsens + eps)
    return float(score.item())


def betti_number_error(pred_mask: torch.Tensor, truth_mask: torch.Tensor) -> dict:
    """计算 2D 二值掩膜的 Betti number error。

    beta0 表示前景连通分量数量，beta1 表示前景孔洞数量。这里对每张图分别计算
    |beta0_pred - beta0_truth| 和 |beta1_pred - beta1_truth|，再对 batch 求平均。
    betti_error 为 beta0_error + beta1_error；训练评价中暂不返回总和，只保留 beta0/beta1 细项。
    """
    pred_array = _as_4d_binary_array(pred_mask)
    truth_array = _as_4d_binary_array(truth_mask)
    if pred_array.shape != truth_array.shape:
        raise ValueError(f"pred_mask and truth_mask must have the same shape, got {pred_array.shape} and {truth_array.shape}.")

    beta0_errors = []
    beta1_errors = []
    for sample_index in range(pred_array.shape[0]):
        for channel_index in range(pred_array.shape[1]):
            pred_slice = pred_array[sample_index, channel_index]
            truth_slice = truth_array[sample_index, channel_index]
            pred_beta0, pred_beta1 = _betti_numbers_2d(pred_slice)
            truth_beta0, truth_beta1 = _betti_numbers_2d(truth_slice)
            beta0_errors.append(abs(pred_beta0 - truth_beta0))
            beta1_errors.append(abs(pred_beta1 - truth_beta1))

    beta0_error = float(np.mean(beta0_errors)) if beta0_errors else 0.0
    beta1_error = float(np.mean(beta1_errors)) if beta1_errors else 0.0
    return {
        "betti0_error": beta0_error,
        "betti1_error": beta1_error,
        # 暂不输出 Betti 总误差，避免训练日志和结果表重复展示 beta0/beta1 的简单求和。
        # "betti_error": beta0_error + beta1_error,
    }


def _as_4d_binary_array(mask: torch.Tensor) -> np.ndarray:
    """把 2D/3D/4D 掩膜统一成 N,C,H,W 的 0/1 numpy 数组。"""
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    array = (array >= 0.5).astype(np.uint8)
    if array.ndim == 2:
        return array[None, None, :, :]
    if array.ndim == 3:
        return array[:, None, :, :]
    if array.ndim == 4:
        return array
    raise ValueError(f"Betti number error expects 2D, 3D, or 4D masks, got {array.ndim}D.")


def _betti_numbers_2d(mask: np.ndarray) -> tuple[int, int]:
    """返回 2D 二值前景的 beta0 和 beta1。"""
    beta0 = _count_components_2d(mask, target_value=1, connectivity=8, count_border_components=True)
    beta1 = _count_components_2d(mask, target_value=0, connectivity=4, count_border_components=False)
    return beta0, beta1


def _count_components_2d(
    mask: np.ndarray,
    *,
    target_value: int,
    connectivity: int,
    count_border_components: bool,
) -> int:
    """统计 2D 掩膜中指定像素值的连通分量数量。"""
    if mask.ndim != 2:
        raise ValueError(f"_count_components_2d expects a 2D mask, got {mask.ndim}D.")
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        raise ValueError("connectivity must be 4 or 8.")

    height, width = mask.shape
    target = mask == target_value
    visited = np.zeros_like(target, dtype=bool)
    count = 0

    for row in range(height):
        for col in range(width):
            if not target[row, col] or visited[row, col]:
                continue

            touches_border = False
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                if current_row in {0, height - 1} or current_col in {0, width - 1}:
                    touches_border = True
                for row_offset, col_offset in offsets:
                    next_row = current_row + row_offset
                    next_col = current_col + col_offset
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row, next_col] or not target[next_row, next_col]:
                        continue
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            if count_border_components or not touches_border:
                count += 1

    return count


def average_metric_dicts(metric_dicts: list[dict]) -> dict:
    """对多个 batch 或多张图像的指标字典逐项求平均。"""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([item[key] for item in metric_dicts])) for key in keys}
