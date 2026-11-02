"""Data-dependent upsampling implementations for semantic segmentation.

This module keeps two historically distinct implementations separate:

``OpenSourceDUpsampling``
    Reproduces the public, unofficial PyTorch repository's DUpsampling layer:
    a trainable 1x1 convolution followed by its original depth-to-space tensor
    rearrangement.

``PaperDUpsampling``
    Implements the workflow described in *Decoders Matter for Semantic
    Segmentation: Data-Dependent Decoding Enables Flexible Feature
    Aggregation* (CVPR 2019).  ``LabelPCACodec`` fits the label-space PCA
    projection/reconstruction matrices, and ``PaperDUpsampling`` installs the
    precomputed reconstruction matrix as a fixed 1x1 convolution before
    depth-to-space reconstruction.  ``AdaptiveTemperatureSoftmax`` provides
    the paper's learnable-temperature classification step.

The unofficial repository does not contain the paper's PCA fitting or
adaptive-temperature workflow, so the two paths are intentionally exposed as
different classes instead of being presented as one numerically identical
implementation.

Paper: https://arxiv.org/abs/1903.02120
Reference repository: https://github.com/haochange/DUpsampling
Reference commit inspected: a187e3d093f5c5a85f18c98edd9dc92cc46dbe4f
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "AdaptiveTemperatureSoftmax",
    "DUpsampling",
    "LabelPCACodec",
    "OpenSourceDUpsampling",
    "PaperDUpsampling",
]


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _label_batches(
    labels: torch.Tensor | Iterable[torch.Tensor],
) -> Iterator[torch.Tensor]:
    if isinstance(labels, torch.Tensor):
        yield labels
        return

    if not isinstance(labels, Iterable):
        raise TypeError("labels must be a tensor or an iterable of tensors.")

    for batch in labels:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("Every label batch must be a torch.Tensor.")
        yield batch


def _normalize_label_shape(
    labels: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, bool]:
    """Return ``(labels, is_one_hot)`` with an explicit batch dimension."""

    if labels.ndim == 2:
        return labels.unsqueeze(0), False
    if labels.ndim == 3:
        return labels, False
    if labels.ndim != 4:
        raise ValueError(
            "Labels must have shape (H,W), (B,H,W), (B,1,H,W), or "
            f"(B,C,H,W); got {tuple(labels.shape)}."
        )
    if labels.shape[1] == 1:
        return labels[:, 0], False
    if labels.shape[1] == num_classes:
        return labels, True
    raise ValueError(
        "Four-dimensional labels must have one channel or num_classes "
        f"channels; got {labels.shape[1]} channels for {num_classes} classes."
    )


def _pad_one_hot(
    one_hot: torch.Tensor,
    valid: torch.Tensor,
    scale_factor: int,
    pad_class: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = one_hot.shape[-2:]
    pad_height = (-height) % scale_factor
    pad_width = (-width) % scale_factor
    if pad_height == 0 and pad_width == 0:
        return one_hot, valid

    padded_height = height + pad_height
    padded_width = width + pad_width
    padded = one_hot.new_zeros(
        one_hot.shape[0],
        one_hot.shape[1],
        padded_height,
        padded_width,
    )
    padded[:, pad_class] = 1
    padded[..., :height, :width] = one_hot

    padded_valid = torch.ones(
        valid.shape[0],
        1,
        padded_height,
        padded_width,
        dtype=torch.bool,
        device=valid.device,
    )
    padded_valid[..., :height, :width] = valid
    return padded, padded_valid


def _label_patch_vectors(
    labels: torch.Tensor,
    *,
    num_classes: int,
    scale_factor: int,
    pad_class: int,
    ignore_index: int | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    labels, is_one_hot = _normalize_label_shape(labels, num_classes)
    original_size = (int(labels.shape[-2]), int(labels.shape[-1]))

    if is_one_hot:
        if not torch.is_floating_point(labels):
            labels = labels.float()
        if not torch.isfinite(labels).all():
            raise ValueError("One-hot labels must contain only finite values.")
        one_hot = labels.to(dtype=dtype)
        valid = torch.ones(
            labels.shape[0],
            1,
            labels.shape[-2],
            labels.shape[-1],
            dtype=torch.bool,
            device=labels.device,
        )
    else:
        if torch.is_floating_point(labels):
            rounded = labels.round()
            if not torch.equal(labels, rounded):
                raise ValueError("Class-index labels must contain integer values.")
            labels = rounded
        class_indices = labels.long()
        valid = torch.ones_like(class_indices, dtype=torch.bool)
        if ignore_index is not None:
            valid = class_indices != int(ignore_index)

        valid_values = class_indices[valid]
        if valid_values.numel() > 0:
            minimum = int(valid_values.min())
            maximum = int(valid_values.max())
            if minimum < 0 or maximum >= num_classes:
                raise ValueError(
                    "Class-index labels outside the ignored region must be in "
                    f"[0, {num_classes - 1}], got range [{minimum}, {maximum}]."
                )

        safe_indices = torch.where(
            valid,
            class_indices,
            torch.full_like(class_indices, pad_class),
        )
        one_hot = F.one_hot(safe_indices, num_classes=num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=dtype)
        valid = valid.unsqueeze(1)

    one_hot, valid = _pad_one_hot(
        one_hot,
        valid,
        scale_factor=scale_factor,
        pad_class=pad_class,
    )
    patch_vectors = F.pixel_unshuffle(one_hot, scale_factor)
    patch_validity = F.pixel_unshuffle(valid.to(dtype=torch.uint8), scale_factor)
    patch_validity = patch_validity.bool().all(dim=1)
    return patch_vectors, patch_validity, original_size


class OpenSourceDUpsampling(nn.Module):
    """Behavior-compatible version of the public unofficial implementation.

    The argument names intentionally match the repository class.  Its 1x1
    convolution is trainable and is not initialized from the paper's label PCA
    reconstruction matrix.
    """

    def __init__(
        self,
        inplanes: int,
        scale: int,
        num_class: int = 21,
        pad: int = 0,
    ) -> None:
        super().__init__()
        self.inplanes = _positive_int(inplanes, "inplanes")
        self.scale = _positive_int(scale, "scale")
        self.num_class = _positive_int(num_class, "num_class")
        pad = int(pad)
        if pad < 0:
            raise ValueError(f"pad must be non-negative, got {pad}.")

        self.conv1 = nn.Conv2d(
            self.inplanes,
            self.num_class * self.scale * self.scale,
            kernel_size=1,
            padding=pad,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (B,C,H,W), got {tuple(x.shape)}.")
        if x.shape[1] != self.inplanes:
            raise ValueError(
                f"Expected {self.inplanes} input channels, got {x.shape[1]}."
            )

        x = self.conv1(x)
        batch, channels, height, width = x.shape

        # Preserve the public repository's explicit tensor-rearrangement order.
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(
            batch,
            height,
            width * self.scale,
            channels // self.scale,
        )
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(
            batch,
            width * self.scale,
            height * self.scale,
            channels // (self.scale * self.scale),
        )
        return x.permute(0, 3, 2, 1)


class LabelPCACodec(nn.Module):
    """PCA compression/reconstruction model fitted from segmentation labels.

    Each ``scale_factor x scale_factor`` one-hot label window is flattened into
    a vector of length ``num_classes * scale_factor**2``.  PCA supplies the
    compression matrix ``P`` and, under the paper's orthogonality constraint,
    the inverse/reconstruction matrix ``W = P.T``.

    ``fit`` accepts either one label tensor or an iterable of label batches and
    accumulates only first and second moments, so the complete training label
    set does not need to be stacked in memory.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        scale_factor: int,
        projection: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        explained_variance: torch.Tensor,
        explained_variance_ratio: torch.Tensor,
        sample_count: int,
        pad_class: int = 0,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.scale_factor = _positive_int(scale_factor, "scale_factor")
        self.pad_class = int(pad_class)
        if not 0 <= self.pad_class < self.num_classes:
            raise ValueError(
                f"pad_class must be in [0, {self.num_classes - 1}], "
                f"got {self.pad_class}."
            )
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.sample_count = _positive_int(sample_count, "sample_count")

        vector_dim = self.num_classes * self.scale_factor * self.scale_factor
        if projection.ndim != 2 or projection.shape[1] != vector_dim:
            raise ValueError(
                "projection must have shape (compressed_channels, "
                f"{vector_dim}), got {tuple(projection.shape)}."
            )
        compressed_channels = int(projection.shape[0])
        if reconstruction.shape != (vector_dim, compressed_channels):
            raise ValueError(
                "reconstruction must have shape "
                f"({vector_dim}, {compressed_channels}), got "
                f"{tuple(reconstruction.shape)}."
            )
        if mean.shape != (vector_dim,):
            raise ValueError(
                f"mean must have shape ({vector_dim},), got {tuple(mean.shape)}."
            )
        if explained_variance.shape != (compressed_channels,):
            raise ValueError("explained_variance has an incompatible shape.")
        if explained_variance_ratio.shape != (compressed_channels,):
            raise ValueError("explained_variance_ratio has an incompatible shape.")

        self.compressed_channels = compressed_channels
        self.vector_dim = vector_dim
        self.register_buffer("projection", projection.detach().clone())
        self.register_buffer("reconstruction", reconstruction.detach().clone())
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer(
            "explained_variance",
            explained_variance.detach().clone(),
        )
        self.register_buffer(
            "explained_variance_ratio",
            explained_variance_ratio.detach().clone(),
        )

    @classmethod
    def fit(
        cls,
        labels: torch.Tensor | Iterable[torch.Tensor],
        *,
        num_classes: int,
        scale_factor: int,
        compressed_channels: int,
        pad_class: int = 0,
        ignore_index: int | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> "LabelPCACodec":
        num_classes = _positive_int(num_classes, "num_classes")
        scale_factor = _positive_int(scale_factor, "scale_factor")
        compressed_channels = _positive_int(
            compressed_channels,
            "compressed_channels",
        )
        pad_class = int(pad_class)
        if not 0 <= pad_class < num_classes:
            raise ValueError(
                f"pad_class must be in [0, {num_classes - 1}], got {pad_class}."
            )

        vector_dim = num_classes * scale_factor * scale_factor
        if compressed_channels > vector_dim:
            raise ValueError(
                "compressed_channels cannot exceed the flattened label-window "
                f"dimension {vector_dim}; got {compressed_channels}."
            )
        if not torch.empty((), dtype=output_dtype).is_floating_point():
            raise TypeError("output_dtype must be a floating-point torch dtype.")

        sample_count = 0
        vector_sum = torch.zeros(vector_dim, dtype=torch.float64)
        gram_sum = torch.zeros(vector_dim, vector_dim, dtype=torch.float64)
        saw_batch = False

        for label_batch in _label_batches(labels):
            saw_batch = True
            patch_vectors, patch_validity, _ = _label_patch_vectors(
                label_batch,
                num_classes=num_classes,
                scale_factor=scale_factor,
                pad_class=pad_class,
                ignore_index=ignore_index,
                dtype=torch.float64,
            )
            rows = patch_vectors.permute(0, 2, 3, 1).reshape(-1, vector_dim)
            valid_rows = patch_validity.reshape(-1)
            rows = rows[valid_rows].detach().to(device="cpu", dtype=torch.float64)
            if rows.numel() == 0:
                continue

            sample_count += int(rows.shape[0])
            vector_sum += rows.sum(dim=0)
            gram_sum += rows.transpose(0, 1).matmul(rows)

        if not saw_batch:
            raise ValueError("labels iterable is empty.")
        if sample_count == 0:
            raise ValueError("No valid label windows were available for PCA fitting.")

        mean = vector_sum / sample_count
        centered_gram = gram_sum - sample_count * torch.outer(mean, mean)
        centered_gram = (centered_gram + centered_gram.transpose(0, 1)) * 0.5

        eigenvalues, eigenvectors = torch.linalg.eigh(centered_gram)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        eigenvectors = eigenvectors[:, order]

        projection = eigenvectors[:, :compressed_channels].transpose(0, 1)
        reconstruction = projection.transpose(0, 1)
        denominator = max(sample_count - 1, 1)
        explained_variance = eigenvalues[:compressed_channels] / denominator
        total_variance = eigenvalues.sum()
        if float(total_variance) > 0:
            explained_variance_ratio = (
                eigenvalues[:compressed_channels] / total_variance
            )
        else:
            explained_variance_ratio = torch.zeros_like(explained_variance)

        return cls(
            num_classes=num_classes,
            scale_factor=scale_factor,
            projection=projection.to(dtype=output_dtype),
            reconstruction=reconstruction.to(dtype=output_dtype),
            mean=mean.to(dtype=output_dtype),
            explained_variance=explained_variance.to(dtype=output_dtype),
            explained_variance_ratio=explained_variance_ratio.to(
                dtype=output_dtype
            ),
            sample_count=sample_count,
            pad_class=pad_class,
            ignore_index=ignore_index,
        )

    def encode(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.device != self.projection.device:
            raise ValueError(
                "labels and LabelPCACodec must be on the same device; "
                f"got {labels.device} and {self.projection.device}."
            )
        patch_vectors, _, _ = _label_patch_vectors(
            labels,
            num_classes=self.num_classes,
            scale_factor=self.scale_factor,
            pad_class=self.pad_class,
            ignore_index=self.ignore_index,
            dtype=self.projection.dtype,
        )
        centered = patch_vectors - self.mean.view(1, -1, 1, 1)
        return F.conv2d(centered, self.projection[:, :, None, None])

    def decode(
        self,
        codes: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if codes.ndim != 4:
            raise ValueError(
                f"codes must have shape (B,C,H,W), got {tuple(codes.shape)}."
            )
        if codes.shape[1] != self.compressed_channels:
            raise ValueError(
                f"Expected {self.compressed_channels} code channels, "
                f"got {codes.shape[1]}."
            )
        if codes.device != self.reconstruction.device:
            raise ValueError(
                "codes and LabelPCACodec must be on the same device; "
                f"got {codes.device} and {self.reconstruction.device}."
            )
        if codes.dtype != self.reconstruction.dtype:
            raise ValueError(
                "codes and LabelPCACodec must have the same dtype; "
                f"got {codes.dtype} and {self.reconstruction.dtype}."
            )

        flattened = F.conv2d(
            codes,
            self.reconstruction[:, :, None, None],
            bias=self.mean,
        )
        decoded = F.pixel_shuffle(flattened, self.scale_factor)
        return _crop_to_output_size(decoded, output_size)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, scale_factor={self.scale_factor}, "
            f"compressed_channels={self.compressed_channels}, "
            f"sample_count={self.sample_count}"
        )


def _crop_to_output_size(
    tensor: torch.Tensor,
    output_size: tuple[int, int] | None,
) -> torch.Tensor:
    if output_size is None:
        return tensor
    if len(output_size) != 2:
        raise ValueError("output_size must contain exactly (height, width).")
    height = _positive_int(output_size[0], "output height")
    width = _positive_int(output_size[1], "output width")
    if height > tensor.shape[-2] or width > tensor.shape[-1]:
        raise ValueError(
            f"output_size {(height, width)} exceeds decoded size "
            f"{tuple(tensor.shape[-2:])}."
        )
    return tensor[..., :height, :width]


class AdaptiveTemperatureSoftmax(nn.Module):
    """Learnable-temperature softmax used by the paper after DUpsampling.

    Temperature is stored in log-space so it remains positive during gradient
    updates while retaining the paper's initialization ``T = 1``.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        initial_temperature: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = _positive_int(num_classes, "num_classes")
        initial_temperature = float(initial_temperature)
        if not math.isfinite(initial_temperature) or initial_temperature <= 0:
            raise ValueError(
                "initial_temperature must be finite and positive, got "
                f"{initial_temperature}."
            )
        value = torch.tensor(math.log(initial_temperature), dtype=torch.float32)
        if learnable:
            self.log_temperature = nn.Parameter(value)
        else:
            self.register_buffer("log_temperature", value)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim < 2:
            raise ValueError("logits must include batch and class dimensions.")
        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} class channels, got {logits.shape[1]}."
            )
        return logits / self.temperature.to(device=logits.device, dtype=logits.dtype)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.scale_logits(logits), dim=1)


class PaperDUpsampling(nn.Module):
    """Paper-faithful fixed linear reconstruction plus depth-to-space.

    ``reconstruction_weight`` is the precomputed matrix ``W`` with shape
    ``(num_classes * scale_factor**2, code_channels)``.  The reconstruction
    bias is the training-label mean subtracted before PCA compression.
    """

    def __init__(
        self,
        *,
        code_channels: int,
        scale_factor: int,
        num_classes: int,
        reconstruction_weight: torch.Tensor,
        reconstruction_bias: torch.Tensor,
        trainable_reconstruction: bool = False,
        initial_temperature: float = 1.0,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        self.code_channels = _positive_int(code_channels, "code_channels")
        self.scale_factor = _positive_int(scale_factor, "scale_factor")
        self.num_classes = _positive_int(num_classes, "num_classes")
        output_channels = (
            self.num_classes * self.scale_factor * self.scale_factor
        )

        if reconstruction_weight.ndim == 2:
            reconstruction_weight = reconstruction_weight[:, :, None, None]
        expected_weight_shape = (output_channels, self.code_channels, 1, 1)
        if tuple(reconstruction_weight.shape) != expected_weight_shape:
            raise ValueError(
                f"reconstruction_weight must have shape {expected_weight_shape}, "
                f"got {tuple(reconstruction_weight.shape)}."
            )
        if tuple(reconstruction_bias.shape) != (output_channels,):
            raise ValueError(
                f"reconstruction_bias must have shape ({output_channels},), "
                f"got {tuple(reconstruction_bias.shape)}."
            )
        if reconstruction_weight.dtype != reconstruction_bias.dtype:
            raise ValueError(
                "reconstruction_weight and reconstruction_bias must have the "
                "same dtype."
            )

        self.reconstruction = nn.Conv2d(
            self.code_channels,
            output_channels,
            kernel_size=1,
            bias=True,
            dtype=reconstruction_weight.dtype,
            device=reconstruction_weight.device,
        )
        with torch.no_grad():
            self.reconstruction.weight.copy_(reconstruction_weight)
            self.reconstruction.bias.copy_(
                reconstruction_bias.to(device=reconstruction_weight.device)
            )
        self.reconstruction.weight.requires_grad_(trainable_reconstruction)
        self.reconstruction.bias.requires_grad_(trainable_reconstruction)

        self.adaptive_softmax = AdaptiveTemperatureSoftmax(
            self.num_classes,
            initial_temperature=initial_temperature,
            learnable=learnable_temperature,
        ).to(device=reconstruction_weight.device, dtype=reconstruction_weight.dtype)

    @classmethod
    def from_codec(
        cls,
        codec: LabelPCACodec,
        *,
        trainable_reconstruction: bool = False,
        initial_temperature: float = 1.0,
        learnable_temperature: bool = True,
    ) -> "PaperDUpsampling":
        if not isinstance(codec, LabelPCACodec):
            raise TypeError("codec must be a LabelPCACodec instance.")
        return cls(
            code_channels=codec.compressed_channels,
            scale_factor=codec.scale_factor,
            num_classes=codec.num_classes,
            reconstruction_weight=codec.reconstruction,
            reconstruction_bias=codec.mean,
            trainable_reconstruction=trainable_reconstruction,
            initial_temperature=initial_temperature,
            learnable_temperature=learnable_temperature,
        )

    def forward(
        self,
        codes: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if codes.ndim != 4:
            raise ValueError(
                f"codes must have shape (B,C,H,W), got {tuple(codes.shape)}."
            )
        if codes.shape[1] != self.code_channels:
            raise ValueError(
                f"Expected {self.code_channels} code channels, got {codes.shape[1]}."
            )
        flattened = self.reconstruction(codes)
        decoded = F.pixel_shuffle(flattened, self.scale_factor)
        return _crop_to_output_size(decoded, output_size)

    def temperature_scaled_logits(
        self,
        codes: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        decoded = self(codes, output_size=output_size)
        return self.adaptive_softmax.scale_logits(decoded)

    def probabilities(
        self,
        codes: torch.Tensor,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        decoded = self(codes, output_size=output_size)
        return self.adaptive_softmax(decoded)

    def extra_repr(self) -> str:
        return (
            f"code_channels={self.code_channels}, "
            f"num_classes={self.num_classes}, "
            f"scale_factor={self.scale_factor}"
        )


# The selected BiMorphNet candidate uses the public repository's directly trainable
# 1x1-convolution + depth-to-space implementation.  The complete PCA-based
# paper workflow remains available explicitly as ``PaperDUpsampling``.
DUpsampling = OpenSourceDUpsampling
