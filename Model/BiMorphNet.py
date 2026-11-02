from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .KDRM import KDRM
from .Dupsample import DUpsampling
from .FADE import CascadedFADE, FADEGateFusion


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        padding = dilation if kernel_size == 3 else 0
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


@dataclass(frozen=True)
class BackboneSpec:
    layers: tuple[int, int, int, int]
    stem_channels: int
    planes: tuple[int, int, int, int]


BACKBONES: dict[str, BackboneSpec] = {
    "tiny": BackboneSpec(layers=(1, 1, 1, 1), stem_channels=32, planes=(16, 32, 64, 128)),
    "resnet50": BackboneSpec(layers=(3, 4, 6, 3), stem_channels=64, planes=(64, 128, 256, 512)),
    "resnet101": BackboneSpec(layers=(3, 4, 23, 3), stem_channels=64, planes=(64, 128, 256, 512)),
}


class ResidualBackbone(nn.Module):
    def __init__(self, in_channels: int, spec: BackboneSpec) -> None:
        super().__init__()
        self.inplanes = spec.stem_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            spec.stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(spec.stem_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(spec.planes[0], spec.layers[0])
        self.layer2 = self._make_layer(spec.planes[1], spec.layers[1], stride=2)
        self.layer3 = self._make_layer(spec.planes[2], spec.layers[2], stride=2)
        self.layer4 = self._make_layer(spec.planes[3], spec.layers[3], stride=2)
        self.stage_channels = tuple(plane * Bottleneck.expansion for plane in spec.planes)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        out_channels = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = [Bottleneck(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        low = self.layer1(x)
        mid1 = self.layer2(low)
        mid2 = self.layer3(mid1)
        high = self.layer4(mid2)
        return low, mid1, mid2, high


class RCGI(nn.Module):
    """Multi-modal cross attention using V/Q/K projections."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.opt_value = conv1x1(channels, channels)
        self.opt_query = conv1x1(channels, channels)
        self.opt_key = conv1x1(channels, channels)
        self.sar_value = conv1x1(channels, channels)
        self.sar_query = conv1x1(channels, channels)
        self.sar_key = conv1x1(channels, channels)
        self.project = ConvBNReLU(channels, channels, kernel_size=1)

    def forward(self, sar_features: torch.Tensor, optical_features: torch.Tensor) -> torch.Tensor:
        if sar_features.shape != optical_features.shape:
            raise ValueError(
                "RCGI expects aligned SAR and optical feature maps with the same shape, "
                f"got {tuple(sar_features.shape)} and {tuple(optical_features.shape)}."
            )

        opt_gate = torch.sigmoid(self.opt_query(optical_features) * self.sar_key(sar_features))
        sar_gate = torch.sigmoid(self.sar_query(sar_features) * self.opt_key(optical_features))
        opt_attention = self.opt_value(optical_features) * opt_gate
        sar_attention = self.sar_value(sar_features) * sar_gate
        return self.project(opt_attention + sar_attention)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: Sequence[int] = (6, 12, 18),
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [ConvBNReLU(in_channels, out_channels, kernel_size=1)]
        modules.extend(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, dilation=rate)
            for rate in atrous_rates
        )
        modules.append(ASPPPooling(in_channels, out_channels))
        self.branches = nn.ModuleList(modules)
        self.project = ConvBNReLU(
            len(modules) * out_channels,
            out_channels,
            kernel_size=1,
            dropout=0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class BiMorphEncoder(nn.Module):
    def __init__(
        self,
        rgb_channels: int,
        sar_channels: int,
        backbone: str,
        decoder_channels: int,
        low_project_channels: int,
        aspp_rates: Sequence[int],
    ) -> None:
        super().__init__()
        normalized_backbone = backbone.lower()
        if normalized_backbone not in BACKBONES:
            supported = ", ".join(sorted(BACKBONES))
            raise ValueError(f"Unsupported BiMorphNet backbone {backbone!r}. Choose from: {supported}.")

        spec = BACKBONES[normalized_backbone]
        self.rgb_backbone = ResidualBackbone(rgb_channels, spec)
        self.sar_backbone = ResidualBackbone(sar_channels, spec)
        low_channels = self.rgb_backbone.stage_channels[0]
        high_channels = self.rgb_backbone.stage_channels[3]

        self.low_rcgi = RCGI(low_channels)
        mid1_channels = self.rgb_backbone.stage_channels[1]
        mid2_channels = self.rgb_backbone.stage_channels[2]
        self.mid1_rcgi = RCGI(mid1_channels)
        self.mid2_rcgi = RCGI(mid2_channels)
        self.high_rcgi = RCGI(high_channels)
        self.high_reduce_rgb = nn.Sequential(
            conv1x1(high_channels, decoder_channels),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.high_reduce_sar = nn.Sequential(
            conv1x1(high_channels, decoder_channels),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.low_project = ConvBNReLU(
            low_channels * 3,
            low_project_channels,
            kernel_size=1,
        )
        self.aspp = ASPP(
            in_channels=high_channels + decoder_channels * 2,
            out_channels=decoder_channels,
            atrous_rates=aspp_rates,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb_low, rgb_mid1, rgb_mid2, rgb_high = self.rgb_backbone(rgb)
        sar_low, sar_mid1, sar_mid2, sar_high = self.sar_backbone(sar)

        # Layer1-Layer4 cross-modal fusion features.
        hcfr_fc1 = self.low_rcgi(sar_low, rgb_low)
        hcfr_fc2 = self.mid1_rcgi(sar_mid1, rgb_mid1)
        hcfr_fc3 = self.mid2_rcgi(sar_mid2, rgb_mid2)
        hcfr_fc4 = self.high_rcgi(sar_high, rgb_high)

        # Original low-level BiMorphNet path.
        low_features = torch.cat([rgb_low, sar_low, hcfr_fc1], dim=1)
        low_features = self.low_project(low_features)

        # Original high-level BiMorphNet path.
        high_features = torch.cat(
            [
                self.high_reduce_rgb(rgb_high),
                self.high_reduce_sar(sar_high),
                hcfr_fc4,
            ],
            dim=1,
        )
        high_features = self.aspp(high_features)
        high_features = F.interpolate(
            high_features,
            size=low_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        decoder_features = torch.cat([high_features, low_features], dim=1)
        return decoder_features, hcfr_fc1, hcfr_fc2, hcfr_fc3, hcfr_fc4


class BiMorphDecoder(nn.Module):
    """BiMorphNet decoder: KDRM recalibration, HCFR relay and HDGR reconstruction.

    Data flow:
        D0_raw = DecoderConv1(Concat(F_high, F_low))
        D0_kdrm = KDRM(D0_raw)
        D0     = 0.5 * D0_raw + 0.5 * D0_kdrm
        FD4    = CascadedFADE3x(FC4_hat, D0)
        FD3    = CascadedFADE2x(FC3_hat, FD4)
        D1     = DecoderConv2(FD3)
        FD2    = CascadedFADE1x(FC2_hat, D1)
        FD1    = SameScaleFADEGate(FC1_hat, FD2)
        logits = DUpsampling(FD1)

    FC2-FC4 retain their native spatial resolutions after 1x1 Conv-BN-ReLU
    channel projection. Their 3/2/1-stage cascades use independent FADE2xBlock
    parameters and decoder-derived guidance pyramids. FC1 is fused at H/4 by
    a decoder-dependent same-scale gate.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hcfr_channels: tuple[int, int, int, int],
        decoder_channels: int = 256,
        kdrm_reduction: int = 16,
        kdrm_rff_dim: int = 8,
        kdrm_rff_gamma: float = 1.0,
        kdrm_dependency_pool_size: int | tuple[int, int] | None = 8,
        kdrm_dependency_chunk_size: int = 32,
        kdrm_dependency_mode: str = "full",
        kdrm_fusion_hidden: int = 8,
        kdrm_eps: float = 1e-5,
        kdrm_rff_seed: int = 3407,
        fade_up_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        fc1_channels, fc2_channels, fc3_channels, fc4_channels = hcfr_channels
        self.decoder_channels = int(decoder_channels)

        # Original first decoder block: high/low concatenation -> decoder_channels.
        self.conv1 = ConvBNReLU(
            in_channels,
            self.decoder_channels,
            kernel_size=3,
        )

        # Standalone KDRM is inserted immediately after the first decoder
        # 3x3 Conv+BN+ReLU. Its output is equally blended with the untouched
        # decoder feature before hierarchical FADE injection.
        self.kdrm = KDRM(
            channels=self.decoder_channels,
            reduction=kdrm_reduction,
            rff_dim=kdrm_rff_dim,
            rff_gamma=kdrm_rff_gamma,
            dependency_pool_size=kdrm_dependency_pool_size,
            dependency_chunk_size=kdrm_dependency_chunk_size,
            dependency_mode=kdrm_dependency_mode,
            fusion_hidden=kdrm_fusion_hidden,
            eps=kdrm_eps,
            rff_seed=kdrm_rff_seed,
        )

        # Original second decoder block remains between FD3 and FD2.
        self.conv2 = ConvBNReLU(
            self.decoder_channels,
            self.decoder_channels,
            kernel_size=3,
        )

        # FC1 is already 256 channels for ResNet50/101; keep it unchanged in
        # that setting. The fallback projection only preserves compatibility
        # with optional backbones whose layer1 width differs.
        self.hcfr_fc1_project: nn.Module
        if fc1_channels == self.decoder_channels:
            self.hcfr_fc1_project = nn.Identity()
        else:
            self.hcfr_fc1_project = ConvBNReLU(
                fc1_channels,
                self.decoder_channels,
                kernel_size=1,
            )

        self.hcfr_fc2_project = ConvBNReLU(
            fc2_channels,
            self.decoder_channels,
            kernel_size=1,
        )
        self.hcfr_fc3_project = ConvBNReLU(
            fc3_channels,
            self.decoder_channels,
            kernel_size=1,
        )
        self.hcfr_fc4_project = ConvBNReLU(
            fc4_channels,
            self.decoder_channels,
            kernel_size=1,
        )

        # Four decoder skip positions with independent parameters. FC4/FC3/FC2
        # use 3/2/1 successive standard 2x FADE stages; FC1 is already H/4.
        self.fade4 = CascadedFADE(
            channels=self.decoder_channels,
            stages=3,
            up_kernel_size=fade_up_kernel_size,
        )
        self.fade3 = CascadedFADE(
            channels=self.decoder_channels,
            stages=2,
            up_kernel_size=fade_up_kernel_size,
        )
        self.fade2 = CascadedFADE(
            channels=self.decoder_channels,
            stages=1,
            up_kernel_size=fade_up_kernel_size,
        )
        self.fade1 = FADEGateFusion(channels=self.decoder_channels)

        # Replace the original 1x1 classifier + final bilinear interpolation
        # with the selected public-repository DUpsampling implementation.
        # FD1 is at output stride 4, so scale=4 restores input resolution.
        self.dupsample = DUpsampling(
            inplanes=self.decoder_channels,
            scale=4,
            num_class=num_classes,
        )

    @staticmethod
    def _crop_to_output_size(
        x: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Crop stride-rounded DUpsampling output back to the input size."""

        output_height, output_width = output_size
        if x.shape[-2] < output_height or x.shape[-1] < output_width:
            raise ValueError(
                "DUpsampling output cannot be smaller than the requested size, "
                f"got {tuple(x.shape[-2:])} and {output_size}."
            )
        return x[..., :output_height, :output_width]

    def forward(
        self,
        features: torch.Tensor,
        hcfr_fc1: torch.Tensor,
        hcfr_fc2: torch.Tensor,
        hcfr_fc3: torch.Tensor,
        hcfr_fc4: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        # D0_raw: first original decoder 3x3 Conv+BN+ReLU.
        d0_raw = self.conv1(features)

        # Preserve an untouched decoder path and equally blend it with the
        # KDRM channel-recalibrated path.
        d0_kdrm = self.kdrm(d0_raw)
        d0 = 0.5 * d0_raw + 0.5 * d0_kdrm

        # HCFR: relay four native-scale RCGI features into decoder channel space.
        hcfr_fc1_hat, hcfr_fc2_hat, hcfr_fc3_hat, hcfr_fc4_hat = self.hcfr_relay(
            hcfr_fc1, hcfr_fc2, hcfr_fc3, hcfr_fc4,
        )

        # HDGR: FADE + same-scale fusion + DUpsampling reconstruction.
        # FC4/FC3/FC2 use 3/2/1 independent 2x
        # stages; FC1 uses decoder-dependent same-scale gated fusion.
        logits = self.hdgr_reconstruct(
            d0, hcfr_fc1_hat, hcfr_fc2_hat, hcfr_fc3_hat, hcfr_fc4_hat,
        )
        return self._crop_to_output_size(logits, output_size)


    def hcfr_relay(self, hcfr_fc1, hcfr_fc2, hcfr_fc3, hcfr_fc4):
        """HCFR: preserve native scales and map four RCGI outputs to decoder channels."""
        hcfr_fc4_hat = self.hcfr_fc4_project(hcfr_fc4)
        hcfr_fc3_hat = self.hcfr_fc3_project(hcfr_fc3)
        hcfr_fc2_hat = self.hcfr_fc2_project(hcfr_fc2)
        hcfr_fc1_hat = self.hcfr_fc1_project(hcfr_fc1)
        return hcfr_fc1_hat, hcfr_fc2_hat, hcfr_fc3_hat, hcfr_fc4_hat

    def hdgr_reconstruct(self, d0, hcfr_fc1_hat, hcfr_fc2_hat, hcfr_fc3_hat, hcfr_fc4_hat):
        """HDGR: cascaded FADE, same-scale FADE gating and DUpsampling prediction."""
        fd4 = self.fade4(hcfr_fc4_hat, d0)
        fd3 = self.fade3(hcfr_fc3_hat, fd4)
        d1 = self.conv2(fd3)
        fd2 = self.fade2(hcfr_fc2_hat, d1)
        fd1 = self.fade1(hcfr_fc1_hat, fd2)
        return self.dupsample(fd1)


class BiMorphNet(nn.Module):
    """BiMorphNet with hierarchical FADE skips and standalone KDRM.

    This model combines the two previously separated changes:
    1. Cross-modal RCGI fusion is extended to layer1-layer4, producing FC1-FC4,
       which are injected by hierarchical FADE fusion.
    2. KDRM is inserted after the decoder's first 3x3 Conv+BN+ReLU; its
       output is equally blended with the untouched decoder feature before
       FC4/FC3 FADE injection.

    The original dual residual backbones, low/high BiMorphNet paths, ASPP, decoder
    convolution ordering, and public ``model(rgb, sar)`` contract are retained.
    The final 1x1 classifier plus bilinear interpolation is replaced by the
    directly trainable DUpsampling output head.
    """

    def __init__(
        self,
        num_classes: int = 1,
        rgb_channels: int = 3,
        sar_channels: int = 2,
        backbone: str = "resnet101",
        decoder_channels: int = 256,
        low_project_channels: int = 48,
        aspp_rates: Sequence[int] = (6, 12, 18),
        pretrained: bool = False,
        kdrm_reduction: int = 16,
        kdrm_rff_dim: int = 8,
        kdrm_rff_gamma: float = 1.0,
        kdrm_dependency_pool_size: int | tuple[int, int] | None = 8,
        kdrm_dependency_chunk_size: int = 32,
        kdrm_dependency_mode: str = "full",
        kdrm_fusion_hidden: int = 8,
        kdrm_eps: float = 1e-5,
        kdrm_rff_seed: int = 3407,
        fade_up_kernel_size: int = 5,
    ) -> None:
        super().__init__()

        if pretrained:
            raise ValueError(
                "BiMorphNet(pretrained=True) is not enabled; "
                "use external checkpoint loading in the training script instead."
            )

        self.rgb_channels = rgb_channels
        self.sar_channels = sar_channels

        self.encoder = BiMorphEncoder(
            rgb_channels=rgb_channels,
            sar_channels=sar_channels,
            backbone=backbone,
            decoder_channels=decoder_channels,
            low_project_channels=low_project_channels,
            aspp_rates=aspp_rates,
        )

        self.decoder = BiMorphDecoder(
            in_channels=decoder_channels + low_project_channels,
            num_classes=num_classes,
            hcfr_channels=self.encoder.rgb_backbone.stage_channels,
            decoder_channels=decoder_channels,
            kdrm_reduction=kdrm_reduction,
            kdrm_rff_dim=kdrm_rff_dim,
            kdrm_rff_gamma=kdrm_rff_gamma,
            kdrm_dependency_pool_size=kdrm_dependency_pool_size,
            kdrm_dependency_chunk_size=kdrm_dependency_chunk_size,
            kdrm_dependency_mode=kdrm_dependency_mode,
            kdrm_fusion_hidden=kdrm_fusion_hidden,
            kdrm_eps=kdrm_eps,
            kdrm_rff_seed=kdrm_rff_seed,
            fade_up_kernel_size=fade_up_kernel_size,
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Accept historical checkpoint member names without changing model mathematics."""
        legacy_paths = {
            "encoder.low_mcam.": "encoder.low_rcgi.",
            "encoder.mid1_mcam.": "encoder.mid1_rcgi.",
            "encoder.mid2_mcam.": "encoder.mid2_rcgi.",
            "encoder.high_mcam.": "encoder.high_rcgi.",
            "decoder.cdm_only.": "decoder.kdrm.",
            "decoder.fc1_project.": "decoder.hcfr_fc1_project.",
            "decoder.fc2_project.": "decoder.hcfr_fc2_project.",
            "decoder.fc3_project.": "decoder.hcfr_fc3_project.",
            "decoder.fc4_project.": "decoder.hcfr_fc4_project.",
        }
        for old_path, new_path in legacy_paths.items():
            for key in list(state_dict):
                if key.startswith(prefix + old_path):
                    new_key = prefix + new_path + key[len(prefix + old_path):]
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        if rgb.dim() != 4 or sar.dim() != 4:
            raise ValueError(
                "BiMorphNet expects 4D tensors: rgb=(B,3,H,W), sar=(B,2,H,W)."
            )

        if rgb.size(1) != self.rgb_channels:
            raise ValueError(
                f"Expected RGB input with {self.rgb_channels} channels, "
                f"got {rgb.size(1)}."
            )

        if sar.size(1) != self.sar_channels:
            raise ValueError(
                f"Expected SAR input with {self.sar_channels} channels, "
                f"got {sar.size(1)}."
            )

        if rgb.shape[0] != sar.shape[0] or rgb.shape[-2:] != sar.shape[-2:]:
            raise ValueError(
                "RGB and SAR inputs must share batch size and spatial size, "
                f"got {tuple(rgb.shape)} and {tuple(sar.shape)}."
            )

        features, hcfr_fc1, hcfr_fc2, hcfr_fc3, hcfr_fc4 = self.encoder(rgb, sar)

        return self.decoder(
            features=features,
            hcfr_fc1=hcfr_fc1,
            hcfr_fc2=hcfr_fc2,
            hcfr_fc3=hcfr_fc3,
            hcfr_fc4=hcfr_fc4,
            output_size=rgb.shape[-2:],
        )


if __name__ == "__main__":
    # Lightweight sanity example. Use diagonal dependency mode here only to
    # keep this standalone smoke test inexpensive; training defaults remain full.
    torch.manual_seed(0)
    model = BiMorphNet(
        num_classes=1,
        backbone="tiny",
        decoder_channels=32,
        low_project_channels=8,
        kdrm_rff_dim=2,
        kdrm_dependency_pool_size=2,
        kdrm_dependency_chunk_size=8,
        kdrm_dependency_mode="diagonal",
        kdrm_fusion_hidden=4,
    )
    model.eval()

    rgb = torch.randn(1, 3, 64, 64)
    sar = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        output = model(rgb, sar)

    print("RGB input:", rgb.shape)
    print("SAR input:", sar.shape)
    print("Output:", output.shape)
