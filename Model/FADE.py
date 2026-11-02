# README:
# Implementation of FADE: A Task-Agnostic Upsampling Operator for Encoder-Decoder Architectures.
# Requirements:
# 1. PyTorch


import torch
from torch import nn
import torch.nn.functional as F

def xavier_init(module, distribution='uniform'):
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight)
    elif distribution == 'normal':
        nn.init.xavier_normal_(module.weight)
    else:
        raise ValueError(f"Unsupported Xavier distribution: {distribution!r}.")
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def carafe(x, masks, kernel_size, group_size, scale_factor):
    """Pure-PyTorch CARAFE implementation used by FADE."""

    if x.dim() != 4 or masks.dim() != 4:
        raise ValueError("CARAFE expects 4D feature and mask tensors.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if group_size <= 0 or x.size(1) % group_size != 0:
        raise ValueError("Feature channels must be divisible by group_size.")
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive.")

    batch, channels, height, width = x.shape
    output_height = height * scale_factor
    output_width = width * scale_factor
    kernel_elements = kernel_size ** 2
    expected_mask_shape = (
        batch,
        group_size * kernel_elements,
        output_height,
        output_width,
    )
    if tuple(masks.shape) != expected_mask_shape:
        raise ValueError(
            "CARAFE mask shape must be "
            f"{expected_mask_shape}, got {tuple(masks.shape)}."
        )

    patches = F.unfold(
        x,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
    )
    patches = patches.view(
        batch,
        channels,
        kernel_elements,
        height,
        width,
    )
    channels_per_group = channels // group_size
    patches = patches.view(
        batch,
        group_size,
        channels_per_group,
        kernel_elements,
        height,
        width,
    )
    subpixel_count = scale_factor ** 2
    masks = F.pixel_unshuffle(masks, scale_factor)
    masks = masks.view(
        batch,
        group_size,
        kernel_elements,
        subpixel_count,
        height,
        width,
    )
    output = torch.einsum('bgckhw,bgkshw->bgcshw', patches, masks)
    output = output.reshape(
        batch,
        channels * subpixel_count,
        height,
        width,
    )
    return F.pixel_shuffle(output, scale_factor)


class GateGenerator(nn.Module):
    def __init__(self, in_channels):
        super(GateGenerator, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.weights_init_random()

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.conv(x), scale_factor=2))

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class Aligner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Aligner, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weights_init_random()

    def forward(self, x):
        return self.conv(x)

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShift(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, out_channels, embedding_dim=64, kernel_size=3):
        super(SemiShift, self).__init__()
        self.compressor_en = nn.Conv2d(in_channels_en, embedding_dim, kernel_size=1)
        self.compressor_de = nn.Conv2d(in_channels_de, embedding_dim, kernel_size=1, bias=False)
        self.content_encoder = nn.Conv2d(embedding_dim, out_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2)
        self.weights_init_random()

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        output = self.content_encoder(enc) + F.interpolate(self.content_encoder(dec), scale_factor=2)
        return output

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class SemiShiftDepthWise(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, out_channels, kernel_size=3):
        super(SemiShiftDepthWise, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.compressor_en = nn.Conv2d(in_channels_en, out_channels, kernel_size=1)
        self.compressor_de = nn.Conv2d(in_channels_de, out_channels, kernel_size=1, bias=False)
        self.content_encoder = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                                         padding=kernel_size // 2, groups=out_channels)
        self.weights_init_random()

    def forward(self, en, de):
        enc = self.compressor_en(en)
        dec = self.compressor_de(de)
        output = self.content_encoder(enc) + F.interpolate(self.content_encoder(dec), scale_factor=2)
        return output

    def weights_init_random(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


class KernelGenerator(nn.Module):
    def __init__(self, in_channels_en, in_channels_de, conv, up_kernel_size=5):
        super(KernelGenerator, self).__init__()
        self.conv = conv(in_channels_en, in_channels_de, out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        return F.softmax(self.conv(en, de), dim=1)


class SemiShiftUp(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(SemiShiftUp, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.ker_generator = SemiShift(in_channels_en, in_channels_de, out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADE(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADE, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        # self.aligner = Aligner(in_channels_en, in_channels_de)
        self.ker_generator = SemiShift(in_channels_en, in_channels_de,
                                       out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return gate * en + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADELite(nn.Module):
    def __init__(self, in_channels_en, in_channels_de=None, scale=2, up_kernel_size=5):
        super(FADELite, self).__init__()
        in_channels_de = in_channels_de if in_channels_de is not None else in_channels_en
        self.scale = scale
        self.up_kernel_size = up_kernel_size
        self.gate_generator = GateGenerator(in_channels_de)
        # self.aligner = Aligner(in_channels_en, in_channels_de)
        self.ker_generator = SemiShiftDepthWise(in_channels_en, in_channels_de,
                                                out_channels=up_kernel_size ** 2)

    def forward(self, en, de):
        gate = self.gate_generator(de)
        kernels = F.softmax(self.ker_generator(en, de), dim=1)
        return gate * en + (1 - gate) * carafe(de, kernels, self.up_kernel_size, 1, self.scale)


class FADE2xBlock(nn.Module):
    """One independent FADE stage from a low to the next higher resolution.

    ``low_resolution_feature`` is the feature being restored. The
    ``high_resolution_guidance`` tensor supplies the boundary/detail guidance
    at the target scale. Stride-rounded encoder sizes may be one pixel smaller
    than an exact 2x size; the guidance is padded only for the original FADE
    calculation and the result is cropped back to the true target size.
    """

    def __init__(self, channels, up_kernel_size=5):
        super(FADE2xBlock, self).__init__()
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.fade = FADE(
            in_channels_en=self.channels,
            in_channels_de=self.channels,
            scale=2,
            up_kernel_size=up_kernel_size,
        )

    def forward(self, low_resolution_feature, high_resolution_guidance):
        if low_resolution_feature.dim() != 4 or high_resolution_guidance.dim() != 4:
            raise ValueError("FADE2xBlock expects two 4D tensors (B,C,H,W).")
        if low_resolution_feature.shape[:2] != high_resolution_guidance.shape[:2]:
            raise ValueError(
                "FADE2xBlock inputs must share batch size and channels, got "
                f"{tuple(low_resolution_feature.shape)} and "
                f"{tuple(high_resolution_guidance.shape)}."
            )
        if low_resolution_feature.size(1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got "
                f"{low_resolution_feature.size(1)}."
            )

        target_height, target_width = high_resolution_guidance.shape[-2:]
        exact_height = low_resolution_feature.size(-2) * 2
        exact_width = low_resolution_feature.size(-1) * 2
        height_padding = exact_height - target_height
        width_padding = exact_width - target_width
        if height_padding not in (0, 1) or width_padding not in (0, 1):
            raise ValueError(
                "Each FADE2xBlock stage requires an exact 2x target or a "
                "stride-rounded target one pixel smaller; got low size "
                f"{tuple(low_resolution_feature.shape[-2:])} and guidance size "
                f"{tuple(high_resolution_guidance.shape[-2:])}."
            )

        padded_guidance = high_resolution_guidance
        if height_padding or width_padding:
            padded_guidance = F.pad(
                padded_guidance,
                pad=(0, width_padding, 0, height_padding),
                mode='replicate',
            )

        output = self.fade(padded_guidance, low_resolution_feature)
        return output[..., :target_height, :target_width]


class CascadedFADE(nn.Module):
    """Cascade independent 2x FADE stages using a decoder guidance pyramid."""

    def __init__(self, channels, stages, up_kernel_size=5):
        super(CascadedFADE, self).__init__()
        self.channels = int(channels)
        self.num_stages = int(stages)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if self.num_stages <= 0:
            raise ValueError(f"stages must be positive, got {stages}.")
        self.stages = nn.ModuleList(
            FADE2xBlock(
                channels=self.channels,
                up_kernel_size=up_kernel_size,
            )
            for _ in range(self.num_stages)
        )

    @staticmethod
    def _ceil_div(value, divisor):
        return (int(value) + int(divisor) - 1) // int(divisor)

    def forward(self, low_resolution_feature, high_resolution_guidance):
        if low_resolution_feature.dim() != 4 or high_resolution_guidance.dim() != 4:
            raise ValueError("CascadedFADE expects two 4D tensors (B,C,H,W).")
        if low_resolution_feature.shape[:2] != high_resolution_guidance.shape[:2]:
            raise ValueError(
                "CascadedFADE inputs must share batch size and channels, got "
                f"{tuple(low_resolution_feature.shape)} and "
                f"{tuple(high_resolution_guidance.shape)}."
            )

        final_height, final_width = high_resolution_guidance.shape[-2:]
        feature = low_resolution_feature
        for stage_index, stage in enumerate(self.stages):
            remaining_stages = self.num_stages - stage_index - 1
            divisor = 2 ** remaining_stages
            guidance_size = (
                self._ceil_div(final_height, divisor),
                self._ceil_div(final_width, divisor),
            )
            if guidance_size == (final_height, final_width):
                stage_guidance = high_resolution_guidance
            else:
                stage_guidance = F.interpolate(
                    high_resolution_guidance,
                    size=guidance_size,
                    mode='bilinear',
                    align_corners=False,
                )
            feature = stage(feature, stage_guidance)
        return feature


class FADEGateFusion(nn.Module):
    """FADE-inspired decoder-dependent fusion for same-scale Layer1 features."""

    def __init__(self, channels):
        super(FADEGateFusion, self).__init__()
        self.channels = int(channels)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        self.gate_generator = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.weights_init_random()

    def weights_init_random(self):
        xavier_init(self.gate_generator, distribution='uniform')
        xavier_init(self.refine[0], distribution='uniform')
        nn.init.ones_(self.refine[1].weight)
        nn.init.zeros_(self.refine[1].bias)

    def forward(self, encoder_feature, decoder_feature):
        if encoder_feature.dim() != 4 or decoder_feature.dim() != 4:
            raise ValueError("FADEGateFusion expects two 4D tensors (B,C,H,W).")
        if encoder_feature.shape != decoder_feature.shape:
            raise ValueError(
                "FADEGateFusion inputs must have the same shape, got "
                f"{tuple(encoder_feature.shape)} and "
                f"{tuple(decoder_feature.shape)}."
            )
        if encoder_feature.size(1) != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {encoder_feature.size(1)}."
            )

        gate = torch.sigmoid(self.gate_generator(decoder_feature))
        fused = gate * encoder_feature + (1 - gate) * decoder_feature
        return self.refine(fused)


if __name__ == '__main__':
    x = torch.randn(2, 3, 4, 4).to('cuda')
    y = torch.randn(2, 3, 8, 8).to('cuda')
    fade = FADE(3).to('cuda')
    fade_lite = FADELite(3).to('cuda')
    print(fade(y, x).shape)
    print(fade_lite(y, x).shape)
