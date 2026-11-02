"""受原论文 CDM 思想启发的“依赖度-任务相关性”通道注意力模块。

重要说明
--------
本文件不是原论文 Causal Decorrelation Module（CDM）的严格复现。原始 CDM
通过“样本重加权”降低特征之间的虚假相关；这里仅借用了其中“利用随机傅里叶
特征（Random Fourier Features, RFF）度量非线性依赖”的思想，并将其改造成
可以直接插入卷积网络的通道重标定模块。

整体流程
--------
输入特征 F: [B, C, H, W]
    ├─ 依赖度分支（Dependency branch）
    │    空间池化 -> 每个通道独立标准化 -> RFF 非线性映射
    │    -> 两两通道依赖矩阵 D: [B, C, C]
    │    -> 每个通道的平均依赖描述 d: [B, C]
    │
    └─ 任务相关性分支（Relevance branch）
         全局平均池化 GAP + 全局最大池化 GMP
         -> 多层感知机 MLP -> 任务相关性 r: [B, C]

将每个通道的 [r, d] 输入共享的轻量融合 MLP，得到门控 A: [B, C, 1, 1]。
最终输出：F_out = F + (A - 1) * F = A * F。

门控 A 的范围为 (0, 2)：A < 1 表示抑制通道，A > 1 表示增强通道。
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# 依赖度分支使用的池化尺寸：可以是正方形边长、(高度, 宽度)，或 None。
# None 表示不进行空间池化，直接使用输入特征图的原始空间尺寸。
PoolSize = Union[int, Tuple[int, int], None]


class RandomFourierFeatureMap(nn.Module):
    """用固定随机参数实现标量 RBF 核的随机傅里叶特征映射。

    对 RBF 核 k(x, y) = exp(-gamma * ||x-y||^2)，从对应的频谱分布
    omega ~ N(0, 2*gamma) 中采样随机频率。对于一个标量激活值 x，输出：

        phi(x) = sqrt(2/M) * cos(omega * x + phase)

    其中 M 是随机特征数量。该映射把一个标量扩展为 M 维非线性表示，使后续
    的协方差统计能够感知一部分非线性依赖，而不只局限于原始空间的线性相关。

    omega 和 phase 被注册为 buffer，而不是可训练参数：它们会跟随模型移动到
    CPU/GPU、会保存在 state_dict 中，但不会被优化器更新。
    """

    def __init__(
        self,
        num_features: int = 8,
        gamma: float = 1.0,
        seed: int = 3407,
    ) -> None:
        super().__init__()
        # 先检查超参数，避免创建无意义的随机映射。
        if num_features < 1:
            raise ValueError("num_features must be >= 1")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")

        self.num_features = int(num_features)
        self.gamma = float(gamma)

        # 使用独立的 CPU 随机数生成器。手动设置种子可以保证每次构造模块时
        # 得到相同的随机频率和相位，而不会修改程序全局的随机状态。
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        # 从标准正态分布采样频率，再按 RBF 核参数 gamma 调整频率尺度。
        omega = torch.randn(self.num_features, generator=generator)
        omega = omega * math.sqrt(2.0 * self.gamma)
        # 相位在 [0, 2*pi) 上均匀采样。
        phase = 2.0 * math.pi * torch.rand(
            self.num_features, generator=generator
        )

        # buffer 会被保存和迁移设备，但不参与梯度下降。
        self.register_buffer("omega", omega, persistent=True)
        self.register_buffer("phase", phase, persistent=True)
        # RFF 的归一化系数，避免随机特征数量 M 改变整体输出尺度。
        self.scale = math.sqrt(2.0 / self.num_features)

    def forward(self, x: Tensor) -> Tensor:
        """将任意形状 ``[...]`` 的标量张量映射为 ``[..., M]`` 的 RFF 张量。"""
        # 保证固定随机参数与输入处于同一设备、使用同一数据类型。
        omega = self.omega.to(device=x.device, dtype=x.dtype)
        phase = self.phase.to(device=x.device, dtype=x.dtype)
        # 在输入末尾增加 RFF 维度，并同时应用 M 组随机频率和相位。
        return self.scale * torch.cos(x.unsqueeze(-1) * omega + phase)


class DependencyBranch(nn.Module):
    """独立计算通道非线性依赖度的分支。

    输入为卷积特征 [B,C,H,W]，输出 ``(dependency, dependency_matrix)``：
    dependency 的形状为 [B,C]，表示每个通道与其他通道的平均依赖度；
    dependency_matrix 的形状为 [B,C,C]，保留完整的两两通道依赖关系。
    """

    def __init__(
        self,
        channels: int,
        rff_dim: int,
        rff_gamma: float,
        dependency_pool_size: PoolSize,
        dependency_chunk_size: int,
        dependency_mode: str,
        eps: float,
        rff_seed: int,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dependency_pool_size = dependency_pool_size
        self.dependency_chunk_size = int(dependency_chunk_size)
        self.dependency_mode = dependency_mode
        self.eps = float(eps)

        # 将标准化后的通道响应映射到固定的非线性 RFF 空间。
        self.rff = RandomFourierFeatureMap(
            num_features=rff_dim,
            gamma=rff_gamma,
            seed=rff_seed,
        )

    def _pool_for_dependency(self, x: Tensor) -> Tensor:
        """为依赖度统计调整空间尺寸，以控制后续计算量。"""
        if self.dependency_pool_size is None:
            return x

        if isinstance(self.dependency_pool_size, int):
            output_size = (
                self.dependency_pool_size,
                self.dependency_pool_size,
            )
        else:
            output_size = self.dependency_pool_size
        return F.adaptive_avg_pool2d(x, output_size)

    def _rff_channel_features(self, x: Tensor) -> Tensor:
        """得到标准化的通道 RFF 表示，输出形状为 [B,C,N,M]。"""
        # 1. 池化，并将每个空间位置视为当前样本内的一个观测。
        pooled = self._pool_for_dependency(x)
        b, c, h, w = pooled.shape
        n = h * w
        if n < 2:
            raise ValueError(
                "Dependency estimation needs at least two spatial observations; "
                f"got pooled size {h}x{w}."
            )

        # 2. 用 fp32 计算统计量，并对每个样本、通道沿空间维标准化。
        z = pooled.float().reshape(b, c, n)
        z = z - z.mean(dim=-1, keepdim=True)
        z = z / torch.sqrt(z.var(dim=-1, keepdim=True, unbiased=False) + self.eps)

        # 3. RFF 映射、中心化与逐坐标方差归一化。
        phi = self.rff(z)  # [B,C,N,M]
        phi = phi - phi.mean(dim=2, keepdim=True)
        phi_var = phi.square().mean(dim=2, keepdim=True)
        return phi / torch.sqrt(phi_var + self.eps)

    def _dependency_matrix(self, x: Tensor) -> Tensor:
        """计算每个样本的 RFF 通道依赖矩阵 D，形状为 [B,C,C]。"""
        phi = self._rff_channel_features(x)
        _, channels, n, _ = phi.shape
        denom = float(max(n - 1, 1))

        # 按 query 通道分块，减少 full 模式的峰值显存。
        blocks = []
        for start in range(0, channels, self.dependency_chunk_size):
            end = min(start + self.dependency_chunk_size, channels)
            query = phi[:, start:end]  # [B,K,N,M]

            if self.dependency_mode == "full":
                # 完整 M*M RFF 交叉协方差，再取平方 Frobenius 能量。
                cross_cov = torch.einsum(
                    "bknm,bcnr->bkcmr", query, phi
                ) / denom
                dep = cross_cov.square().mean(dim=(-1, -2))  # [B,K,C]
            else:
                # 只使用相同 RFF 坐标的交叉协方差，计算代价更低。
                cross_cov = torch.einsum(
                    "bknm,bcnm->bkcm", query, phi
                ) / denom
                dep = cross_cov.square().mean(dim=-1)  # [B,K,C]
            blocks.append(dep)

        dependency = torch.cat(blocks, dim=1)  # [B,C,C]
        # 数值保护，并移除通道与自身的依赖 D_ii。
        dependency = dependency.clamp(min=0.0, max=1.0)
        eye = torch.eye(
            channels,
            device=dependency.device,
            dtype=dependency.dtype,
        )
        return dependency * (1.0 - eye.unsqueeze(0))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """返回每通道依赖描述 [B,C] 和完整依赖矩阵 [B,C,C]。"""
        matrix = self._dependency_matrix(x)
        if self.channels == 1:
            descriptor = torch.zeros(
                matrix.shape[0], 1, device=matrix.device, dtype=matrix.dtype
            )
        else:
            descriptor = matrix.sum(dim=-1) / float(self.channels - 1)
        return descriptor, matrix


class RelevanceBranch(nn.Module):
    """独立学习通道任务相关性的分支。

    该分支将 GAP 和 GMP 结果拼接后输入 MLP，输出每个通道位于 (0,1) 的相关性
    分数。该分数由下游任务损失通过反向传播间接学习。
    """

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.reduction = int(reduction)
        hidden = max(self.channels // self.reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, self.channels, bias=True),
        )

    def reset_parameters(self) -> None:
        """按原实现初始化 MLP 的线性层。"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """根据全局平均/最大响应输出任务相关性，形状为 [B,C]。"""
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        maxv = F.adaptive_max_pool2d(x, 1).flatten(1)
        global_stat = torch.cat([avg, maxv], dim=1)  # [B,2*C]
        return torch.sigmoid(self.mlp(global_stat))


class KDRM(nn.Module):
    """双分支通道注意力：DependencyBranch + RelevanceBranch + 融合门控。

    输入输出均为 [B,C,H,W]。DependencyBranch 输出通道依赖度，RelevanceBranch
    输出任务相关性；主模块将二者融合为门控 A∈(0,2)，并输出 A*X。
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        rff_dim: int = 8,
        rff_gamma: float = 1.0,
        dependency_pool_size: PoolSize = 8,
        dependency_chunk_size: int = 32,
        dependency_mode: str = "full",
        fusion_hidden: int = 8,
        eps: float = 1e-5,
        rff_seed: int = 3407,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")
        if dependency_chunk_size < 1:
            raise ValueError("dependency_chunk_size must be >= 1")
        if dependency_mode not in {"full", "diagonal"}:
            raise ValueError("dependency_mode must be 'full' or 'diagonal'")
        if fusion_hidden < 1:
            raise ValueError("fusion_hidden must be >= 1")

        self.channels = int(channels)

        # 两个子模块各自只负责产生一种描述量。
        self.dependency_branch = DependencyBranch(
            channels=self.channels,
            rff_dim=rff_dim,
            rff_gamma=rff_gamma,
            dependency_pool_size=dependency_pool_size,
            dependency_chunk_size=dependency_chunk_size,
            dependency_mode=dependency_mode,
            eps=eps,
            rff_seed=rff_seed,
        )
        self.relevance_branch = RelevanceBranch(
            channels=self.channels,
            reduction=reduction,
        )

        # 主模块只保留两个描述量的共享融合 MLP。
        self.fusion = nn.Sequential(
            nn.Linear(2, fusion_hidden, bias=True),
            nn.GELU(),
            nn.Linear(fusion_hidden, 1, bias=True),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """保持原有初始化：相关性 MLP 常规初始化，融合门控接近恒等映射。"""
        self.relevance_branch.reset_parameters()
        nn.init.kaiming_uniform_(self.fusion[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.fusion[0].bias)
        nn.init.normal_(self.fusion[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Tensor],
        prefix: str,
        local_metadata: Dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """加载旧版权重时，将扁平键名迁移到两个新分支的键名。"""
        legacy_prefixes = {
            "rff.": "dependency_branch.rff.",
            "relevance_mlp.": "relevance_branch.mlp.",
        }
        for old_prefix, new_prefix in legacy_prefixes.items():
            for key in list(state_dict.keys()):
                if key.startswith(prefix + old_prefix):
                    new_key = prefix + new_prefix + key[len(prefix + old_prefix):]
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: Tensor,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """依次调用两条分支，融合后输出重标定特征。"""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input [B, C, H, W], got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Configured for C={self.channels}, but input has C={x.shape[1]}"
            )

        # 两条分支相互独立地产生 dependency 和 relevance。
        dependency, dependency_matrix = self.dependency_branch(x)
        relevance = self.relevance_branch(x)

        # 依赖统计固定在 fp32 完成；进入可训练融合层前转回模型数据类型。
        dependency = dependency.to(dtype=relevance.dtype)
        joint = torch.stack([relevance, dependency], dim=-1)  # [B,C,2]
        gate_logits = self.fusion(joint).squeeze(-1)  # [B,C]

        # A∈(0,2)，再沿空间维广播到输入特征图。
        attention = 2.0 * torch.sigmoid(gate_logits)
        attention_4d = attention.unsqueeze(-1).unsqueeze(-1)
        out = x + (attention_4d - 1.0) * x

        if not return_aux:
            return out

        aux: Dict[str, Tensor] = {
            "attention": attention_4d,
            "relevance": relevance,
            "dependency": dependency,
            "dependency_matrix": dependency_matrix,
        }
        return out, aux

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"rff_dim={self.dependency_branch.rff.num_features}, "
            f"dependency_pool_size={self.dependency_branch.dependency_pool_size}, "
            f"dependency_mode='{self.dependency_branch.dependency_mode}'"
        )


if __name__ == "__main__":
    # 最小可运行示例：验证输入输出尺寸以及辅助统计量的尺寸。
    torch.manual_seed(0)
    module = KDRM(
        channels=64,
        reduction=8,
        rff_dim=8,
        dependency_pool_size=8,
        dependency_chunk_size=16,
        dependency_mode="full",
    )

    # 模拟一个 batch=2、通道数=64、空间尺寸=32*32 的中间特征图。
    feature = torch.randn(2, 64, 32, 32)
    # return_aux=True：除增强特征外，同时取得注意力和依赖矩阵等中间结果。
    enhanced, info = module(feature, return_aux=True)

    print("input: ", feature.shape)
    print("output:", enhanced.shape)
    print("attention:", info["attention"].shape)
    print("dependency matrix:", info["dependency_matrix"].shape)
