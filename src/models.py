"""共享主干定义、使用不同默认宽度的基线与三特征 U-Net。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


UNET_BASE_CHANNELS = 32
TRI_UNET_BASE_CHANNELS = 24
AUXILIARY_MODEL_NAMES = (
    "tri_unet",
    "tri_retire",
    "gcsas_unet",
    "tri_gate",
    "quad_gate_lite",
    "lite_sdf_distill",
    "performance_context",
    "compressed_context",
    "compressed_context_fullpath",
    "ambiguity_context",
    "ultralite_sdf_distill",
    "nanolite_sdf_distill",
    "microlite_sdf_distill",
    "picolite_sdf_distill",
    "microlite_fullpath_distill",
    "picolite_fullpath_distill",
    "picolite_selective_fullpath_distill",
    "picolite_shallow_semantic_distill",
    "picolite_learned_head_fullpath_distill",
    "picolite_encoder_semantic_distill",
)
BASELINE_MODEL_NAMES = ("unet", "tiny_unet_5k")
RETIRING_AUXILIARY_MODEL_NAMES = ("tri_retire", "tri_gate", "quad_gate_lite")


class DoubleConv(nn.Sequential):
    """两次 ``3x3 Conv-BN-ReLU``。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNetBackbone(nn.Module):
    """两种模型复用的编码器、瓶颈、解码器和分割头定义。"""

    downsample_factor = 16

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = UNET_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) <= 0:
            raise ValueError("channel counts must be positive")

        c1, c2, c3, c4, c5 = (base_channels * 2**i for i in range(5))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.encoder_channels = (c1, c2, c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.encoder1 = DoubleConv(in_channels, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.encoder4 = DoubleConv(c3, c4)
        self.bottleneck = DoubleConv(c4, c5)

        self.up4 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.decoder4 = DoubleConv(2 * c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.decoder3 = DoubleConv(2 * c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.decoder2 = DoubleConv(2 * c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.decoder1 = DoubleConv(2 * c1, c1)
        self.segmentation_head = nn.Conv2d(c1, out_channels, 1)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected input shaped [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        height, width = x.shape[-2:]
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError(
                "input height and width must both be divisible by 16; "
                f"got {(height, width)}"
            )

    def encode(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """返回瓶颈和编码特征，供可学习跳连调制使用。"""
        self._validate_input(x)
        features: list[Tensor] = []
        for encoder in (self.encoder1, self.encoder2, self.encoder3, self.encoder4):
            x = encoder(x if not features else self.pool(x))
            features.append(x)
        return self.bottleneck(self.pool(x)), tuple(features)

    def decode_features(
        self, x: Tensor, features: tuple[Tensor, ...]
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """解码并返回 ``(D4, D3, D2, D1)`` 特征供深监督使用。"""

        if len(features) != 4:
            raise ValueError(f"expected four skip features, got {len(features)}")
        decoded: list[Tensor] = []
        for up, decoder, skip in zip(
            (self.up4, self.up3, self.up2, self.up1),
            (self.decoder4, self.decoder3, self.decoder2, self.decoder1),
            reversed(features),
        ):
            x = decoder(torch.cat((up(x), skip), dim=1))
            decoded.append(x)
        return self.segmentation_head(x), tuple(decoded)

    def decode(self, x: Tensor, features: tuple[Tensor, ...]) -> Tensor:
        """从瓶颈和四层跳连特征解码分割 logits。"""

        return self.decode_features(x, features)[0]

    def forward_features(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """返回分割 logits 与编码特征 ``(E1, E2, E3, E4)``。"""

        bottleneck, features = self.encode(x)
        return self.decode(bottleneck, features), features

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]


def split_feature_channels(feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """按 ``1:1:2`` 固定切分边界、区域和自由通道。"""

    if feature.ndim != 4:
        raise ValueError(
            f"expected a feature map shaped [B, C, H, W], got {tuple(feature.shape)}"
        )
    quarter = feature.shape[1] // 4
    if not quarter:
        raise ValueError("feature map must contain at least four channels")
    return (
        feature[:, :quarter],
        feature[:, quarter : 2 * quarter],
        feature[:, 2 * quarter :],
    )


def split_quad_feature_channels(
    feature: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """按 ``1:1:1:2`` 切分边界、纹理、形状和自由通道。"""

    if feature.ndim != 4:
        raise ValueError(
            f"expected a feature map shaped [B, C, H, W], got {tuple(feature.shape)}"
        )
    fifth = feature.shape[1] // 5
    if fifth == 0 or feature.shape[1] % 5:
        raise ValueError("quad feature channels must be positive and divisible by five")
    return (
        feature[:, :fifth],
        feature[:, fifth : 2 * fifth],
        feature[:, 2 * fifth : 3 * fifth],
        feature[:, 3 * fifth :],
    )


class MultiScaleConceptHead(nn.Module):
    """Training-only top-down readout for one dedicated concept group."""

    def __init__(
        self,
        in_channels: tuple[int, ...],
        hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        if not in_channels or min(in_channels) <= 0 or hidden_channels <= 0:
            raise ValueError("concept-head channel counts must be positive")
        groups = 4 if hidden_channels % 4 == 0 else 1
        self.in_channels = tuple(in_channels)
        self.hidden_channels = hidden_channels
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, hidden_channels, 1, bias=False),
                nn.GroupNorm(groups, hidden_channels),
                nn.SiLU(inplace=True),
            )
            for channels in self.in_channels
        )
        self.readouts = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    3,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.GroupNorm(groups, hidden_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, 1),
            )
            for _ in self.in_channels
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature scales, got {len(features)}"
            )
        for index, (feature, channels) in enumerate(zip(features, self.in_channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"feature {index} must be [B, {channels}, H, W], "
                    f"got {tuple(feature.shape)}"
                )

        fused = [
            projection(feature)
            for projection, feature in zip(self.projections, features)
        ]
        for index in range(len(fused) - 2, -1, -1):
            top_down = F.interpolate(
                fused[index + 1],
                size=fused[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused[index] = fused[index] + top_down
        return [readout(feature) for readout, feature in zip(self.readouts, fused)]


class DirectConceptHead(nn.Module):
    """轻量逐尺度 ``1x1`` 概念读出，避免高分辨率隐藏金字塔。"""

    def __init__(self, in_channels: tuple[int, ...]) -> None:
        super().__init__()
        if not in_channels or min(in_channels) <= 0:
            raise ValueError("direct concept-head channel counts must be positive")
        self.in_channels = tuple(in_channels)
        self.readouts = nn.ModuleList(
            nn.Conv2d(channels, 1, 1) for channels in self.in_channels
        )

    def read(self, feature: Tensor, index: int) -> Tensor:
        if not 0 <= index < len(self.readouts):
            raise IndexError(index)
        if feature.ndim != 4 or feature.shape[1] != self.in_channels[index]:
            raise ValueError(
                f"feature {index} must be [B, {self.in_channels[index]}, H, W], "
                f"got {tuple(feature.shape)}"
            )
        return self.readouts[index](feature)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != len(self.readouts):
            raise ValueError(
                f"expected {len(self.readouts)} feature scales, got {len(features)}"
            )
        return [self.read(feature, index) for index, feature in enumerate(features)]


class UNet(nn.Module):
    """基线 U-Net。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = UNET_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_channels, out_channels, base_channels)

    def forward(self, x: Tensor, return_aux: bool = False) -> dict[str, Tensor]:
        del return_aux
        return {"seg_logits": self.backbone(x)}


class TriFeatureUNet(nn.Module):
    """在较窄 U-Net 主干的 E1--E3 通道切片上增加轻量辅助头。"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = TRI_UNET_BASE_CHANNELS,
    ) -> None:
        super().__init__()
        if base_channels % 4:
            raise ValueError("base_channels must be divisible by four")
        self.backbone = UNetBackbone(in_channels, out_channels, base_channels)
        channels = self.backbone.encoder_channels[:3]
        concept_channels = tuple(c // 4 for c in channels)
        self.boundary_auxiliary = MultiScaleConceptHead(concept_channels)
        self.texture_auxiliary = MultiScaleConceptHead(concept_channels)

    # 原三特征候选的辅助头只服务训练，不计入正式推理图。
    inference_uses_auxiliary = False

    @property
    def channel_widths(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (c // 4, c // 4, c - 2 * (c // 4))
            for c in self.backbone.encoder_channels
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        seg_logits, features = self.backbone.forward_features(x)
        if not return_aux:  # 辅助头在推理路径中完全跳过。
            return {"seg_logits": seg_logits}

        groups = [split_feature_channels(feature) for feature in features[:3]]
        boundary_features = [group[0] for group in groups]
        texture_features = [group[1] for group in groups]
        return {
            "seg_logits": seg_logits,
            "boundary_logits": self.boundary_auxiliary(boundary_features),
            "texture_logits": self.texture_auxiliary(texture_features),
        }


class ConceptResidualGate(nn.Module):
    """用边界/纹理概念证据生成有界的逐通道空间残差门控。

    零初始化使新候选在训练起点严格保持恒等跳连；之后分割梯度可直接
    学习哪些概念、位置和通道应被增强或抑制。
    """

    def __init__(
        self, channels: int, gain: float = 0.5, concept_channels: int = 2
    ) -> None:
        super().__init__()
        if channels <= 0 or concept_channels <= 0 or not 0 < gain <= 1:
            raise ValueError("gate channels must be positive and gain must be in (0, 1]")
        self.gain = float(gain)
        self.concept_channels = int(concept_channels)
        self.projection = nn.Conv2d(concept_channels, channels, 1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature: Tensor, *concept_logits: Tensor) -> Tensor:
        if len(concept_logits) != self.concept_channels:
            raise ValueError(f"expected {self.concept_channels} concept evidence maps")
        if any(logit.shape != concept_logits[0].shape for logit in concept_logits[1:]):
            raise ValueError("concept evidence maps must have matching shapes")
        if concept_logits[0].shape[:2] != (feature.shape[0], 1):
            raise ValueError("concept evidence must be single-channel and batch matched")
        if concept_logits[0].shape[-2:] != feature.shape[-2:]:
            raise ValueError("concept evidence and skip feature spatial sizes must match")
        evidence = torch.cat(
            tuple(torch.sigmoid(logit) for logit in concept_logits), dim=1
        )
        modulation = torch.tanh(self.projection(evidence))
        return feature * (1.0 + self.gain * modulation)


class ConceptGatedUNet(TriFeatureUNet):
    """将训练期概念读出变为正式分割路径中的轻量跳连门控。"""

    inference_uses_auxiliary = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = TRI_UNET_BASE_CHANNELS,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        self.concept_gates = nn.ModuleList(
            ConceptResidualGate(channels)
            for channels in self.backbone.encoder_channels[:3]
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        bottleneck, features = self.backbone.encode(x)
        groups = [split_feature_channels(feature) for feature in features[:3]]
        boundary_logits = self.boundary_auxiliary([group[0] for group in groups])
        texture_logits = self.texture_auxiliary([group[1] for group in groups])
        gated_features = list(features)
        for index, gate in enumerate(self.concept_gates):
            gated_features[index] = gate(
                features[index], boundary_logits[index], texture_logits[index]
            )
        seg_logits = self.backbone.decode(bottleneck, tuple(gated_features))
        outputs: dict[str, Tensor | list[Tensor]] = {"seg_logits": seg_logits}
        if return_aux:
            outputs.update(
                boundary_logits=boundary_logits,
                texture_logits=texture_logits,
            )
        return outputs


class QuadGateLiteUNet(nn.Module):
    """四特征轻量门控 U-Net：边界/纹理/形状/自由 = ``1:1:1:2``。"""

    inference_uses_auxiliary = True
    supports_auxiliary = True

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 20,
    ) -> None:
        super().__init__()
        if base_channels % 5:
            raise ValueError("quad-gate base_channels must be divisible by five")
        self.backbone = UNetBackbone(in_channels, out_channels, base_channels)
        channels = self.backbone.encoder_channels[:3]
        concept_channels = tuple(channels_at_scale // 5 for channels_at_scale in channels)
        self.boundary_auxiliary = DirectConceptHead(concept_channels)
        self.texture_auxiliary = DirectConceptHead(concept_channels)
        self.shape_auxiliary = DirectConceptHead(concept_channels)
        # E2 was neutral/slightly harmful in the seed-42 gate ablation. It keeps
        # training-only concept supervision but is omitted from formal inference.
        self.concept_gates = nn.ModuleDict(
            {
                "e1": ConceptResidualGate(channels[0], concept_channels=3),
                "e3": ConceptResidualGate(channels[2], concept_channels=3),
            }
        )

    @property
    def channel_widths(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (c // 5, c // 5, c // 5, 2 * (c // 5))
            for c in self.backbone.encoder_channels
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        bottleneck, features = self.backbone.encode(x)
        groups = [split_quad_feature_channels(feature) for feature in features[:3]]
        boundary_features = [group[0] for group in groups]
        texture_features = [group[1] for group in groups]
        shape_features = [group[2] for group in groups]

        needed = (0, 1, 2) if return_aux else (0, 2)
        boundary = {
            i: self.boundary_auxiliary.read(boundary_features[i], i) for i in needed
        }
        texture = {
            i: self.texture_auxiliary.read(texture_features[i], i) for i in needed
        }
        shape = {i: self.shape_auxiliary.read(shape_features[i], i) for i in needed}
        gated_features = list(features)
        for index, name in ((0, "e1"), (2, "e3")):
            gated_features[index] = self.concept_gates[name](
                features[index], boundary[index], texture[index], shape[index]
            )
        outputs: dict[str, Tensor | list[Tensor]] = {
            "seg_logits": self.backbone.decode(bottleneck, tuple(gated_features))
        }
        if return_aux:
            outputs.update(
                boundary_logits=[boundary[i] for i in range(3)],
                texture_logits=[texture[i] for i in range(3)],
                shape_logits=[shape[i] for i in range(3)],
            )
        return outputs


class LiteSDFDistillUNet(nn.Module):
    """Base-20 U-Net with richer heads that exist during training only.

    The deployable graph is exactly :class:`UNetBackbone`.  Full encoder
    features feed scale-appropriate boundary, texture, and signed-distance
    heads, avoiding the narrow fixed channel slices used by the earlier lite
    candidate.  ``return_aux=False`` skips every auxiliary module.
    """

    inference_uses_auxiliary = False
    supports_auxiliary = True
    training_only_parameter_prefixes = (
        "boundary_auxiliary.",
        "texture_auxiliary.",
        "sdf_auxiliary.",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 20,
    ) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_channels, out_channels, base_channels)
        e1, e2, e3 = self.backbone.encoder_channels[:3]
        # Boundary keeps all three resolutions. Texture is restricted to E1/E2,
        # while global signed-distance/shape supervision is restricted to E2/E3.
        self.boundary_auxiliary = MultiScaleConceptHead(
            (e1, e2, e3), hidden_channels=12
        )
        self.texture_auxiliary = MultiScaleConceptHead(
            (e1, e2), hidden_channels=12
        )
        self.sdf_auxiliary = MultiScaleConceptHead(
            (e2, e3), hidden_channels=12
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        seg_logits, features = self.backbone.forward_features(x)
        if not return_aux:
            return {"seg_logits": seg_logits}
        return {
            "seg_logits": seg_logits,
            "boundary_logits": self.boundary_auxiliary(list(features[:3])),
            "texture_logits": self.texture_auxiliary(list(features[:2])),
            "sdf_logits": self.sdf_auxiliary(list(features[1:3])),
        }


class DepthwiseResidualBlock(nn.Module):
    """Two depthwise-separable convolutions with a cheap residual projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if min(in_channels, out_channels) <= 0:
            raise ValueError("depthwise block channels must be positive")

        def separable(source: int, target: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(source, source, 3, padding=1, groups=source, bias=False),
                nn.BatchNorm2d(source),
                nn.SiLU(inplace=True),
                nn.Conv2d(source, target, 1, bias=False),
                nn.BatchNorm2d(target),
                nn.SiLU(inplace=True),
            )

        self.body = nn.Sequential(
            separable(in_channels, out_channels),
            separable(out_channels, out_channels),
        )
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.body(x) + self.projection(x))


class LiteChannelGate(nn.Module):
    """ECA-style channel recalibration with only three learned coefficients."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        weights = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights).transpose(1, 2).unsqueeze(-1))
        return x * weights


class LitePyramidContext(nn.Module):
    """Large receptive field from summed dilated depthwise branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            )
            for dilation in (1, 2, 4)
        )
        self.norm = nn.BatchNorm2d(channels)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        mixed = sum((branch(x) for branch in self.branches), start=torch.zeros_like(x))
        return x + self.activation(self.project(self.norm(mixed / len(self.branches))))


class TinyOutputRefinement(nn.Module):
    """Zero-initialized high-resolution residual correction with very few weights."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.project = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, feature: Tensor, coarse: Tensor) -> Tensor:
        correction = self.project(F.silu(self.norm(self.depthwise(feature))))
        return coarse + correction


class UltraLiteCore(nn.Module):
    """Standalone sub-0.5 MiB FP32 segmentation graph."""

    downsample_factor = 16

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 18,
        channel_schedule: tuple[int, int, int, int, int] | None = None,
        refine_output: bool = False,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, base_channels) <= 0:
            raise ValueError("ultralite channel counts must be positive")
        channels = channel_schedule or (
            base_channels,
            2 * base_channels,
            3 * base_channels,
            4 * base_channels,
            6 * base_channels,
        )
        if len(channels) != 5 or min(channels) <= 0 or channels[0] != base_channels:
            raise ValueError(
                "ultralite channel schedule must contain five positive widths "
                "starting with base_channels"
            )
        channels = tuple(int(value) for value in channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_schedule = channels
        self.refine_output = bool(refine_output)
        self.encoder_channels = channels[:4]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
        )
        self.encoders = nn.ModuleList(
            [
                DepthwiseResidualBlock(channels[0], channels[0]),
                DepthwiseResidualBlock(channels[0], channels[1]),
                DepthwiseResidualBlock(channels[1], channels[2]),
                DepthwiseResidualBlock(channels[2], channels[3]),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DepthwiseResidualBlock(channels[3], channels[4])
        self.context = LitePyramidContext(channels[4])
        self.skip_gates = nn.ModuleList(LiteChannelGate() for _ in range(4))
        decoder_inputs = (
            channels[4] + channels[3],
            channels[3] + channels[2],
            channels[2] + channels[1],
            channels[1] + channels[0],
        )
        decoder_outputs = tuple(reversed(channels[:4]))
        self.decoder_fusions = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(source, target, 1, bias=False),
                nn.BatchNorm2d(target),
                nn.SiLU(inplace=True),
            )
            for source, target in zip(decoder_inputs, decoder_outputs)
        )
        self.decoders = nn.ModuleList(
            DepthwiseResidualBlock(channels, channels) for channels in decoder_outputs
        )
        self.segmentation_head = nn.Conv2d(channels[0], out_channels, 1)
        self.output_refinement = (
            TinyOutputRefinement(channels[0]) if self.refine_output else None
        )

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        if any(size % self.downsample_factor for size in x.shape[-2:]):
            raise ValueError("ultralite input dimensions must be divisible by 16")

    def encode(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        self._validate_input(x)
        x = self.stem(x)
        features: list[Tensor] = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x if index == 0 else self.pool(x))
            features.append(x)
        bottleneck = self.context(self.bottleneck(self.pool(x)))
        return bottleneck, tuple(features)

    def forward_all_features(
        self, x: Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        tuple[Tensor, ...],
        Tensor,
        tuple[Tensor, ...],
    ]:
        x, features = self.encode(x)
        bottleneck = x
        decoder_features: list[Tensor] = []
        for fusion, decoder, gate, skip in zip(
            self.decoder_fusions,
            self.decoders,
            reversed(self.skip_gates),
            reversed(features),
        ):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(fusion(torch.cat((x, gate(skip)), dim=1)))
            decoder_features.append(x)
        coarse = self.segmentation_head(x)
        segmentation = (
            coarse
            if self.output_refinement is None
            else self.output_refinement(x, coarse)
        )
        return segmentation, coarse, features, bottleneck, tuple(decoder_features)

    def forward_features(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        segmentation, _, features, _, _ = self.forward_all_features(x)
        return segmentation, features

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]


class TinyUNet5K(nn.Module):
    """Parameter-matched segmentation-only baseline for the 0.005M model.

    It uses exactly the deployable 4,844-parameter encoder-decoder core of
    PicoLite, but has no semantic-channel supervisor, projection head, deep
    supervision, teacher distillation, or auxiliary loss.  This makes it a
    clean architecture/parameter-matched Tiny U-Net baseline.
    """

    channel_schedule = (4, 7, 9, 11, 12)

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__()
        if base_channels != self.channel_schedule[0]:
            raise ValueError("tiny_unet_5k requires base_channels=4")
        self.backbone = UltraLiteCore(
            in_channels,
            out_channels,
            base_channels,
            channel_schedule=self.channel_schedule,
            refine_output=True,
        )

    def forward(self, x: Tensor, return_aux: bool = False) -> dict[str, Tensor]:
        del return_aux
        return {"seg_logits": self.backbone(x)}


class UltraLiteSDFDistillUNet(nn.Module):
    """Ultralight deployable core with removable semantic training heads."""

    inference_uses_auxiliary = False
    supports_auxiliary = True
    training_only_parameter_prefixes = (
        "boundary_auxiliary.",
        "texture_auxiliary.",
        "sdf_auxiliary.",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 18,
    ) -> None:
        super().__init__()
        self.core = UltraLiteCore(in_channels, out_channels, base_channels)
        e1, e2, e3 = self.core.encoder_channels[:3]
        self.boundary_auxiliary = MultiScaleConceptHead((e1, e2, e3), hidden_channels=8)
        self.texture_auxiliary = MultiScaleConceptHead((e1, e2), hidden_channels=8)
        self.sdf_auxiliary = MultiScaleConceptHead((e2, e3), hidden_channels=8)

    @property
    def backbone(self) -> UltraLiteCore:
        """Expose common width metadata without registering the core twice."""

        return self.core

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        seg_logits, features = self.core.forward_features(x)
        if not return_aux:
            return {"seg_logits": seg_logits}
        return {
            "seg_logits": seg_logits,
            "boundary_logits": self.boundary_auxiliary(list(features[:3])),
            "texture_logits": self.texture_auxiliary(list(features[:2])),
            "sdf_logits": self.sdf_auxiliary(list(features[1:3])),
        }


class NanoLiteSDFDistillUNet(UltraLiteSDFDistillUNet):
    """Sub-0.04M-parameter deployment variant of the ultralight student."""

    max_inference_parameters = 40_000

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 9,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        inference_parameters = sum(parameter.numel() for parameter in self.core.parameters())
        if inference_parameters >= self.max_inference_parameters:
            raise ValueError(
                "nanolite inference graph must remain below 0.04M parameters; "
                f"got {inference_parameters}"
            )


class MicroLiteSDFDistillUNet(UltraLiteSDFDistillUNet):
    """Sub-0.01M student for the extreme parameter-efficiency frontier."""

    max_inference_parameters = 10_000

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        inference_parameters = sum(parameter.numel() for parameter in self.core.parameters())
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        if inference_parameters >= self.max_inference_parameters:
            raise ValueError(
                "microlite inference graph must remain below 0.01M parameters; "
                f"got {inference_parameters}"
            )
        if total_parameters >= self.max_inference_parameters:
            raise ValueError(
                "microlite training graph must remain below 0.01M parameters; "
                f"got {total_parameters}"
            )


class PicoLiteSDFDistillUNet(UltraLiteSDFDistillUNet):
    """Sub-0.005M student for the practical lower parameter frontier."""

    max_parameters = 5_000

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 2,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        inference_parameters = sum(parameter.numel() for parameter in self.core.parameters())
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        if inference_parameters >= self.max_parameters:
            raise ValueError(
                "picolite inference graph must remain below 0.005M parameters; "
                f"got {inference_parameters}"
            )
        if total_parameters >= self.max_parameters:
            raise ValueError(
                "picolite training graph must remain below 0.005M parameters; "
                f"got {total_parameters}"
            )


class FullPathChannelSupervisor(nn.Module):
    """Assign boundary, texture, and SDF semantics to fixed channels at every stage."""

    concept_count = 3

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        if not channels or min(channels) < self.concept_count:
            raise ValueError("full-path supervision needs at least three channels per stage")
        self.channels = channels
        self.log_scales = nn.Parameter(torch.zeros(len(channels), self.concept_count))
        self.biases = nn.Parameter(torch.zeros(len(channels), self.concept_count))

    def forward(self, features: tuple[Tensor, ...]) -> tuple[list[Tensor], ...]:
        if len(features) != len(self.channels):
            raise ValueError("full-path feature count does not match the supervisor")
        outputs: tuple[list[Tensor], ...] = tuple(
            [] for _ in range(self.concept_count)
        )
        for stage, (feature, channels) in enumerate(zip(features, self.channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError("invalid full-path feature shape")
            for concept in range(self.concept_count):
                value = F.group_norm(feature[:, concept : concept + 1], 1)
                logit = (
                    value * self.log_scales[stage, concept].exp()
                    + self.biases[stage, concept]
                )
                outputs[concept].append(logit)
        return outputs


class LearnedHeadFullPathSupervisor(nn.Module):
    """Predict each concept from every stage with conventional learned 1x1 heads.

    Unlike :class:`FullPathChannelSupervisor`, these training-only readers may
    mix every channel in a stage.  This provides a conservative control for
    testing whether dedicated semantic channels help beyond ordinary auxiliary
    multi-task heads.  The heads are removed from the deployed inference graph.
    """

    concept_count = 3

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        if not channels or min(channels) <= 0:
            raise ValueError("learned full-path heads require positive stage widths")
        self.channels = channels
        self.heads = nn.ModuleList(
            nn.ModuleList(nn.Conv2d(width, 1, 1) for width in channels)
            for _ in range(self.concept_count)
        )

    def forward(self, features: tuple[Tensor, ...]) -> tuple[list[Tensor], ...]:
        if len(features) != len(self.channels):
            raise ValueError("learned full-path feature count does not match")
        for feature, channels in zip(features, self.channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError("invalid learned full-path feature shape")
        return tuple(
            [head(feature) for head, feature in zip(concept_heads, features)]
            for concept_heads in self.heads
        )


class SelectiveFullPathChannelSupervisor(nn.Module):
    """Supervise two or fewer semantic channels at each U-path stage.

    PicoLite has only four channels at its highest-resolution stage.  Reserving
    three of them for concepts leaves too little unconstrained capacity for the
    segmentation objective.  This training-only reader keeps each concept on a
    continuous, task-appropriate part of the U path while never reserving more
    than two channels at one stage.
    """

    concept_count = 3

    def __init__(
        self,
        channels: tuple[int, ...],
        stage_assignments: tuple[tuple[tuple[int, int], ...], ...],
    ) -> None:
        super().__init__()
        if len(stage_assignments) != self.concept_count:
            raise ValueError("selective supervision requires three concept routes")
        if not channels:
            raise ValueError("selective supervision requires U-path stages")
        for route in stage_assignments:
            if not route:
                raise ValueError("every concept needs at least one supervised stage")
            for stage, channel in route:
                if stage < 0 or stage >= len(channels):
                    raise ValueError("semantic route stage is out of range")
                if channel < 0 or channel >= channels[stage]:
                    raise ValueError("semantic route channel is out of range")
        self.channels = channels
        self.stage_assignments = stage_assignments
        route_lengths = tuple(len(route) for route in stage_assignments)
        self.log_scales = nn.ParameterList(
            nn.Parameter(torch.zeros(length)) for length in route_lengths
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(length)) for length in route_lengths
        )

    def forward(self, features: tuple[Tensor, ...]) -> tuple[list[Tensor], ...]:
        if len(features) != len(self.channels):
            raise ValueError("selective full-path feature count does not match")
        for feature, channels in zip(features, self.channels):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError("invalid selective full-path feature shape")
        outputs: list[list[Tensor]] = []
        for concept, route in enumerate(self.stage_assignments):
            concept_outputs: list[Tensor] = []
            for route_index, (stage, channel) in enumerate(route):
                value = F.group_norm(
                    features[stage][:, channel : channel + 1], 1
                )
                concept_outputs.append(
                    value * self.log_scales[concept][route_index].exp()
                    + self.biases[concept][route_index]
                )
            outputs.append(concept_outputs)
        return tuple(outputs)


class MicroLiteFullPathDistillUNet(nn.Module):
    """Strict sub-0.01M model with semantic channels supervised across the full U path."""

    inference_uses_auxiliary = False
    supports_auxiliary = True
    training_only_parameter_prefixes = (
        "channel_supervisor.",
        "deep_supervision.",
    )
    channel_schedule = (6, 10, 14, 18, 20)
    max_parameters = 10_000

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 6,
    ) -> None:
        super().__init__()
        if base_channels != self.channel_schedule[0]:
            raise ValueError(
                "full-path model requires base_channels="
                f"{self.channel_schedule[0]}"
            )
        self.core = UltraLiteCore(
            in_channels,
            out_channels,
            base_channels,
            channel_schedule=self.channel_schedule,
            refine_output=True,
        )
        decoder_channels = tuple(reversed(self.channel_schedule[:4]))
        path_channels = (
            *self.channel_schedule[:4],
            self.channel_schedule[4],
            *decoder_channels,
        )
        self.channel_supervisor = FullPathChannelSupervisor(path_channels)
        self.deep_supervision = nn.ModuleList(
            nn.Conv2d(channels, 1, 1) for channels in decoder_channels[:3]
        )
        inference_parameters = sum(parameter.numel() for parameter in self.core.parameters())
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        if inference_parameters >= self.max_parameters:
            raise ValueError(
                "full-path inference graph exceeds its parameter budget; "
                f"got {inference_parameters}"
            )
        if total_parameters >= self.max_parameters:
            raise ValueError(
                "full-path training graph exceeds its parameter budget; "
                f"got {total_parameters}"
            )

    @property
    def backbone(self) -> UltraLiteCore:
        return self.core

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        segmentation, coarse, encoders, bottleneck, decoders = (
            self.core.forward_all_features(x)
        )
        outputs: dict[str, Tensor | list[Tensor]] = {"seg_logits": segmentation}
        if not return_aux:
            return outputs
        boundary, texture, sdf = self.channel_supervisor(
            (*encoders, bottleneck, *decoders)
        )
        outputs.update(
            coarse_seg_logits=coarse,
            deep_seg_logits=[
                head(feature)
                for head, feature in zip(self.deep_supervision, decoders[:3])
            ],
            boundary_logits=boundary,
            texture_logits=texture,
            sdf_logits=sdf,
        )
        return outputs


class PicoLiteFullPathDistillUNet(MicroLiteFullPathDistillUNet):
    """Strict sub-0.005M full-path semantic-channel student."""

    channel_schedule = (4, 7, 9, 11, 12)
    max_parameters = 5_000

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)


class PicoLiteSelectiveFullPathDistillUNet(PicoLiteFullPathDistillUNet):
    """Sub-0.005M student with capacity-aware semantic routes.

    Stage order is E1--E4, bottleneck, D4--D1.  Boundary and texture use the
    high-resolution ends of the U, while SDF occupies the deep shape path.
    E1 therefore retains two unconstrained channels instead of only one.  The
    deployable core is exactly the same 4,844-parameter graph as PicoLite.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        path_channels = (
            *self.channel_schedule[:4],
            self.channel_schedule[4],
            *tuple(reversed(self.channel_schedule[:4])),
        )
        # concept order: boundary, texture, SDF.  At most channels 0 and 1 are
        # reserved at any one stage; deep stages use channel 0 only for shape.
        routes = (
            ((0, 0), (1, 0), (7, 0), (8, 0)),
            ((0, 1), (1, 1), (8, 1)),
            ((2, 0), (3, 0), (4, 0), (5, 0), (6, 0)),
        )
        self.channel_supervisor = SelectiveFullPathChannelSupervisor(
            path_channels, routes
        )
        inference_parameters = sum(
            parameter.numel() for parameter in self.core.parameters()
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        if inference_parameters >= self.max_parameters:
            raise ValueError("selective PicoLite inference graph exceeds 0.005M")
        if total_parameters >= self.max_parameters:
            raise ValueError("selective PicoLite training graph exceeds 0.005M")


class PicoLiteShallowSemanticDistillUNet(PicoLiteFullPathDistillUNet):
    """Matched PicoLite core with semantic supervision restricted to E1--E3.

    This is the clean shallow-supervision comparator for FullPath.  Boundary is
    read from E1--E3, texture from E1--E2, and SDF from E2--E3.  No bottleneck
    or decoder semantic feature is supervised.  The deployed core remains
    exactly identical to the 4,844-parameter FullPath model.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        path_channels = (
            *self.channel_schedule[:4],
            self.channel_schedule[4],
            *tuple(reversed(self.channel_schedule[:4])),
        )
        shallow_routes = (
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1)),
            ((1, 2), (2, 2)),
        )
        self.channel_supervisor = SelectiveFullPathChannelSupervisor(
            path_channels, shallow_routes
        )
        inference_parameters = sum(
            parameter.numel() for parameter in self.core.parameters()
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        if inference_parameters >= self.max_parameters:
            raise ValueError("shallow PicoLite inference graph exceeds 0.005M")
        if total_parameters >= self.max_parameters:
            raise ValueError("shallow PicoLite training graph exceeds 0.005M")


class PicoLiteLearnedHeadFullPathDistillUNet(PicoLiteFullPathDistillUNet):
    """Matched PicoLite core with ordinary learned full-path auxiliary heads.

    The deployable core is exactly the same 4,844-parameter graph as the fixed
    semantic-channel model.  Only training-time readers differ.  Their ability
    to mix all stage channels intentionally gives this control more auxiliary
    capacity than the fixed-channel method.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        decoder_channels = tuple(reversed(self.channel_schedule[:4]))
        path_channels = (
            *self.channel_schedule[:4],
            self.channel_schedule[4],
            *decoder_channels,
        )
        self.channel_supervisor = LearnedHeadFullPathSupervisor(path_channels)
        inference_parameters = sum(
            parameter.numel() for parameter in self.core.parameters()
        )
        if inference_parameters >= self.max_parameters:
            raise ValueError("learned-head PicoLite inference graph exceeds 0.005M")


class PicoLiteEncoderSemanticDistillUNet(PicoLiteFullPathDistillUNet):
    """Matched PicoLite core with fixed semantics on E1--E4 and bottleneck only."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        encoder_path_channels = (*self.channel_schedule[:4], self.channel_schedule[4])
        self.channel_supervisor = FullPathChannelSupervisor(encoder_path_channels)

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        segmentation, coarse, encoders, bottleneck, decoders = (
            self.core.forward_all_features(x)
        )
        outputs: dict[str, Tensor | list[Tensor]] = {"seg_logits": segmentation}
        if not return_aux:
            return outputs
        boundary, texture, sdf = self.channel_supervisor((*encoders, bottleneck))
        outputs.update(
            coarse_seg_logits=coarse,
            deep_seg_logits=[
                head(feature)
                for head, feature in zip(self.deep_supervision, decoders[:3])
            ],
            boundary_logits=boundary,
            texture_logits=texture,
            sdf_logits=sdf,
        )
        return outputs


class DilatedPyramidContext(nn.Module):
    """Bottleneck context aggregation with local, dilated, and global views."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("context channels must be positive")
        hidden = max(16, channels // 4)

        def branch(kernel_size: int, dilation: int = 1) -> nn.Sequential:
            padding = dilation if kernel_size == 3 else 0
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    hidden,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                ),
                nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
                nn.SiLU(inplace=True),
            )

        self.branches = nn.ModuleList(
            [branch(1), branch(3, 2), branch(3, 4), branch(3, 8)]
        )
        self.global_projection = branch(1)
        fused_channels = hidden * (len(self.branches) + 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, channels, 1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        spatial = [branch(x) for branch in self.branches]
        global_feature = self.global_projection(F.adaptive_avg_pool2d(x, 1))
        global_feature = F.interpolate(
            global_feature,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return x + self.fuse(torch.cat((*spatial, global_feature), dim=1))


class BoundaryRefinementHead(nn.Module):
    """Refine coarse logits using full-resolution decoder and RGB evidence."""

    def __init__(self, decoder_channels: int, image_channels: int = 3) -> None:
        super().__init__()
        if min(decoder_channels, image_channels) <= 0:
            raise ValueError("refinement channel counts must be positive")
        hidden = max(16, decoder_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels + image_channels + 1, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        # Begin from the coarse segmentation and learn only evidence-backed
        # corrections, which makes the new refinement path stable from step one.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, decoder_feature: Tensor, image: Tensor, coarse: Tensor) -> Tensor:
        if decoder_feature.shape[-2:] != image.shape[-2:] or coarse.shape[-2:] != image.shape[-2:]:
            raise ValueError("refinement inputs must have matching spatial dimensions")
        evidence = torch.cat((decoder_feature, image, torch.sigmoid(coarse)), dim=1)
        return coarse + self.refine(evidence)


class PerformanceContextUNet(nn.Module):
    """Accuracy-first U-Net teacher with context, refinement, and deep supervision.

    Boundary, texture, signed-distance, and decoder side heads are training-only.
    The deployed graph retains the wider backbone, bottleneck context block, and
    final boundary-refinement head; it intentionally prioritizes accuracy before
    a later distillation/compression stage.
    """

    inference_uses_auxiliary = False
    supports_auxiliary = True
    training_only_parameter_prefixes = (
        "boundary_auxiliary.",
        "texture_auxiliary.",
        "sdf_auxiliary.",
        "deep_supervision.",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("performance context model currently requires one output channel")
        self.backbone = UNetBackbone(in_channels, out_channels, base_channels)
        e1, e2, e3, e4 = self.backbone.encoder_channels
        bottleneck_channels = e4 * 2
        self.context = DilatedPyramidContext(bottleneck_channels)
        self.refinement = BoundaryRefinementHead(e1, in_channels)
        self.boundary_auxiliary = MultiScaleConceptHead((e1, e2, e3), hidden_channels=16)
        self.texture_auxiliary = MultiScaleConceptHead((e1, e2), hidden_channels=16)
        self.sdf_auxiliary = MultiScaleConceptHead((e2, e3), hidden_channels=16)
        # D4, D3, and D2 provide coarse-to-fine segmentation constraints. D1 is
        # already supervised by the final coarse and refined segmentation heads.
        self.deep_supervision = nn.ModuleList(
            nn.Conv2d(channels, 1, 1) for channels in (e4, e3, e2)
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        bottleneck, encoder_features = self.backbone.encode(x)
        contextual = self.context(bottleneck)
        coarse, decoder_features = self.backbone.decode_features(
            contextual, encoder_features
        )
        refined = self.refinement(decoder_features[-1], x, coarse)
        outputs: dict[str, Tensor | list[Tensor]] = {"seg_logits": refined}
        if not return_aux:
            return outputs
        outputs.update(
            coarse_seg_logits=coarse,
            deep_seg_logits=[
                head(feature)
                for head, feature in zip(self.deep_supervision, decoder_features[:3])
            ],
            boundary_logits=self.boundary_auxiliary(list(encoder_features[:3])),
            texture_logits=self.texture_auxiliary(list(encoder_features[:2])),
            sdf_logits=self.sdf_auxiliary(list(encoder_features[1:3])),
        )
        return outputs


class CompressedContextUNet(PerformanceContextUNet):
    """Narrow performance-context student intended for teacher distillation.

    The topology deliberately matches the accuracy-first teacher so its coarse
    and decoder-scale predictions can be distilled directly.  Width 18 keeps
    the deployed graph close to the earlier 3M-parameter target while retaining
    the context and refinement paths that improved the teacher's Dice score.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 18,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)


class CompressedContextFullPathUNet(CompressedContextUNet):
    """3M-class context U-Net with semantic channels across the complete U path.

    Boundary, texture, and signed-distance concepts occupy channels 0, 1, and 2
    at E1--E4, the contextual bottleneck, and D4--D1.  The affine readouts and
    decoder side heads are used only during training and are excluded from the
    reported deployment parameter count.
    """

    training_only_parameter_prefixes = (
        "channel_supervisor.",
        "deep_supervision.",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 18,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        # Remove the older projection-based concept heads so the formal 3M
        # model uses exactly the same fixed-channel principle as Micro/PicoLite.
        del self.boundary_auxiliary
        del self.texture_auxiliary
        del self.sdf_auxiliary
        e1, e2, e3, e4 = self.backbone.encoder_channels
        bottleneck = 2 * e4
        self.channel_supervisor = FullPathChannelSupervisor(
            (e1, e2, e3, e4, bottleneck, e4, e3, e2, e1)
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        bottleneck, encoders = self.backbone.encode(x)
        contextual = self.context(bottleneck)
        coarse, decoders = self.backbone.decode_features(contextual, encoders)
        refined = self.refinement(decoders[-1], x, coarse)
        outputs: dict[str, Tensor | list[Tensor]] = {"seg_logits": refined}
        if not return_aux:
            return outputs
        boundary, texture, sdf = self.channel_supervisor(
            (*encoders, contextual, *decoders)
        )
        outputs.update(
            coarse_seg_logits=coarse,
            deep_seg_logits=[
                head(feature)
                for head, feature in zip(self.deep_supervision, decoders[:3])
            ],
            boundary_logits=boundary,
            texture_logits=texture,
            sdf_logits=sdf,
        )
        return outputs


class AmbiguityContextUNet(CompressedContextUNet):
    """Compressed student with a training-only multiscale ambiguity category."""

    training_only_parameter_prefixes = (
        *PerformanceContextUNet.training_only_parameter_prefixes,
        "ambiguity_auxiliary.",
    )

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 18,
    ) -> None:
        super().__init__(in_channels, out_channels, base_channels)
        e1, e2, e3 = self.backbone.encoder_channels[:3]
        self.ambiguity_auxiliary = MultiScaleConceptHead(
            (e1, e2, e3), hidden_channels=16
        )

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> dict[str, Tensor | list[Tensor]]:
        bottleneck, encoder_features = self.backbone.encode(x)
        contextual = self.context(bottleneck)
        coarse, decoder_features = self.backbone.decode_features(
            contextual, encoder_features
        )
        refined = self.refinement(decoder_features[-1], x, coarse)
        outputs: dict[str, Tensor | list[Tensor]] = {"seg_logits": refined}
        if not return_aux:
            return outputs
        outputs.update(
            coarse_seg_logits=coarse,
            deep_seg_logits=[
                head(feature)
                for head, feature in zip(self.deep_supervision, decoder_features[:3])
            ],
            boundary_logits=self.boundary_auxiliary(list(encoder_features[:3])),
            texture_logits=self.texture_auxiliary(list(encoder_features[:2])),
            sdf_logits=self.sdf_auxiliary(list(encoder_features[1:3])),
            ambiguity_logits=self.ambiguity_auxiliary(list(encoder_features[:3])),
        )
        return outputs


def build_model(model_name: str, **kwargs: int) -> UNet | TriFeatureUNet:
    """构建阶段一候选模型。

    三个辅助监督候选共享完全相同的网络结构，只改变训练目标和辅助权重
    调度。这样能够把性能差异归因于训练方法，而不是参数量变化。
    """

    normalized = model_name.strip().lower().replace("-", "_")
    if normalized == "unet":
        return UNet(**kwargs)
    if normalized == "tiny_unet_5k":
        return TinyUNet5K(**kwargs)
    if normalized == "tri_gate":
        return ConceptGatedUNet(**kwargs)
    if normalized == "quad_gate_lite":
        return QuadGateLiteUNet(**kwargs)
    if normalized == "lite_sdf_distill":
        return LiteSDFDistillUNet(**kwargs)
    if normalized == "ultralite_sdf_distill":
        return UltraLiteSDFDistillUNet(**kwargs)
    if normalized == "nanolite_sdf_distill":
        return NanoLiteSDFDistillUNet(**kwargs)
    if normalized == "microlite_sdf_distill":
        return MicroLiteSDFDistillUNet(**kwargs)
    if normalized == "picolite_sdf_distill":
        return PicoLiteSDFDistillUNet(**kwargs)
    if normalized == "microlite_fullpath_distill":
        return MicroLiteFullPathDistillUNet(**kwargs)
    if normalized == "picolite_fullpath_distill":
        return PicoLiteFullPathDistillUNet(**kwargs)
    if normalized == "picolite_selective_fullpath_distill":
        return PicoLiteSelectiveFullPathDistillUNet(**kwargs)
    if normalized == "picolite_shallow_semantic_distill":
        return PicoLiteShallowSemanticDistillUNet(**kwargs)
    if normalized == "picolite_learned_head_fullpath_distill":
        return PicoLiteLearnedHeadFullPathDistillUNet(**kwargs)
    if normalized == "picolite_encoder_semantic_distill":
        return PicoLiteEncoderSemanticDistillUNet(**kwargs)
    if normalized == "performance_context":
        return PerformanceContextUNet(**kwargs)
    if normalized == "compressed_context":
        return CompressedContextUNet(**kwargs)
    if normalized == "compressed_context_fullpath":
        return CompressedContextFullPathUNet(**kwargs)
    if normalized == "ambiguity_context":
        return AmbiguityContextUNet(**kwargs)
    if normalized in AUXILIARY_MODEL_NAMES:
        return TriFeatureUNet(**kwargs)
    expected = ", ".join((*BASELINE_MODEL_NAMES, *AUXILIARY_MODEL_NAMES))
    raise ValueError(f"unknown model {model_name!r}; expected one of: {expected}")


__all__ = [
    "UNET_BASE_CHANNELS",
    "TRI_UNET_BASE_CHANNELS",
    "BASELINE_MODEL_NAMES",
    "AUXILIARY_MODEL_NAMES",
    "RETIRING_AUXILIARY_MODEL_NAMES",
    "DoubleConv",
    "UNetBackbone",
    "UNet",
    "TriFeatureUNet",
    "ConceptResidualGate",
    "ConceptGatedUNet",
    "MultiScaleConceptHead",
    "DirectConceptHead",
    "split_feature_channels",
    "split_quad_feature_channels",
    "QuadGateLiteUNet",
    "LiteSDFDistillUNet",
    "DepthwiseResidualBlock",
    "LiteChannelGate",
    "LitePyramidContext",
    "TinyOutputRefinement",
    "UltraLiteCore",
    "UltraLiteSDFDistillUNet",
    "NanoLiteSDFDistillUNet",
    "MicroLiteSDFDistillUNet",
    "PicoLiteSDFDistillUNet",
    "FullPathChannelSupervisor",
    "LearnedHeadFullPathSupervisor",
    "SelectiveFullPathChannelSupervisor",
    "MicroLiteFullPathDistillUNet",
    "PicoLiteFullPathDistillUNet",
    "PicoLiteSelectiveFullPathDistillUNet",
    "PicoLiteShallowSemanticDistillUNet",
    "PicoLiteLearnedHeadFullPathDistillUNet",
    "PicoLiteEncoderSemanticDistillUNet",
    "TinyUNet5K",
    "DilatedPyramidContext",
    "BoundaryRefinementHead",
    "PerformanceContextUNet",
    "CompressedContextUNet",
    "CompressedContextFullPathUNet",
    "AmbiguityContextUNet",
    "build_model",
]
