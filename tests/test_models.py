"""Shape and structural tests for both U-Net variants."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import src.models as models
from src.models import (
    AmbiguityContextUNet,
    CompressedContextUNet,
    CompressedContextFullPathUNet,
    ConceptGatedUNet,
    LiteSDFDistillUNet,
    PerformanceContextUNet,
    UltraLiteSDFDistillUNet,
    NanoLiteSDFDistillUNet,
    MicroLiteSDFDistillUNet,
    PicoLiteSDFDistillUNet,
    MicroLiteFullPathDistillUNet,
    PicoLiteFullPathDistillUNet,
    PicoLiteSelectiveFullPathDistillUNet,
    PicoLiteShallowSemanticDistillUNet,
    PicoLiteLearnedHeadFullPathDistillUNet,
    PicoLiteEncoderSemanticDistillUNet,
    TRI_UNET_BASE_CHANNELS,
    UNET_BASE_CHANNELS,
    TriFeatureUNet,
    UNet,
    UNetBackbone,
    build_model,
    split_feature_channels,
)
from src.utils import count_inference_parameters, count_trainable_parameters


@pytest.mark.parametrize("model_class", [UNet, TriFeatureUNet])
def test_default_models_preserve_256_shape(model_class: type[nn.Module]) -> None:
    model = model_class().eval()
    image = torch.randn(2, 3, 256, 256)
    with torch.inference_mode():
        outputs = model(image, return_aux=isinstance(model, TriFeatureUNet))
    assert outputs["seg_logits"].shape == (2, 1, 256, 256)
    if isinstance(model, TriFeatureUNet):
        assert [value.shape for value in outputs["boundary_logits"]] == [
            (2, 1, 256, 256),
            (2, 1, 128, 128),
            (2, 1, 64, 64),
        ]
        assert [value.shape for value in outputs["texture_logits"]] == [
            (2, 1, 256, 256),
            (2, 1, 128, 128),
            (2, 1, 64, 64),
        ]


def test_tri_feature_channel_slices_and_concept_heads_are_exact() -> None:
    feature = torch.arange(32.0).reshape(1, 32, 1, 1)
    boundary, region, free = split_feature_channels(feature)
    assert (boundary.shape[1], region.shape[1], free.shape[1]) == (8, 8, 16)
    assert torch.equal(boundary.flatten(), torch.arange(0.0, 8.0))
    assert torch.equal(region.flatten(), torch.arange(8.0, 16.0))
    assert torch.equal(free.flatten(), torch.arange(16.0, 32.0))

    model = TriFeatureUNet()
    assert model.channel_widths == (
        (6, 6, 12),
        (12, 12, 24),
        (24, 24, 48),
        (48, 48, 96),
    )
    assert isinstance(model.boundary_auxiliary, models.MultiScaleConceptHead)
    assert isinstance(model.texture_auxiliary, models.MultiScaleConceptHead)
    assert model.boundary_auxiliary.in_channels == (6, 12, 24)
    assert model.texture_auxiliary.in_channels == (6, 12, 24)


def test_default_widths_and_parameter_counts_are_model_specific() -> None:
    baseline = build_model("unet")
    optimized = build_model("tri_unet")

    assert baseline.backbone.base_channels == UNET_BASE_CHANNELS == 32
    assert optimized.backbone.base_channels == TRI_UNET_BASE_CHANNELS == 24
    assert baseline.backbone.encoder_channels == (32, 64, 128, 256)
    assert optimized.backbone.encoder_channels == (24, 48, 96, 192)
    assert count_trainable_parameters(baseline) == 7_763_041
    assert count_inference_parameters(baseline) == 7_763_041
    assert count_trainable_parameters(optimized) > 4_368_073
    assert count_inference_parameters(optimized) == 4_368_073


@pytest.mark.parametrize(
    ("base_channels", "unet_parameters"),
    [
        (24, 4_368_073),
        (16, 1_942_577),
        (12, 1_093_381),
        (8, 486_553),
    ],
)
def test_width_sweep_variants_have_exact_parameter_counts(
    base_channels: int,
    unet_parameters: int,
) -> None:
    baseline = build_model("unet", base_channels=base_channels)
    optimized = build_model("tri_unet", base_channels=base_channels)

    assert baseline.backbone.base_channels == base_channels
    assert optimized.backbone.base_channels == base_channels
    assert count_trainable_parameters(baseline) == unet_parameters
    assert count_inference_parameters(baseline) == unet_parameters
    assert count_trainable_parameters(optimized) > unet_parameters
    assert count_inference_parameters(optimized) == unet_parameters
    image = torch.randn(1, 3, 32, 32)
    baseline.eval()
    optimized.eval()
    with torch.inference_mode():
        baseline_output = baseline(image)
        optimized_output = optimized(image, return_aux=True)
    assert baseline_output["seg_logits"].shape == (1, 1, 32, 32)
    assert optimized_output["seg_logits"].shape == (1, 1, 32, 32)
    assert optimized.channel_widths[0] == (
        base_channels // 4,
        base_channels // 4,
        base_channels // 2,
    )


def test_both_models_use_the_same_main_backbone_definition() -> None:
    baseline = UNet(base_channels=4)
    tri_model = TriFeatureUNet(base_channels=4)
    assert isinstance(baseline.backbone, UNetBackbone)
    assert type(baseline.backbone) is type(tri_model.backbone)
    baseline_shapes = {
        name: parameter.shape for name, parameter in baseline.backbone.named_parameters()
    }
    tri_shapes = {
        name: parameter.shape for name, parameter in tri_model.backbone.named_parameters()
    }
    assert baseline_shapes == tri_shapes

    # Explicit widths remain supported so checkpoints from the earlier base-32
    # TriFeature model can still be reconstructed.
    assert TriFeatureUNet(base_channels=32).backbone.base_channels == 32


def test_return_aux_false_never_calls_readout_heads() -> None:
    model = TriFeatureUNet(base_channels=4).eval()
    called: list[str] = []
    handles = [
        head.register_forward_hook(
            lambda _module, _inputs, _output, name=name: called.append(name)
        )
        for name, head in (
            ("boundary", model.boundary_auxiliary),
            ("texture", model.texture_auxiliary),
        )
    ]
    try:
        with torch.inference_mode():
            outputs = model(torch.randn(1, 3, 32, 32), return_aux=False)
    finally:
        for handle in handles:
            handle.remove()
    assert set(outputs) == {"seg_logits"}
    assert called == []


def test_concept_gate_starts_as_identity_but_remains_in_inference_path() -> None:
    torch.manual_seed(17)
    tri = TriFeatureUNet(base_channels=4).eval()
    torch.manual_seed(17)
    gated = ConceptGatedUNet(base_channels=4).eval()
    image = torch.randn(1, 3, 32, 32)
    called: list[str] = []
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: called.append(name)
        )
        for name, module in (
            ("boundary", gated.boundary_auxiliary),
            ("texture", gated.texture_auxiliary),
            ("gate", gated.concept_gates[0]),
        )
    ]
    try:
        with torch.inference_mode():
            baseline = tri(image, return_aux=False)["seg_logits"]
            initial = gated(image, return_aux=False)["seg_logits"]
            gated.concept_gates[0].projection.bias.fill_(0.5)
            modulated = gated(image, return_aux=False)["seg_logits"]
    finally:
        for handle in handles:
            handle.remove()

    assert called == ["boundary", "texture", "gate", "boundary", "texture", "gate"]
    assert torch.equal(initial, baseline)
    assert not torch.equal(modulated, initial)
    assert count_inference_parameters(gated) == count_trainable_parameters(gated)


def test_multiscale_concept_head_propagates_deep_context_to_shallow_output() -> None:
    """Removing the top-down path must disconnect E3 from the E1 readout."""

    torch.manual_seed(7)
    head = models.MultiScaleConceptHead((2, 4, 8), hidden_channels=4).eval()
    features = [
        torch.randn(1, 2, 16, 16, requires_grad=True),
        torch.randn(1, 4, 8, 8, requires_grad=True),
        torch.randn(1, 8, 4, 4, requires_grad=True),
    ]

    shallow_logit = head(features)[0]
    shallow_logit.square().mean().backward()

    assert shallow_logit.shape == (1, 1, 16, 16)
    assert features[2].grad is not None
    assert torch.count_nonzero(features[2].grad) > 0


def test_model_factory_and_input_size_validation() -> None:
    assert isinstance(build_model("unet", base_channels=4), UNet)
    assert isinstance(build_model("tri-unet", base_channels=4), TriFeatureUNet)
    assert isinstance(build_model("tri-gate", base_channels=4), ConceptGatedUNet)
    assert isinstance(
        build_model("lite-sdf-distill", base_channels=20), LiteSDFDistillUNet
    )
    assert isinstance(
        build_model("performance-context", base_channels=32),
        PerformanceContextUNet,
    )
    assert isinstance(
        build_model("compressed-context", base_channels=18),
        CompressedContextUNet,
    )
    assert isinstance(
        build_model("compressed-context-fullpath", base_channels=18),
        CompressedContextFullPathUNet,
    )
    assert isinstance(
        build_model("ambiguity-context", base_channels=18),
        AmbiguityContextUNet,
    )
    assert isinstance(
        build_model("ultralite-sdf-distill", base_channels=18),
        UltraLiteSDFDistillUNet,
    )
    assert isinstance(
        build_model("nanolite-sdf-distill", base_channels=9),
        NanoLiteSDFDistillUNet,
    )
    assert isinstance(
        build_model("microlite-sdf-distill", base_channels=4),
        MicroLiteSDFDistillUNet,
    )
    assert isinstance(
        build_model("picolite-sdf-distill", base_channels=2),
        PicoLiteSDFDistillUNet,
    )
    assert isinstance(
        build_model("microlite-fullpath-distill", base_channels=6),
        MicroLiteFullPathDistillUNet,
    )
    assert isinstance(
        build_model("picolite-fullpath-distill", base_channels=4),
        PicoLiteFullPathDistillUNet,
    )
    with pytest.raises(ValueError, match="unknown model"):
        build_model("transformer")
    with pytest.raises(ValueError, match="divisible by 16"):
        UNet(base_channels=4)(torch.randn(1, 3, 31, 32))


def test_lite_sdf_distill_has_a_bare_base20_inference_graph() -> None:
    model = LiteSDFDistillUNet(base_channels=20).eval()
    baseline = UNet(base_channels=20).eval()
    image = torch.randn(1, 3, 32, 32)
    called: list[str] = []
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: called.append(name)
        )
        for name, module in (
            ("boundary", model.boundary_auxiliary),
            ("texture", model.texture_auxiliary),
            ("sdf", model.sdf_auxiliary),
        )
    ]
    try:
        with torch.inference_mode():
            deployed = model(image, return_aux=False)
            trained = model(image, return_aux=True)
    finally:
        for handle in handles:
            handle.remove()

    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert called == ["boundary", "texture", "sdf"]
    assert count_inference_parameters(model) == count_inference_parameters(baseline)
    assert count_inference_parameters(model) == 3_034_141


def test_compressed_context_fullpath_supervises_all_nine_u_stages() -> None:
    model = CompressedContextFullPathUNet(base_channels=18).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        deployed = model(image, return_aux=False)
        trained = model(image, return_aux=True)

    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert len(trained["boundary_logits"]) == 9
    assert len(trained["texture_logits"]) == 9
    assert len(trained["sdf_logits"]) == 9
    assert len(trained["deep_seg_logits"]) == 3
    assert count_inference_parameters(model) == 3_170_954
    assert count_trainable_parameters(model) > count_inference_parameters(model)


def test_ultralite_deployable_graph_is_below_half_mib_fp32() -> None:
    model = UltraLiteSDFDistillUNet(base_channels=18).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        deployed = model(image, return_aux=False)
        trained = model(image, return_aux=True)

    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert len(trained["boundary_logits"]) == 3
    assert len(trained["texture_logits"]) == 2
    assert len(trained["sdf_logits"]) == 2
    assert count_inference_parameters(model) == 121_063
    assert count_inference_parameters(model) * 4 < 512 * 1024
    assert [value.shape[-2:] for value in trained["boundary_logits"]] == [
        (32, 32), (16, 16), (8, 8)
    ]
    assert [value.shape[-2:] for value in trained["texture_logits"]] == [
        (32, 32), (16, 16)
    ]
    assert [value.shape[-2:] for value in trained["sdf_logits"]] == [
        (16, 16), (8, 8)
    ]


def test_nanolite_deployable_graph_is_below_point_zero_four_million() -> None:
    model = NanoLiteSDFDistillUNet().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), return_aux=False)
    assert output["seg_logits"].shape == (1, 1, 32, 32)
    assert count_inference_parameters(model) == 34_213
    assert count_inference_parameters(model) < 40_000


def test_microlite_entire_training_graph_is_below_point_zero_one_million() -> None:
    model = MicroLiteSDFDistillUNet().eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert outputs["seg_logits"].shape == (1, 1, 32, 32)
    assert count_inference_parameters(model) == 8_713
    assert count_trainable_parameters(model) == 9_952
    assert count_inference_parameters(model) < 10_000
    assert count_trainable_parameters(model) < 10_000


def test_picolite_entire_training_graph_is_below_point_zero_zero_five_million() -> None:
    model = PicoLiteSDFDistillUNet().eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert outputs["seg_logits"].shape == (1, 1, 32, 32)
    assert count_inference_parameters(model) == 3_063
    assert count_trainable_parameters(model) == 4_078
    assert count_inference_parameters(model) < 5_000
    assert count_trainable_parameters(model) < 5_000


def test_microlite_fullpath_supervises_every_u_stage_under_ten_thousand() -> None:
    model = MicroLiteFullPathDistillUNet().eval()
    with torch.inference_mode():
        deployed = model(torch.randn(1, 3, 32, 32), return_aux=False)
        trained = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert len(trained["boundary_logits"]) == 9
    assert len(trained["texture_logits"]) == 9
    assert len(trained["sdf_logits"]) == 9
    assert len(trained["deep_seg_logits"]) == 3
    assert count_inference_parameters(model) == 9_824
    assert count_trainable_parameters(model) == 9_923
    assert count_inference_parameters(model) < 10_000
    assert count_trainable_parameters(model) < 10_000


def test_picolite_fullpath_supervises_every_u_stage_under_five_thousand() -> None:
    model = PicoLiteFullPathDistillUNet().eval()
    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert outputs["seg_logits"].shape == (1, 1, 32, 32)
    assert len(outputs["boundary_logits"]) == 9
    assert len(outputs["texture_logits"]) == 9
    assert len(outputs["sdf_logits"]) == 9
    assert len(outputs["deep_seg_logits"]) == 3
    assert count_inference_parameters(model) == 4_844
    assert count_trainable_parameters(model) == 4_928
    assert count_inference_parameters(model) < 5_000
    assert count_trainable_parameters(model) < 5_000


def test_picolite_selective_fullpath_preserves_capacity_and_budget() -> None:
    model = PicoLiteSelectiveFullPathDistillUNet().eval()
    with torch.inference_mode():
        deployed = model(torch.randn(1, 3, 32, 32), return_aux=False)
        trained = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert [len(trained[name]) for name in (
        "boundary_logits", "texture_logits", "sdf_logits"
    )] == [4, 3, 5]
    assert len(trained["deep_seg_logits"]) == 3
    assert count_inference_parameters(model) == 4_844
    assert count_trainable_parameters(model) == 4_898
    assert count_trainable_parameters(model) < 5_000


def test_picolite_shallow_semantic_matches_fullpath_deployment_core() -> None:
    model = PicoLiteShallowSemanticDistillUNet().eval()
    with torch.inference_mode():
        deployed = model(torch.randn(1, 3, 32, 32), return_aux=False)
        trained = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert [len(trained[name]) for name in (
        "boundary_logits", "texture_logits", "sdf_logits"
    )] == [3, 2, 2]
    assert count_inference_parameters(model) == 4_844
    assert count_trainable_parameters(model) == 4_888
    assert count_trainable_parameters(model) < 5_000


def test_picolite_learned_head_control_preserves_deployment_core() -> None:
    model = PicoLiteLearnedHeadFullPathDistillUNet().eval()
    fixed = PicoLiteFullPathDistillUNet().eval()
    with torch.inference_mode():
        deployed = model(torch.randn(1, 3, 32, 32), return_aux=False)
        trained = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert [len(trained[name]) for name in (
        "boundary_logits", "texture_logits", "sdf_logits"
    )] == [9, 9, 9]
    assert all(
        logit.shape[1] == 1
        for name in ("boundary_logits", "texture_logits", "sdf_logits")
        for logit in trained[name]
    )
    assert count_inference_parameters(model) == 4_844
    assert count_inference_parameters(model) == count_inference_parameters(fixed)
    assert count_trainable_parameters(model) == 5_123


def test_tiny_unet_5k_is_a_clean_parameter_matched_baseline() -> None:
    model = build_model("tiny_unet_5k", base_channels=4).eval()
    fullpath = PicoLiteFullPathDistillUNet().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert set(output) == {"seg_logits"}
    assert output["seg_logits"].shape == (1, 1, 32, 32)
    assert count_trainable_parameters(model) == 4_844
    assert count_inference_parameters(model) == 4_844
    assert count_inference_parameters(model) == count_inference_parameters(fullpath)


def test_picolite_encoder_semantic_control_preserves_deployment_core() -> None:
    model = PicoLiteEncoderSemanticDistillUNet().eval()
    fixed = PicoLiteFullPathDistillUNet().eval()
    with torch.inference_mode():
        deployed = model(torch.randn(1, 3, 32, 32), return_aux=False)
        trained = model(torch.randn(1, 3, 32, 32), return_aux=True)
    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert [len(trained[name]) for name in (
        "boundary_logits", "texture_logits", "sdf_logits"
    )] == [5, 5, 5]
    assert count_inference_parameters(model) == 4_844
    assert count_inference_parameters(model) == count_inference_parameters(fixed)
    assert count_trainable_parameters(model) == 4_904


def test_performance_context_keeps_only_accuracy_modules_at_inference() -> None:
    model = PerformanceContextUNet(base_channels=8).eval()
    image = torch.randn(1, 3, 32, 32)
    called: list[str] = []
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: called.append(name)
        )
        for name, module in (
            ("boundary", model.boundary_auxiliary),
            ("texture", model.texture_auxiliary),
            ("sdf", model.sdf_auxiliary),
            ("deep", model.deep_supervision[0]),
        )
    ]
    try:
        with torch.inference_mode():
            deployed = model(image, return_aux=False)
            trained = model(image, return_aux=True)
    finally:
        for handle in handles:
            handle.remove()

    assert set(deployed) == {"seg_logits"}
    assert deployed["seg_logits"].shape == (1, 1, 32, 32)
    assert called == ["deep", "boundary", "texture", "sdf"]
    assert [value.shape[-2:] for value in trained["deep_seg_logits"]] == [
        (4, 4),
        (8, 8),
        (16, 16),
    ]
    assert trained["coarse_seg_logits"].shape == (1, 1, 32, 32)
    assert count_trainable_parameters(model) > count_inference_parameters(model)


def test_compressed_context_stays_near_three_million_inference_parameters() -> None:
    model = CompressedContextUNet().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), return_aux=False)
    assert output["seg_logits"].shape == (1, 1, 32, 32)
    assert 3_000_000 < count_inference_parameters(model) < 3_250_000
    assert count_trainable_parameters(model) > count_inference_parameters(model)


def test_ambiguity_context_is_training_only_and_multiscale() -> None:
    model = AmbiguityContextUNet(base_channels=8).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        deployed = model(image, return_aux=False)
        trained = model(image, return_aux=True)
    assert set(deployed) == {"seg_logits"}
    assert [value.shape[-2:] for value in trained["ambiguity_logits"]] == [
        (32, 32), (16, 16), (8, 8)
    ]
    assert count_inference_parameters(model) == count_inference_parameters(
        CompressedContextUNet(base_channels=8)
    )
