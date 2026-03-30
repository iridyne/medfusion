import torch

from med_core.models.three_phase_ct_fusion import ThreePhaseCTFusionModel


def test_three_phase_ct_fusion_model_returns_probability_and_risk() -> None:
    model = ThreePhaseCTFusionModel(
        phase_feature_dim=32,
        clinical_input_dim=8,
        clinical_hidden_dim=16,
        fusion_hidden_dim=24,
        phase_fusion_type="concatenate",
        share_phase_encoder=False,
        use_risk_head=True,
    )

    outputs = model(
        arterial=torch.randn(2, 1, 8, 32, 32),
        portal=torch.randn(2, 1, 8, 32, 32),
        noncontrast=torch.randn(2, 1, 8, 32, 32),
        clinical=torch.randn(2, 8),
    )

    assert outputs["logits"].shape == (2, 2)
    assert outputs["probability"].shape == (2,)
    assert outputs["risk_score"].shape == (2,)
    assert outputs["fused_features"].shape[0] == 2


def test_three_phase_ct_fusion_model_supports_mean_fusion_and_freeze() -> None:
    model = ThreePhaseCTFusionModel(
        phase_feature_dim=16,
        clinical_input_dim=4,
        clinical_hidden_dim=8,
        fusion_hidden_dim=12,
        phase_fusion_type="mean",
        share_phase_encoder=True,
        freeze_phase_encoder=True,
        use_risk_head=False,
    )

    outputs = model(
        arterial=torch.randn(2, 1, 8, 16, 16),
        portal=torch.randn(2, 1, 8, 16, 16),
        noncontrast=torch.randn(2, 1, 8, 16, 16),
        clinical=torch.randn(2, 4),
    )

    assert outputs["logits"].shape == (2, 2)
    assert outputs["risk_score"].shape == (2,)
    assert model.arterial_encoder is model.portal_encoder
    assert not any(
        parameter.requires_grad for parameter in model.arterial_encoder.parameters()
    )
