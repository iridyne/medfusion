import pytest
import torch

from med_core.models.doctor_interest import (
    DoctorInterestMapHead,
    InterestGuidedPooling3D,
    TopKLocalFocus3D,
    build_body_mask_prior,
    compute_doctor_interest_losses,
)


def test_doctor_interest_head_returns_score_and_prob_maps() -> None:
    head = DoctorInterestMapHead(in_channels=16, hidden_channels=8, temperature=6.0)
    feature_map = torch.randn(2, 16, 4, 8, 8)
    outputs = head(feature_map)
    assert outputs["score_map"].shape == (2, 1, 4, 8, 8)
    assert outputs["prob_map"].shape == (2, 1, 4, 8, 8)
    assert torch.allclose(
        outputs["prob_map"].view(2, -1).sum(dim=1),
        torch.ones(2),
        atol=1e-5,
    )


def test_interest_guided_pooling_returns_channel_vector() -> None:
    pooling = InterestGuidedPooling3D()
    feature_map = torch.randn(2, 16, 4, 8, 8)
    prob_map = torch.softmax(
        torch.randn(2, 1, 4, 8, 8).view(2, -1),
        dim=1,
    ).view(2, 1, 4, 8, 8)
    pooled = pooling(feature_map, prob_map)
    assert pooled.shape == (2, 16)


def test_topk_focus_returns_centers_scores_and_features() -> None:
    focus = TopKLocalFocus3D(k=3, patch_size=(2, 2, 2), projection_dim=12)
    feature_map = torch.randn(2, 16, 4, 8, 8)
    score_map = torch.rand(2, 1, 4, 8, 8)
    outputs = focus(feature_map, score_map)
    assert outputs["centers"].shape == (2, 3, 3)
    assert outputs["scores"].shape == (2, 3)
    assert outputs["feature"].shape == (2, 12)


def test_topk_focus_rejects_multi_channel_score_maps() -> None:
    focus = TopKLocalFocus3D(k=2, patch_size=(2, 2, 2), projection_dim=8)
    feature_map = torch.randn(1, 16, 2, 4, 4)
    score_map = torch.rand(1, 2, 2, 4, 4)

    with pytest.raises(ValueError, match="single-channel score_map"):
        focus(feature_map, score_map)


def test_topk_focus_uses_available_positions_on_tiny_maps() -> None:
    focus = TopKLocalFocus3D(k=5, patch_size=(2, 2, 2), projection_dim=8)
    feature_map = torch.randn(1, 16, 1, 1, 2)
    score_map = torch.rand(1, 1, 1, 1, 2)

    outputs = focus(feature_map, score_map)

    assert outputs["centers"].shape == (1, 2, 3)
    assert outputs["scores"].shape == (1, 2)
    assert outputs["feature"].shape == (1, 8)


def test_topk_focus_reuses_same_instance_across_different_map_sizes() -> None:
    focus = TopKLocalFocus3D(k=4, patch_size=(2, 2, 2), projection_dim=8)

    normal_outputs = focus(torch.randn(1, 16, 3, 4, 4), torch.rand(1, 1, 3, 4, 4))
    tiny_outputs = focus(torch.randn(1, 16, 1, 1, 2), torch.rand(1, 1, 1, 1, 2))

    assert normal_outputs["feature"].shape == (1, 8)
    assert tiny_outputs["feature"].shape == (1, 8)


def test_body_mask_prior_suppresses_outer_air() -> None:
    volume = torch.full((1, 1, 4, 8, 8), -500.0)
    volume[:, :, :, 2:6, 2:6] = 40.0
    prior = build_body_mask_prior(volume, threshold_hu=-200.0, border_ratio=0.1)
    assert prior.shape == (1, 1, 4, 8, 8)
    assert float(prior[:, :, :, 0, 0].mean()) < float(prior[:, :, :, 3, 3].mean())


def test_compute_doctor_interest_losses_returns_named_terms() -> None:
    score_map = torch.rand(2, 1, 4, 8, 8, requires_grad=True)
    prob_map = torch.softmax(score_map.view(2, -1), dim=1).view_as(score_map)
    teacher_map = torch.rand(2, 1, 4, 8, 8)
    loss = compute_doctor_interest_losses(
        prob_map=prob_map,
        teacher_map=teacher_map,
        augmented_prob_map=prob_map.detach(),
        topk_centers=torch.tensor([[[0, 1, 1], [1, 2, 2], [2, 3, 3]]] * 2),
        body_prior=torch.ones_like(prob_map),
        cam_align_weight=0.05,
        consistency_weight=0.02,
        sparse_weight=0.01,
        diverse_weight=0.01,
        body_prior_weight=0.02,
    )
    assert set(loss["components"]) == {
        "cam_align",
        "consistency",
        "sparse",
        "diverse",
        "body_prior",
    }
    assert loss["total"].requires_grad


def test_compute_doctor_interest_losses_zeroes_diversity_for_single_center() -> None:
    prob_map = torch.softmax(torch.rand(1, 1, 2, 2, 2).view(1, -1), dim=1).view(1, 1, 2, 2, 2)
    teacher_map = torch.rand(1, 1, 2, 2, 2)

    loss = compute_doctor_interest_losses(
        prob_map=prob_map,
        teacher_map=teacher_map,
        augmented_prob_map=None,
        topk_centers=torch.tensor([[[0, 0, 0]]]),
        body_prior=None,
        cam_align_weight=0.05,
        consistency_weight=0.02,
        sparse_weight=0.01,
        diverse_weight=0.01,
        body_prior_weight=0.02,
    )

    assert loss["components"]["diverse"] == 0


def test_compute_doctor_interest_losses_zeroes_diversity_for_well_separated_centers() -> None:
    prob_map = torch.softmax(torch.rand(1, 1, 2, 2, 2).view(1, -1), dim=1).view(1, 1, 2, 2, 2)
    teacher_map = torch.rand(1, 1, 2, 2, 2)

    loss = compute_doctor_interest_losses(
        prob_map=prob_map,
        teacher_map=teacher_map,
        augmented_prob_map=None,
        topk_centers=torch.tensor([[[0, 0, 0], [0, 0, 3], [0, 3, 3]]]),
        body_prior=None,
        cam_align_weight=0.05,
        consistency_weight=0.02,
        sparse_weight=0.01,
        diverse_weight=0.01,
        body_prior_weight=0.02,
    )

    assert loss["components"]["diverse"] == 0
