# Doctor Interest Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explanation-ready `doctor_interest_map` capability to the native three-phase CT + clinical MVI path so the model can emit stable clinician-facing focus regions without manual ROI annotations.

**Architecture:** Keep the existing `three_phase_ct_fusion` entrypoint and extend it with a lightweight per-phase interest-map head, interest-guided pooling, optional Top-K local focus features, and training-time regularizers. Reuse the current three-phase dataset, training loop, and result-builder flow so `validate-config`, `train --config`, and `build-results --config` stay unchanged while the artifact contract grows to include doctor-interest overlays and focus-region JSON.

**Tech Stack:** Python 3.13, PyTorch, pytest, YAML dataclass configs, existing MedFusion three-phase CT run layout and visualization helpers.

---

## File Map

### New files

- `med_core/models/doctor_interest.py`
  - Home for reusable 3D doctor-interest modules and regularizer helpers:
  - `DoctorInterestMapHead`
  - `InterestGuidedPooling3D`
  - `TopKLocalFocus3D`
  - `build_body_mask_prior`
  - `compute_doctor_interest_losses`
- `tests/test_doctor_interest_modules.py`
  - Unit tests for the new reusable modules and losses.

### Modified files

- `med_core/configs/base_config.py`
  - Add config surface for doctor-interest modules and losses.
- `med_core/configs/validation.py`
  - Validate the new config branches.
- `configs/demo/three_phase_ct_mvi_dr_z.yaml`
  - Turn on the new path in the canonical demo config.
- `tests/test_three_phase_mainline_config.py`
  - Cover config loading and round-tripping for the new fields.
- `tests/test_config_validation.py`
  - Cover invalid doctor-interest config combinations.
- `med_core/models/three_phase_ct_fusion.py`
  - Wire doctor-interest heads and new forward outputs into the native three-phase model.
- `tests/test_three_phase_ct_fusion_model.py`
  - Cover forward outputs, map shapes, and focus-region metadata.
- `med_core/cli/train.py`
  - Compute and log doctor-interest regularization loss in the native three-phase loop.
- `tests/test_three_phase_mainline_train.py`
  - Confirm training writes checkpoints and doctor-interest-aware history.
- `med_core/postprocessing/results.py`
  - Export doctor-interest overlays and focus-region artifacts.
- `tests/test_three_phase_mainline_build_results.py`
  - Verify artifact files, manifest payloads, and report wording.
- `tests/test_three_phase_mainline_e2e.py`
  - Guard the upgraded path end-to-end.

### Existing files to read before coding

- `docs/superpowers/specs/2026-04-17-doctor-interest-map-design.md`
- `med_core/models/three_phase_ct_fusion.py`
- `med_core/cli/train.py`
- `med_core/postprocessing/results.py`
- `med_core/shared/visualization/heatmaps.py`
- `med_core/attention_supervision/cam_supervision.py`

## Task 1: Add config surface for doctor-interest modules and losses

**Files:**
- Modify: `med_core/configs/base_config.py`
- Modify: `med_core/configs/validation.py`
- Modify: `configs/demo/three_phase_ct_mvi_dr_z.yaml`
- Test: `tests/test_three_phase_mainline_config.py`
- Test: `tests/test_config_validation.py`

- [ ] **Step 1: Write the failing config tests**

Add assertions to `tests/test_three_phase_mainline_config.py`:

```python
    assert config.model.doctor_interest.enabled is True
    assert config.model.doctor_interest.hidden_channels == 8
    assert config.model.doctor_interest.temperature == 6.0
    assert config.model.topk_focus.enabled is True
    assert config.model.topk_focus.k == 3
    assert config.training.doctor_interest_loss.cam_align_weight == 0.05
    assert config.training.doctor_interest_loss.body_prior_weight == 0.02
    assert config.explainability.export_doctor_interest_maps is True
```

Add validation coverage to `tests/test_config_validation.py`:

```python
def test_three_phase_invalid_doctor_interest_temperature() -> None:
    config = ExperimentConfig(
        data=DataConfig(dataset_type="three_phase_ct_tabular", target_shape=[16, 64, 64], window_preset="liver"),
        model=ModelConfig(
            model_type="three_phase_ct_fusion",
            doctor_interest={"enabled": True, "temperature": 0.0},
        ),
    )
    findings = validate_config(config)
    assert any(item.path == "model.doctor_interest.temperature" for item in findings)


def test_three_phase_invalid_topk_focus_patch_size() -> None:
    config = ExperimentConfig(
        data=DataConfig(dataset_type="three_phase_ct_tabular", target_shape=[16, 64, 64], window_preset="liver"),
        model=ModelConfig(
            model_type="three_phase_ct_fusion",
            topk_focus={"enabled": True, "patch_size": [0, 4, 4]},
        ),
    )
    findings = validate_config(config)
    assert any(item.path == "model.topk_focus.patch_size" for item in findings)
```

- [ ] **Step 2: Run the focused config tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_config.py \
  tests/test_config_validation.py
```

Expected:

- `AttributeError` or failed assertions because `doctor_interest`, `topk_focus`, `doctor_interest_loss`, and `export_doctor_interest_maps` do not exist yet.

- [ ] **Step 3: Implement the config dataclasses and validation**

Add these dataclasses to `med_core/configs/base_config.py`:

```python
@dataclass
class DoctorInterestMapConfig(BaseConfig):
    enabled: bool = False
    hidden_channels: int = 8
    temperature: float = 6.0
    normalization: Literal["softmax", "sigmoid"] = "softmax"


@dataclass
class TopKFocusConfig(BaseConfig):
    enabled: bool = False
    k: int = 3
    patch_size: list[int] = field(default_factory=lambda: [4, 4, 4])
    projection_dim: int = 16


@dataclass
class DoctorInterestLossConfig(BaseConfig):
    cam_align_weight: float = 0.05
    consistency_weight: float = 0.02
    sparse_weight: float = 0.01
    diverse_weight: float = 0.01
    body_prior_weight: float = 0.02
```

Wire them into `ModelConfig`, `TrainingConfig`, and `ExplainabilityConfig`:

```python
class ModelConfig(BaseConfig):
    doctor_interest: DoctorInterestMapConfig = field(
        default_factory=DoctorInterestMapConfig
    )
    topk_focus: TopKFocusConfig = field(default_factory=TopKFocusConfig)


class TrainingConfig(BaseConfig):
    doctor_interest_loss: DoctorInterestLossConfig = field(
        default_factory=DoctorInterestLossConfig
    )


class ExplainabilityConfig(BaseConfig):
    export_phase_importance: bool = False
    export_case_explanations: bool = False
    heatmap_ready: bool = False
    build_results_split: Literal["train", "val", "test", "all"] = "test"
    min_global_importance_samples: int = 8
    export_doctor_interest_maps: bool = False
```

Normalize nested dicts in `__post_init__`:

```python
        if isinstance(self.doctor_interest, dict):
            self.doctor_interest = DoctorInterestMapConfig(**self.doctor_interest)
        if isinstance(self.topk_focus, dict):
            self.topk_focus = TopKFocusConfig(**self.topk_focus)
```

Add validation branches in `med_core/configs/validation.py`:

```python
if config.model.doctor_interest.enabled:
    if config.model.doctor_interest.temperature <= 0:
        findings.append(
            ValidationIssue(
                path="model.doctor_interest.temperature",
                message="doctor_interest.temperature must be positive",
                suggestion="Set model.doctor_interest.temperature to a value > 0",
            )
        )
    if config.model.doctor_interest.hidden_channels <= 0:
        findings.append(
            ValidationIssue(
                path="model.doctor_interest.hidden_channels",
                message="doctor_interest.hidden_channels must be positive",
                suggestion="Set model.doctor_interest.hidden_channels to a value > 0",
            )
        )

if config.model.topk_focus.enabled:
    if config.model.topk_focus.k <= 0:
        findings.append(
            ValidationIssue(
                path="model.topk_focus.k",
                message="topk_focus.k must be positive",
                suggestion="Set model.topk_focus.k to a value > 0",
            )
        )
    if len(config.model.topk_focus.patch_size) != 3 or any(
        int(v) <= 0 for v in config.model.topk_focus.patch_size
    ):
        findings.append(
            ValidationIssue(
                path="model.topk_focus.patch_size",
                message="topk_focus.patch_size must contain three positive integers",
                suggestion="Set model.topk_focus.patch_size like [4, 4, 4]",
            )
        )
```

Enable the path in `configs/demo/three_phase_ct_mvi_dr_z.yaml`:

```yaml
model:
  doctor_interest:
    enabled: true
    hidden_channels: 8
    temperature: 6.0
  topk_focus:
    enabled: true
    k: 3
    patch_size: [4, 4, 4]
    projection_dim: 16

training:
  doctor_interest_loss:
    cam_align_weight: 0.05
    consistency_weight: 0.02
    sparse_weight: 0.01
    diverse_weight: 0.01
    body_prior_weight: 0.02

explainability:
  export_doctor_interest_maps: true
```

- [ ] **Step 4: Run the focused config tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_config.py \
  tests/test_config_validation.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/configs/base_config.py \
  med_core/configs/validation.py \
  configs/demo/three_phase_ct_mvi_dr_z.yaml \
  tests/test_three_phase_mainline_config.py \
  tests/test_config_validation.py
git commit -m "feat: add doctor-interest config surface"
```

## Task 2: Add reusable doctor-interest modules and loss helpers

**Files:**
- Create: `med_core/models/doctor_interest.py`
- Test: `tests/test_doctor_interest_modules.py`

- [ ] **Step 1: Write the failing reusable-module tests**

Create `tests/test_doctor_interest_modules.py` with:

```python
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
    prob_map = torch.softmax(torch.randn(2, 1, 4, 8, 8).view(2, -1), dim=1).view(2, 1, 4, 8, 8)
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
```

- [ ] **Step 2: Run the new reusable-module tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_doctor_interest_modules.py
```

Expected:

- FAIL with `ModuleNotFoundError` for `med_core.models.doctor_interest`

- [ ] **Step 3: Implement the reusable modules**

Create `med_core/models/doctor_interest.py` with:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _spatial_softmax3d(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = logits.size(0)
    scaled = logits.view(batch_size, -1) * temperature
    weights = torch.softmax(scaled, dim=1)
    return weights.view_as(logits)


class DoctorInterestMapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 8, temperature: float = 6.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(feature_map)
        return {
            "logits": logits,
            "score_map": torch.sigmoid(logits),
            "prob_map": _spatial_softmax3d(logits, temperature=self.temperature),
        }


class InterestGuidedPooling3D(nn.Module):
    def forward(self, feature_map: torch.Tensor, prob_map: torch.Tensor) -> torch.Tensor:
        weighted = feature_map * prob_map
        return weighted.sum(dim=(2, 3, 4))


class TopKLocalFocus3D(nn.Module):
    def __init__(self, k: int, patch_size: tuple[int, int, int], projection_dim: int) -> None:
        super().__init__()
        self.k = k
        self.patch_size = patch_size
        self.projection = nn.LazyLinear(projection_dim)

    def forward(self, feature_map: torch.Tensor, score_map: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, channels, depth, height, width = feature_map.shape
        flat_scores = score_map.view(batch, -1)
        topk_scores, topk_indices = torch.topk(flat_scores, k=self.k, dim=1)
        z = topk_indices // (height * width)
        y = (topk_indices % (height * width)) // width
        x = topk_indices % width
        centers = torch.stack([z, y, x], dim=-1)
        patches = []
        for batch_index in range(batch):
            patch_features = []
            for center_index in range(self.k):
                cz, cy, cx = [int(v) for v in centers[batch_index, center_index]]
                z0 = max(cz - self.patch_size[0] // 2, 0)
                y0 = max(cy - self.patch_size[1] // 2, 0)
                x0 = max(cx - self.patch_size[2] // 2, 0)
                patch = feature_map[
                    batch_index : batch_index + 1,
                    :,
                    z0 : min(z0 + self.patch_size[0], depth),
                    y0 : min(y0 + self.patch_size[1], height),
                    x0 : min(x0 + self.patch_size[2], width),
                ]
                patch_features.append(F.adaptive_avg_pool3d(patch, (1, 1, 1)).flatten(1))
            patches.append(torch.cat(patch_features, dim=1))
        patch_tensor = torch.cat(patches, dim=0)
        return {
            "centers": centers,
            "scores": topk_scores,
            "feature": self.projection(patch_tensor),
        }


def build_body_mask_prior(volume: torch.Tensor, threshold_hu: float, border_ratio: float) -> torch.Tensor:
    body = (volume > threshold_hu).float()
    _, _, _, height, width = body.shape
    border_h = int(height * border_ratio)
    border_w = int(width * border_ratio)
    prior = body.clone()
    if border_h > 0:
        prior[:, :, :, :border_h, :] *= 0.5
        prior[:, :, :, height - border_h :, :] *= 0.5
    if border_w > 0:
        prior[:, :, :, :, :border_w] *= 0.5
        prior[:, :, :, :, width - border_w :] *= 0.5
    return prior


def compute_doctor_interest_losses(
    *,
    prob_map: torch.Tensor,
    teacher_map: torch.Tensor,
    augmented_prob_map: torch.Tensor | None,
    topk_centers: torch.Tensor,
    body_prior: torch.Tensor | None,
    cam_align_weight: float,
    consistency_weight: float,
    sparse_weight: float,
    diverse_weight: float,
    body_prior_weight: float,
) -> dict[str, object]:
    components: dict[str, torch.Tensor] = {}
    teacher_norm = teacher_map / (teacher_map.sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
    components["cam_align"] = F.mse_loss(prob_map, teacher_norm)
    components["consistency"] = (
        F.mse_loss(prob_map, augmented_prob_map) if augmented_prob_map is not None else torch.zeros((), device=prob_map.device)
    )
    components["sparse"] = -(prob_map * torch.log(prob_map + 1e-8)).sum(dim=(2, 3, 4)).mean()
    pairwise = topk_centers.unsqueeze(2) - topk_centers.unsqueeze(1)
    distances = pairwise.float().pow(2).sum(dim=-1).sqrt()
    components["diverse"] = torch.relu(2.0 - distances).mean()
    if body_prior is not None:
        components["body_prior"] = ((1.0 - body_prior) * prob_map).mean()
    else:
        components["body_prior"] = torch.zeros((), device=prob_map.device)
    total = (
        cam_align_weight * components["cam_align"]
        + consistency_weight * components["consistency"]
        + sparse_weight * components["sparse"]
        + diverse_weight * components["diverse"]
        + body_prior_weight * components["body_prior"]
    )
    return {"total": total, "components": components}
```

- [ ] **Step 4: Run the reusable-module tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_doctor_interest_modules.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/models/doctor_interest.py \
  tests/test_doctor_interest_modules.py
git commit -m "feat: add reusable doctor-interest modules"
```

## Task 3: Wire doctor-interest outputs into the three-phase fusion model

**Files:**
- Modify: `med_core/models/three_phase_ct_fusion.py`
- Test: `tests/test_three_phase_ct_fusion_model.py`

- [ ] **Step 1: Write the failing model tests**

Extend `tests/test_three_phase_ct_fusion_model.py` with:

```python
def test_three_phase_ct_fusion_model_returns_doctor_interest_outputs() -> None:
    model = ThreePhaseCTFusionModel(
        phase_feature_dim=32,
        clinical_input_dim=8,
        clinical_mask_dim=8,
        clinical_hidden_dim=16,
        fusion_hidden_dim=24,
        phase_fusion_type="gated",
        share_phase_encoder=False,
        use_risk_head=True,
        doctor_interest_enabled=True,
        doctor_interest_hidden_channels=8,
        doctor_interest_temperature=6.0,
        topk_focus_enabled=True,
        topk_focus_k=3,
        topk_focus_patch_size=(2, 2, 2),
        topk_focus_projection_dim=12,
    )

    outputs = model(
        arterial=torch.randn(2, 1, 8, 32, 32),
        portal=torch.randn(2, 1, 8, 32, 32),
        noncontrast=torch.randn(2, 1, 8, 32, 32),
        clinical=torch.randn(2, 8),
        clinical_missing_mask=torch.zeros(2, 8),
    )

    assert set(outputs["doctor_interest_maps"]) == {"arterial", "portal", "noncontrast"}
    assert outputs["doctor_interest_maps"]["arterial"]["score_map"].shape[1] == 1
    assert outputs["topk_focus_centers"]["arterial"].shape == (2, 3, 3)
    assert outputs["topk_focus_scores"]["arterial"].shape == (2, 3)
    assert outputs["phase_embeddings"]["arterial"].shape[0] == 2
```

- [ ] **Step 2: Run the focused model tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_ct_fusion_model.py
```

Expected:

- FAIL because `ThreePhaseCTFusionModel` does not accept doctor-interest constructor args and does not return the new payload.

- [ ] **Step 3: Implement the model wiring**

Update the imports and constructor in `med_core/models/three_phase_ct_fusion.py`:

```python
from med_core.models.doctor_interest import (
    DoctorInterestMapHead,
    InterestGuidedPooling3D,
    TopKLocalFocus3D,
)


class _PhaseEncoder3D(nn.Module):
    @property
    def output_channels(self) -> int:
        return self.projection.in_features
```

Add constructor args:

```python
        doctor_interest_enabled: bool = False,
        doctor_interest_hidden_channels: int = 8,
        doctor_interest_temperature: float = 6.0,
        topk_focus_enabled: bool = False,
        topk_focus_k: int = 3,
        topk_focus_patch_size: tuple[int, int, int] = (4, 4, 4),
        topk_focus_projection_dim: int = 16,
```

Instantiate heads after the phase encoders:

```python
        self.doctor_interest_enabled = doctor_interest_enabled
        self.topk_focus_enabled = topk_focus_enabled
        self.interest_pool = InterestGuidedPooling3D() if doctor_interest_enabled else None

        if doctor_interest_enabled:
            last_channels = self.arterial_encoder.output_channels
            self.arterial_interest_head = DoctorInterestMapHead(
                in_channels=last_channels,
                hidden_channels=doctor_interest_hidden_channels,
                temperature=doctor_interest_temperature,
            )
            self.portal_interest_head = DoctorInterestMapHead(
                in_channels=last_channels,
                hidden_channels=doctor_interest_hidden_channels,
                temperature=doctor_interest_temperature,
            )
            self.noncontrast_interest_head = DoctorInterestMapHead(
                in_channels=last_channels,
                hidden_channels=doctor_interest_hidden_channels,
                temperature=doctor_interest_temperature,
            )
            if topk_focus_enabled:
                self.arterial_focus = TopKLocalFocus3D(
                    k=topk_focus_k,
                    patch_size=topk_focus_patch_size,
                    projection_dim=topk_focus_projection_dim,
                )
                self.portal_focus = TopKLocalFocus3D(
                    k=topk_focus_k,
                    patch_size=topk_focus_patch_size,
                    projection_dim=topk_focus_projection_dim,
                )
                self.noncontrast_focus = TopKLocalFocus3D(
                    k=topk_focus_k,
                    patch_size=topk_focus_patch_size,
                    projection_dim=topk_focus_projection_dim,
                )
```

In `forward`, compute the new phase embeddings:

```python
        doctor_interest_maps: dict[str, dict[str, torch.Tensor]] = {}
        topk_focus_centers: dict[str, torch.Tensor] = {}
        topk_focus_scores: dict[str, torch.Tensor] = {}
        phase_embeddings: dict[str, torch.Tensor] = {}

        def _augment_phase(
            phase_name: str,
            feature_vector: torch.Tensor,
            feature_map: torch.Tensor,
            interest_head: nn.Module | None,
            focus_module: nn.Module | None,
        ) -> torch.Tensor:
            if not self.doctor_interest_enabled or interest_head is None or self.interest_pool is None:
                phase_embeddings[phase_name] = feature_vector
                return feature_vector

            interest_outputs = interest_head(feature_map)
            doctor_interest_maps[phase_name] = interest_outputs
            interest_feature = self.interest_pool(feature_map, interest_outputs["prob_map"])
            feature_parts = [feature_vector, interest_feature]

            if self.topk_focus_enabled and focus_module is not None:
                focus_outputs = focus_module(feature_map, interest_outputs["score_map"])
                topk_focus_centers[phase_name] = focus_outputs["centers"]
                topk_focus_scores[phase_name] = focus_outputs["scores"]
                feature_parts.append(focus_outputs["feature"])
            else:
                topk_focus_centers[phase_name] = torch.zeros(feature_vector.size(0), 0, 3, device=feature_vector.device, dtype=torch.long)
                topk_focus_scores[phase_name] = torch.zeros(feature_vector.size(0), 0, device=feature_vector.device)

            enriched = torch.cat(feature_parts, dim=1)
            projected = nn.functional.layer_norm(enriched, enriched.shape[1:])
            phase_embeddings[phase_name] = projected
            return projected
```

Use the enriched phase vectors in the fusion stack and return:

```python
        return {
            "logits": logits,
            "probability": probability,
            "risk_score": risk_score,
            "fused_features": fused_features,
            "phase_features": phase_features,
            "clinical_features": clinical_features,
            "phase_importance": phase_importance,
            "feature_maps": {
                "arterial": arterial_map,
                "portal": portal_map,
                "noncontrast": noncontrast_map,
            },
            "doctor_interest_maps": doctor_interest_maps,
            "topk_focus_centers": topk_focus_centers,
            "topk_focus_scores": topk_focus_scores,
            "phase_embeddings": phase_embeddings,
        }
```

When `doctor_interest_enabled` is `False`, return empty dicts for the new keys to preserve a stable contract:

```python
            "doctor_interest_maps": doctor_interest_maps if doctor_interest_maps else {},
            "topk_focus_centers": topk_focus_centers if topk_focus_centers else {},
            "topk_focus_scores": topk_focus_scores if topk_focus_scores else {},
            "phase_embeddings": phase_embeddings if phase_embeddings else {},
```

- [ ] **Step 4: Run the focused model tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_ct_fusion_model.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/models/three_phase_ct_fusion.py \
  tests/test_three_phase_ct_fusion_model.py
git commit -m "feat: add doctor-interest outputs to three-phase fusion"
```

## Task 4: Add doctor-interest regularization to the native training path

**Files:**
- Modify: `med_core/cli/train.py`
- Modify: `tests/test_three_phase_mainline_train.py`

- [ ] **Step 1: Write the failing training-path tests**

Extend `tests/test_three_phase_mainline_train.py`:

```python
    history = json.loads((output_dir / "logs" / "history.json").read_text())
    assert "train_doctor_interest_loss" in history["entries"][0]
    assert "val_doctor_interest_loss" in history["entries"][0]
```

Enable the path in the smoke config:

```python
                "model": {
                    "model_type": "three_phase_ct_fusion",
                    "num_classes": 2,
                    "phase_feature_dim": 16,
                    "doctor_interest": {"enabled": True, "hidden_channels": 8, "temperature": 6.0},
                    "topk_focus": {"enabled": True, "k": 3, "patch_size": [2, 2, 2], "projection_dim": 8},
                    "share_phase_encoder": False,
                    "use_risk_head": True,
                    "tabular": {"hidden_dims": [16], "output_dim": 8, "dropout": 0.1},
                    "fusion": {"fusion_type": "gated", "hidden_dim": 12, "dropout": 0.1},
                },
                "training": {
                    "num_epochs": 1,
                    "doctor_interest_loss": {
                        "cam_align_weight": 0.05,
                        "consistency_weight": 0.02,
                        "sparse_weight": 0.01,
                        "diverse_weight": 0.01,
                        "body_prior_weight": 0.02,
                    },
                    "mixed_precision": False,
                    "use_progressive_training": False,
                    "optimizer": {
                        "optimizer": "adam",
                        "learning_rate": 0.0005,
                        "weight_decay": 0.0,
                    },
                    "scheduler": {"scheduler": "none"},
                },
```

- [ ] **Step 2: Run the focused training smoke test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_mainline_train.py
```

Expected:

- FAIL because `history.json` entries do not include doctor-interest loss fields.

- [ ] **Step 3: Implement doctor-interest loss computation in `train.py`**

Import the helper:

```python
from med_core.models.doctor_interest import (
    build_body_mask_prior,
    compute_doctor_interest_losses,
)
```

Add a helper in `med_core/cli/train.py`:

```python
def _compute_three_phase_doctor_interest_loss(outputs, batch, config, device):
    if not config.model.doctor_interest.enabled:
        zero = torch.zeros((), device=device)
        return zero, {}

    total_loss = torch.zeros((), device=device)
    component_sums: dict[str, torch.Tensor] = {}
    phase_inputs = {
        "arterial": batch["arterial"].to(device),
        "portal": batch["portal"].to(device),
        "noncontrast": batch["noncontrast"].to(device),
    }
    for phase_name, input_volume in phase_inputs.items():
        interest_outputs = outputs["doctor_interest_maps"][phase_name]
        teacher_map = interest_outputs["score_map"].detach()
        body_prior = build_body_mask_prior(
            input_volume,
            threshold_hu=-200.0,
            border_ratio=0.1,
        )
        loss_payload = compute_doctor_interest_losses(
            prob_map=interest_outputs["prob_map"],
            teacher_map=teacher_map,
            augmented_prob_map=interest_outputs["prob_map"].detach(),
            topk_centers=outputs["topk_focus_centers"][phase_name],
            body_prior=body_prior,
            cam_align_weight=config.training.doctor_interest_loss.cam_align_weight,
            consistency_weight=config.training.doctor_interest_loss.consistency_weight,
            sparse_weight=config.training.doctor_interest_loss.sparse_weight,
            diverse_weight=config.training.doctor_interest_loss.diverse_weight,
            body_prior_weight=config.training.doctor_interest_loss.body_prior_weight,
        )
        total_loss = total_loss + loss_payload["total"]
        for name, value in loss_payload["components"].items():
            component_sums[name] = component_sums.get(name, torch.zeros_like(value)) + value
    return total_loss / 3.0, {name: value / 3.0 for name, value in component_sums.items()}
```

Use it in the training loop:

```python
            cls_loss = criterion(outputs["logits"], batch["label"].to(device))
            doctor_interest_loss, _doctor_components = _compute_three_phase_doctor_interest_loss(
                outputs, batch, config, device
            )
            loss = cls_loss + doctor_interest_loss
```

Track both train and validation doctor-interest loss:

```python
        train_doctor_interest_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                arterial=batch["arterial"].to(device),
                portal=batch["portal"].to(device),
                noncontrast=batch["noncontrast"].to(device),
                clinical=batch["clinical"].to(device),
                clinical_missing_mask=(
                    batch["clinical_missing_mask"].to(device)
                    if clinical_mask_dim > 0
                    else None
                ),
            )
            cls_loss = criterion(outputs["logits"], batch["label"].to(device))
            doctor_interest_loss, _ = _compute_three_phase_doctor_interest_loss(
                outputs, batch, config, device
            )
            loss = cls_loss + doctor_interest_loss
            loss.backward()
            optimizer.step()
            train_doctor_interest_sum += float(doctor_interest_loss.item())
        val_doctor_interest_sum = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                outputs = model(
                    arterial=batch["arterial"].to(device),
                    portal=batch["portal"].to(device),
                    noncontrast=batch["noncontrast"].to(device),
                    clinical=batch["clinical"].to(device),
                    clinical_missing_mask=(
                        batch["clinical_missing_mask"].to(device)
                        if clinical_mask_dim > 0
                        else None
                    ),
                )
                doctor_interest_loss, _ = _compute_three_phase_doctor_interest_loss(
                    outputs, batch, config, device
                )
                val_doctor_interest_sum += float(doctor_interest_loss.item())
        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_doctor_interest_loss": train_doctor_interest_sum / max(len(train_loader), 1),
            "val_doctor_interest_loss": val_doctor_interest_sum / max(len(val_loader), 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
```

Pass the new constructor args into the `ThreePhaseCTFusionModel` constructor:

```python
        doctor_interest_enabled=config.model.doctor_interest.enabled,
        doctor_interest_hidden_channels=config.model.doctor_interest.hidden_channels,
        doctor_interest_temperature=config.model.doctor_interest.temperature,
        topk_focus_enabled=config.model.topk_focus.enabled,
        topk_focus_k=config.model.topk_focus.k,
        topk_focus_patch_size=tuple(config.model.topk_focus.patch_size),
        topk_focus_projection_dim=config.model.topk_focus.projection_dim,
```

- [ ] **Step 4: Run the training smoke test to verify it passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_mainline_train.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/cli/train.py \
  tests/test_three_phase_mainline_train.py
git commit -m "feat: train three-phase model with doctor-interest regularization"
```

## Task 5: Export doctor-interest artifacts in build-results

**Files:**
- Modify: `med_core/postprocessing/results.py`
- Modify: `tests/test_three_phase_mainline_build_results.py`

- [ ] **Step 1: Write the failing artifact-export tests**

Extend `tests/test_three_phase_mainline_build_results.py`:

```python
    summary = json.loads((output_dir / "reports" / "summary.json").read_text())
    assert summary["artifacts"]["doctor_interest_manifest_path"].endswith(
        "visualizations/doctor_interest/manifest.json"
    )

    manifest = json.loads(
        (output_dir / "artifacts" / "visualizations" / "doctor_interest" / "manifest.json").read_text()
    )
    assert manifest["cases"]
    first_heatmap = manifest["cases"][0]["doctor_interest"][0]
    assert any(item["space"] == "original_image" for item in first_heatmap["renderings"])
    assert "focus_regions" in first_heatmap
```

Check the report wording:

```python
    report_text = (output_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "- 医生建议关注区清单:" in report_text
    assert "- 关注区来源: 模型内生 doctor interest map" in report_text
```

- [ ] **Step 2: Run the focused build-results test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_build_results.py::test_three_phase_build_results_emits_heatmap_artifacts_when_enabled
```

Expected:

- FAIL because there is no doctor-interest manifest or report wording yet.

- [ ] **Step 3: Implement artifact export in `results.py`**

Load the new constructor args when rebuilding the model:

```python
    model = ThreePhaseCTFusionModel(
        phase_feature_dim=config.model.phase_feature_dim,
        clinical_input_dim=len(config.data.clinical_feature_columns),
        clinical_mask_dim=clinical_mask_dim,
        clinical_hidden_dim=config.model.tabular.output_dim,
        fusion_hidden_dim=config.model.fusion.hidden_dim,
        num_classes=config.model.num_classes,
        phase_fusion_type=config.model.phase_fusion.mode,
        share_phase_encoder=config.model.share_phase_encoder,
        freeze_phase_encoder=False,
        use_risk_head=config.model.use_risk_head,
        phase_encoder_base_channels=config.model.phase_encoder.base_channels,
        phase_encoder_num_blocks=config.model.phase_encoder.num_blocks,
        phase_encoder_dropout=config.model.phase_encoder.dropout,
        phase_encoder_norm_type=config.model.phase_encoder.norm,
        doctor_interest_enabled=config.model.doctor_interest.enabled,
        doctor_interest_hidden_channels=config.model.doctor_interest.hidden_channels,
        doctor_interest_temperature=config.model.doctor_interest.temperature,
        topk_focus_enabled=config.model.topk_focus.enabled,
        topk_focus_k=config.model.topk_focus.k,
        topk_focus_patch_size=tuple(config.model.topk_focus.patch_size),
        topk_focus_projection_dim=config.model.topk_focus.projection_dim,
    )
```

Add a new exporter:

```python
def _generate_three_phase_doctor_interest_artifacts(
    *,
    model: ThreePhaseCTFusionModel,
    dataset: ThreePhaseCTCaseDataset,
    device: torch.device,
    layout: RunOutputLayout,
    cases: list[dict[str, Any]],
    enabled: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if not enabled or not cases:
        return {}, {}

    artifact_dir = layout.visualizations_dir / "doctor_interest"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _reset_directory_children(artifact_dir)
    phase_names = ("arterial", "portal", "noncontrast")
    manifest_cases = []
    per_case_payload = {}

    for index, case in enumerate(cases):
        sample = dataset[index]
        outputs = model(
            arterial=sample["arterial"].unsqueeze(0).to(device),
            portal=sample["portal"].unsqueeze(0).to(device),
            noncontrast=sample["noncontrast"].unsqueeze(0).to(device),
            clinical=sample["clinical"].unsqueeze(0).to(device),
            clinical_missing_mask=sample["clinical_missing_mask"].unsqueeze(0).to(device),
        )
        case_dir = artifact_dir / str(case["case_id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        phase_payloads = []
        for phase_name in phase_names:
            input_volume = sample[phase_name].squeeze(0).numpy()
            render_context = dataset.get_phase_render_context(index=index, phase_name=phase_name)
            original_volume = np.asarray(render_context["original_volume"], dtype=np.float32)
            score_map = outputs["doctor_interest_maps"][phase_name]["score_map"]
            score_volume = torch.nn.functional.interpolate(
                score_map,
                size=input_volume.shape,
                mode="trilinear",
                align_corners=False,
            ).squeeze().detach().cpu().numpy()
            slice_index = select_representative_slice(score_volume)
            original_slice_index = map_slice_index_between_depths(
                source_index=slice_index,
                source_depth=input_volume.shape[0],
                target_depth=original_volume.shape[0],
            )
            model_overlay_path = case_dir / f"{phase_name}_doctor_interest_overlay.png"
            original_overlay_path = (
                case_dir / f"{phase_name}_doctor_interest_original_overlay.png"
            )
            original_slice_path = (
                case_dir / f"{phase_name}_doctor_interest_original_slice.png"
            )
            model_rendering = render_overlay_artifact(
                image_slice=input_volume[slice_index],
                attention_slice=score_volume[slice_index],
                save_path=model_overlay_path,
                space="model_input",
                kind="overlay",
                slice_index=int(slice_index),
                title=f"Case {case['case_id']} | {phase_name} | doctor_interest",
            )
            original_rendering = render_overlay_artifact(
                image_slice=original_volume[original_slice_index],
                attention_slice=score_volume[slice_index],
                save_path=original_overlay_path,
                space="original_image",
                kind="overlay",
                slice_index=int(original_slice_index),
                title=f"Case {case['case_id']} | {phase_name} | doctor_interest_original",
            )
            original_rgb = (
                prepare_overlay_image(original_volume[original_slice_index]) * 255.0
            ).round().astype(np.uint8)
            Image.fromarray(original_rgb).save(original_slice_path)
            original_slice_rendering = build_rendering_metadata(
                space="original_image",
                kind="base_slice",
                image_path=original_slice_path,
                slice_index=int(original_slice_index),
                image_shape=original_volume[original_slice_index].shape,
            )
            renderings = [model_rendering, original_rendering, original_slice_rendering]
            phase_payloads.append(
                {
                    "phase": phase_name,
                    "renderings": renderings,
                    "focus_regions": [],
                    "render_space": "model_input",
                    "image_path": str(model_overlay_path),
                    "slice_index": int(slice_index),
                }
            )
```

Populate `focus_regions` from the stored centers and scores:

```python
            focus_regions = [
                {
                    "rank": rank + 1,
                    "feature_map_center": [int(v) for v in outputs["topk_focus_centers"][phase_name][0, rank].tolist()],
                    "score": round(float(outputs["topk_focus_scores"][phase_name][0, rank]), 6),
                }
                for rank in range(outputs["topk_focus_scores"][phase_name].shape[1])
            ]
```

Add the new artifact map to `summary_payload["artifacts"]`:

```python
            "doctor_interest_manifest_path": str(doctor_interest_manifest_path),
```

Extend `case_explanations_payload`:

```python
                "doctor_interest_artifacts": doctor_interest_cases.get(case["case_id"], []),
```

Update report wording:

```python
    if doctor_interest_artifact_paths:
        report_lines.extend(["", "## Doctor Interest Maps", ""])
        report_lines.append(
            "- 医生建议关注区清单: "
            f"{doctor_interest_artifact_paths['doctor_interest_manifest_path']}"
        )
        report_lines.append("- 关注区来源: 模型内生 doctor interest map")
        report_lines.append("- 展示方式: 原始切片叠加图 + 低分辨率兴趣图回映")
```

- [ ] **Step 4: Run the focused build-results test to verify it passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_build_results.py::test_three_phase_build_results_emits_heatmap_artifacts_when_enabled
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/postprocessing/results.py \
  tests/test_three_phase_mainline_build_results.py
git commit -m "feat: export doctor-interest artifacts for three-phase results"
```

## Task 6: Run end-to-end regression and refresh the user-facing docs

**Files:**
- Modify: `tests/test_three_phase_mainline_e2e.py`
- Modify: `README.md`
- Modify: `docs/contents/playbooks/external-demo-path.md`

- [ ] **Step 1: Add end-to-end assertions for the new artifacts**

Extend `tests/test_three_phase_mainline_e2e.py`:

```python
    summary = json.loads((output_dir / "reports" / "summary.json").read_text())
    assert "doctor_interest_manifest_path" in summary["artifacts"]
    case_explanations = json.loads((output_dir / "metrics" / "case_explanations.json").read_text())
    assert "doctor_interest_artifacts" in case_explanations["cases"][0]
```

- [ ] **Step 2: Run the full three-phase regression suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_doctor_interest_modules.py \
  tests/test_three_phase_ct_fusion_model.py \
  tests/test_three_phase_mainline_config.py \
  tests/test_three_phase_mainline_train.py \
  tests/test_three_phase_mainline_build_results.py \
  tests/test_three_phase_mainline_e2e.py
```

Expected:

- PASS

- [ ] **Step 3: Update the docs once tests are stable**

Add a short section to `README.md`:

```markdown
### Three-Phase CT Doctor Interest Maps

The native three-phase CT + clinical MVI path can now export:

- per-phase `phase_importance`
- case-level `doctor_interest_map` overlays
- optional Top-K focus-region candidates for clinician review

These artifacts are generated through the existing `build-results` workflow and are intended as clinician-facing review aids, not lesion contours.
```

Update `docs/contents/playbooks/external-demo-path.md`:

```markdown
- `artifacts/visualizations/doctor_interest/manifest.json`
- `metrics/case_explanations.json`

The doctor-interest overlays show model-internal suggested review regions. They are not segmentation masks and should be presented as “建议关注区”.
```

- [ ] **Step 4: Run the docs-adjacent contract tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_build_results.py \
  tests/test_web_report_generator.py
```

Expected:

- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  tests/test_three_phase_mainline_e2e.py \
  README.md \
  docs/contents/playbooks/external-demo-path.md
git commit -m "docs: describe doctor-interest artifacts in three-phase workflow"
```

## Verification Checklist

Before calling the work complete:

- [ ] `tests/test_doctor_interest_modules.py` passes
- [ ] `tests/test_three_phase_ct_fusion_model.py` passes
- [ ] `tests/test_three_phase_mainline_config.py` passes
- [ ] `tests/test_config_validation.py` passes
- [ ] `tests/test_three_phase_mainline_train.py` passes
- [ ] `tests/test_three_phase_mainline_build_results.py` passes
- [ ] `tests/test_three_phase_mainline_e2e.py` passes
- [ ] `tests/test_build_results.py` passes
- [ ] `tests/test_web_report_generator.py` passes
- [ ] `configs/demo/three_phase_ct_mvi_dr_z.yaml` still loads through `load_config`
- [ ] build-results summary includes both `phase_importance_path` and `doctor_interest_manifest_path`
- [ ] clinician-facing wording says `建议关注区`, not `病灶轮廓` or `segmentation`

## Non-Goals

This plan does not add:

- manual ROI annotation support
- voxel-accurate lesion segmentation
- true gaze or eye-tracking supervision
- prototype-network or concept-bottleneck refactors
- new CLI entrypoints outside the current `train/build-results` path
