# SMURF Feature Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the three-phase CT + clinical demo into a clinician-facing, explanation-ready mainline model without changing the `validate-config/train/build-results --config ...` entrypoints.

**Architecture:** Keep the current mainline YAML and CLI path, but replace the shallow three-phase feature extractor with a phase-aware encoder/fusion stack, add clinical preprocessing with normalization and missing-mask support, and extend build-results with phase-importance and case-explanation artifacts. The work is split so each task lands a working slice with tests before the next layer is added.

**Tech Stack:** Python 3.13, PyTorch, pytest, YAML config dataclasses, MedFusion mainline CLI/output layout.

---

## File Map

### Core files to modify

- `med_core/models/three_phase_ct_fusion.py`
  - Replace the shallow `_PhaseEncoder` with a more structured 3D encoder
  - Add phase-gated fusion and richer forward outputs
- `med_core/datasets/three_phase_ct.py`
  - Add clinical preprocessing hooks and missing-mask support
- `med_core/cli/train.py`
  - Fit/save clinical preprocessing state during training
  - Pass new model/data outputs through the native three-phase training path
- `med_core/postprocessing/results.py`
  - Load clinical preprocessing state
  - Emit `phase_importance` and `case_explanations` artifacts
  - Extend report/summary payloads
- `med_core/configs/base_config.py`
  - Add config dataclasses/fields for phase encoder, phase fusion, clinical preprocessing, explainability
- `med_core/configs/validation.py`
  - Validate the new config fields

### New files to create

- `med_core/shared/preprocessing/clinical.py`
  - Shared clinical normalizer / missing-mask utilities
- `tests/test_three_phase_clinical_preprocessing.py`
  - Unit tests for normalization and missing-mask behavior

### Tests to modify

- `tests/test_three_phase_ct_fusion_model.py`
- `tests/test_three_phase_ct_dataset.py`
- `tests/test_three_phase_mainline_train.py`
- `tests/test_three_phase_mainline_build_results.py`
- `tests/test_three_phase_mainline_config.py`
- `tests/test_config_validation.py`

## Task 1: Add config surface for phase encoder, clinical preprocessing, and explainability

**Files:**
- Modify: `med_core/configs/base_config.py`
- Modify: `med_core/configs/validation.py`
- Test: `tests/test_three_phase_mainline_config.py`
- Test: `tests/test_config_validation.py`

- [ ] **Step 1: Write the failing config tests**

Add assertions in `tests/test_three_phase_mainline_config.py` for:
- `config.model.phase_encoder.base_channels`
- `config.model.phase_fusion.mode`
- `config.data.clinical_preprocessing.normalize`
- `config.explainability.export_phase_importance`

Add validation tests in `tests/test_config_validation.py` covering:
- invalid `phase_fusion.mode`
- invalid clinical preprocessing strategy
- explainability enabled with unsupported options

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_config.py \
  tests/test_config_validation.py
```

Expected:
- failures for missing config fields / missing validation branches

- [ ] **Step 3: Implement minimal config dataclasses and validation**

Add dataclasses / fields in `med_core/configs/base_config.py`:

- `PhaseEncoderConfig`
- `PhaseFusionConfig`
- `ClinicalPreprocessingConfig`
- `ExplainabilityConfig`

Wire them into the main experiment config and add defaults that preserve current behavior.

Add validation rules in `med_core/configs/validation.py`:

- valid `phase_fusion.mode` in `{"concatenate", "mean", "gated"}`
- positive channel/hidden dimensions
- valid clinical preprocessing strategy in `{"none", "zero_with_mask"}`

- [ ] **Step 4: Run the focused tests to verify they pass**

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
  tests/test_three_phase_mainline_config.py \
  tests/test_config_validation.py
git commit -m "feat: add three-phase feature-network config surface"
```

## Task 2: Add shared clinical preprocessing utilities

**Files:**
- Create: `med_core/shared/preprocessing/clinical.py`
- Test: `tests/test_three_phase_clinical_preprocessing.py`

- [ ] **Step 1: Write the failing preprocessing tests**

Create tests for:
- fitting mean/std on observed values
- transforming features with zero-fill + missing mask
- round-tripping fitted stats through dict payloads

Include a concrete test like:

```python
def test_zero_with_mask_outputs_scaled_values_and_missing_mask():
    rows = [[10.0, None], [14.0, 2.0]]
    preprocessor = ClinicalFeaturePreprocessor(strategy="zero_with_mask")
    payload = preprocessor.fit_transform(rows)
    assert payload.values.shape == (2, 2)
    assert payload.missing_mask.shape == (2, 2)
```

- [ ] **Step 2: Run the new test file to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_clinical_preprocessing.py
```

Expected:
- FAIL because module/class does not exist yet

- [ ] **Step 3: Implement the shared clinical preprocessor**

Create `med_core/shared/preprocessing/clinical.py` with:
- `ClinicalFeatureTransform`
- `ClinicalFeaturePreprocessor`
- `fit`, `transform`, `fit_transform`, `to_dict`, `from_dict`

Behavior:
- compute per-column mean/std from non-missing values
- output normalized values
- zero-fill missing positions after normalization
- emit missing mask

- [ ] **Step 4: Run the new test file to verify it passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_clinical_preprocessing.py
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/shared/preprocessing/clinical.py \
  tests/test_three_phase_clinical_preprocessing.py
git commit -m "feat: add shared clinical preprocessing for three-phase demos"
```

## Task 3: Upgrade dataset outputs to include normalized clinical values and missing masks

**Files:**
- Modify: `med_core/datasets/three_phase_ct.py`
- Test: `tests/test_three_phase_ct_dataset.py`

- [ ] **Step 1: Write the failing dataset tests**

Extend `tests/test_three_phase_ct_dataset.py` to assert:
- dataset sample includes `clinical_missing_mask`
- dataset can consume a fitted clinical preprocessor
- missing values are represented in mask output

- [ ] **Step 2: Run the focused dataset tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_ct_dataset.py
```

Expected:
- FAIL because samples do not yet include the new fields

- [ ] **Step 3: Implement dataset preprocessing hooks**

Modify `med_core/datasets/three_phase_ct.py` to:
- accept optional clinical preprocessing state/object
- preserve raw clinical rows internally
- transform clinical values at sample build time or construction time
- return both:
  - `clinical`
  - `clinical_missing_mask`

Keep current path working when preprocessing is disabled.

- [ ] **Step 4: Run the focused dataset tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_ct_dataset.py
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/datasets/three_phase_ct.py \
  tests/test_three_phase_ct_dataset.py
git commit -m "feat: expose clinical masks in three-phase dataset"
```

## Task 4: Replace the shallow phase encoder with an explanation-ready encoder/fusion stack

**Files:**
- Modify: `med_core/models/three_phase_ct_fusion.py`
- Test: `tests/test_three_phase_ct_fusion_model.py`

- [ ] **Step 1: Write the failing model tests**

Extend `tests/test_three_phase_ct_fusion_model.py` to cover:
- `phase_importance` exists and sums to 1 for gated fusion
- `clinical_features` exists in forward output
- per-phase `feature_maps` exist for heatmap readiness

Example assertion:

```python
assert outputs["phase_importance"].shape == (2, 3)
assert torch.allclose(outputs["phase_importance"].sum(dim=1), torch.ones(2))
assert set(outputs["feature_maps"]) == {"arterial", "portal", "noncontrast"}
```

- [ ] **Step 2: Run the focused model tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_ct_fusion_model.py
```

Expected:
- FAIL because outputs and fusion mode do not exist yet

- [ ] **Step 3: Implement the new model structure**

Refactor `med_core/models/three_phase_ct_fusion.py` to include:
- a deeper `_PhaseEncoder3D`
- explicit `feature_map` return per phase
- `gated` phase fusion mode
- richer forward payload:
  - `phase_features`
  - `clinical_features`
  - `fused_features`
  - `phase_importance`
  - `feature_maps`

Keep `concatenate` and `mean` modes functioning for compatibility.

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
git commit -m "feat: add explanation-ready three-phase feature encoder"
```

## Task 5: Fit/save clinical preprocessing state in native three-phase training

**Files:**
- Modify: `med_core/cli/train.py`
- Modify: `tests/test_three_phase_mainline_train.py`

- [ ] **Step 1: Write the failing training-path tests**

Extend `tests/test_three_phase_mainline_train.py` to assert training emits a reusable preprocessing artifact, for example:
- `artifacts/clinical_preprocessing.json`

- [ ] **Step 2: Run the focused training test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_mainline_train.py
```

Expected:
- FAIL because preprocessing state is not saved

- [ ] **Step 3: Implement preprocessing fit/save in training**

Modify `med_core/cli/train.py` to:
- fit `ClinicalFeaturePreprocessor` on training rows
- construct train/val datasets with shared preprocessing state
- write the fitted payload into run artifacts

Keep output path under the mainline run layout, not a demo-private folder.

- [ ] **Step 4: Run the focused training test to verify it passes**

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
git commit -m "feat: persist clinical preprocessing in three-phase training"
```

## Task 6: Extend build-results with phase importance and case explanations

**Files:**
- Modify: `med_core/postprocessing/results.py`
- Modify: `tests/test_three_phase_mainline_build_results.py`

- [ ] **Step 1: Write the failing build-results tests**

Extend `tests/test_three_phase_mainline_build_results.py` to assert:
- `metrics/case_explanations.json` exists
- summary/report artifacts include phase-importance paths
- report text includes clinician-friendly phase contribution wording

Example assertions:

```python
assert (output_dir / "metrics" / "case_explanations.json").exists()
assert "- 三期贡献概览:" in report_text
assert "phase_importance_path" in summary["artifacts"]
```

- [ ] **Step 2: Run the focused build-results tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_mainline_build_results.py
```

Expected:
- FAIL because artifacts/report content do not exist yet

- [ ] **Step 3: Implement result artifact expansion**

Modify `med_core/postprocessing/results.py` to:
- load clinical preprocessing state
- collect `phase_importance` from model outputs
- write `phase_importance.json`
- write `case_explanations.json`
- extend `summary.json` artifact map
- extend `report.md` with:
  - phase contribution summary
  - key clinical factor summary

- [ ] **Step 4: Run the focused build-results tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_mainline_build_results.py
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/postprocessing/results.py \
  tests/test_three_phase_mainline_build_results.py
git commit -m "feat: add phase importance artifacts to three-phase reports"
```

## Task 7: Run end-to-end regression on the upgraded three-phase mainline path

**Files:**
- Test: `tests/test_three_phase_mainline_e2e.py`

- [ ] **Step 1: Add or adjust assertions only if the end-to-end contract changed**

If needed, extend `tests/test_three_phase_mainline_e2e.py` to check the new essential artifacts without over-coupling to all internals.

- [ ] **Step 2: Run the end-to-end three-phase suite**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_clinical_preprocessing.py \
  tests/test_three_phase_ct_dataset.py \
  tests/test_three_phase_ct_fusion_model.py \
  tests/test_three_phase_mainline_config.py \
  tests/test_three_phase_mainline_train.py \
  tests/test_three_phase_mainline_build_results.py \
  tests/test_three_phase_mainline_e2e.py
```

Expected:
- PASS

- [ ] **Step 3: Run the broader contracts that guard results wording**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_build_results.py \
  tests/test_web_report_generator.py
```

Expected:
- PASS

- [ ] **Step 4: Review `git diff` for contract stability**

Check specifically:
- no new demo-private module names
- no CLI entrypoint changes
- artifact keys remain backward compatible where practical

- [ ] **Step 5: Commit**

```bash
git add \
  tests/test_three_phase_mainline_e2e.py
git commit -m "test: regress upgraded three-phase mainline path"
```

## Task 8: Document the new explanation-ready behavior for users

**Files:**
- Modify: `README.md`
- Modify: `docs/contents/getting-started/public-datasets.md` only if shared wording needs extension
- Modify: `docs/contents/playbooks/external-demo-path.md` if artifact list changed

- [ ] **Step 1: Write doc updates after code is stable**

Document:
- new phase contribution artifact
- case explanation artifact
- expectation that heatmap support is “ready” but not yet fully rendered

- [ ] **Step 2: Run doc-adjacent tests if touched**

Run the relevant tests already guarding result wording and examples if doc strings overlap.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/contents/playbooks/external-demo-path.md
git commit -m "docs: describe explanation-ready three-phase demo outputs"
```

## Recommended execution notes

- Keep each task self-contained and passing before moving on.
- Prefer preserving existing artifact keys and adding new keys instead of renaming aggressively.
- Avoid coupling the model to disease-specific language; clinician-friendly wording belongs in reports, not model internals.
- Do not start the heatmap renderer itself in this plan; only land the interfaces and artifacts needed for the later step.
