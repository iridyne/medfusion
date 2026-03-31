# Original-Slice Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable OSS-level original-image overlay capability so heatmaps can be rendered both in model space and on the source slice, while keeping the current `validate-config/train/build-results --config ...` workflow unchanged.

**Architecture:** Keep Grad-CAM generation in model space, then add a shared visualization layer that can map a selected heatmap slice into another image space and render an overlay with explicit provenance metadata. The three-phase CT results pipeline becomes the first consumer: it will export both the current model-space overlay and a native-resolution original-slice overlay, with manifest fields that make the mapping strategy visible instead of pretending it is voxel-perfect registration.

**Tech Stack:** Python 3.13, NumPy, PyTorch, Matplotlib, pytest, MedFusion run artifact layout.

---

## File Map

### Core files to modify

- `med_core/shared/visualization/heatmaps.py`
  - Extend the current heatmap helper from "CAM volume + representative slice" into a reusable slice-mapping and overlay-rendering utility.
  - Keep the API modality-agnostic: it should accept generic image slices / heatmap slices / metadata, not DICOM-specific names.
- `med_core/datasets/three_phase_ct.py`
  - Expose a safe way for postprocessing to obtain both resampled model-input volume metadata and native source volume slices for a case/phase without affecting the training sample contract.
- `med_core/postprocessing/results.py`
  - Generate two heatmap renderings per phase:
    - model-input overlay
    - original-slice overlay
  - Extend manifest payloads, case explanations, summary, and report text with rendering provenance.

### New files to create

- `tests/test_heatmap_visualization.py`
  - Unit tests for slice-index mapping, 2D heatmap resizing, overlay metadata, and artifact rendering helpers.

### Tests to modify

- `tests/test_three_phase_mainline_build_results.py`
  - Add integration coverage for original-slice overlay artifact export and manifest structure.
- `tests/test_three_phase_ct_dataset.py`
  - Add focused tests for retrieving native-phase rendering context without regressing training behavior.

## Design Constraints

- Shared code in `med_core/shared/visualization/heatmaps.py` must stay generic:
  - no `smurf`
  - no `mvi`
  - no hard-coded DICOM assumptions
- The helper must be honest about coordinate fidelity:
  - it is a mapped overlay from model space back to source space
  - it is not a segmentation mask
  - it is not guaranteed voxel-level registration
- The first implementation may use proportional depth mapping plus 2D interpolation because that matches the current preprocessing path and is cheap, deterministic, and explainable.
- Artifact metadata must preserve enough context for future consumers:
  - source space
  - target space
  - selected slice indices
  - shapes before and after mapping
  - mapping strategy
- Training and inference behavior must remain unchanged; this is a results/export enhancement.

## Artifact Contract

Per case and phase, export:

- `model_input_overlay.png`
  - The current overlay in model-input space.
- `original_slice_overlay.png`
  - Heatmap resized onto the mapped native slice.
- `original_slice.png`
  - The native slice without overlay for side-by-side review.

Manifest entry shape should keep backward compatibility for existing consumers by retaining the current top-level `image_path` and `slice_index`, but add richer fields:

```json
{
  "phase": "arterial",
  "method": "gradcam_3d_slice_overlay",
  "image_path": ".../model_input_overlay.png",
  "slice_index": 7,
  "render_space": "model_input",
  "renderings": [
    {
      "space": "model_input",
      "kind": "overlay",
      "image_path": ".../model_input_overlay.png",
      "slice_index": 7,
      "image_shape": [64, 64]
    },
    {
      "space": "original_image",
      "kind": "overlay",
      "image_path": ".../original_slice_overlay.png",
      "slice_index": 58,
      "image_shape": [512, 512]
    },
    {
      "space": "original_image",
      "kind": "base_slice",
      "image_path": ".../original_slice.png",
      "slice_index": 58,
      "image_shape": [512, 512]
    }
  ],
  "mapping": {
    "strategy": "proportional_depth",
    "source_depth": 16,
    "target_depth": 118,
    "source_slice_index": 7,
    "target_slice_index": 58
  }
}
```

This keeps the current consumer path working while making the new original-space rendering explicit.

## Task 1: Add shared slice-mapping and overlay-rendering primitives

**Files:**
- Modify: `med_core/shared/visualization/heatmaps.py`
- Test: `tests/test_heatmap_visualization.py`

- [ ] **Step 1: Write the failing shared-visualization tests**

Create `tests/test_heatmap_visualization.py` with focused tests for:

- mapping a selected slice from one depth to another
- resizing a 2D heatmap into a target image shape
- rendering an overlay from a grayscale source slice and resized attention map
- preserving metadata for source/target spaces

Include concrete tests like:

```python
def test_map_slice_index_between_depths_uses_proportional_strategy() -> None:
    mapped = map_slice_index_between_depths(
        source_index=7,
        source_depth=16,
        target_depth=118,
    )
    assert mapped == 58


def test_resize_heatmap_slice_to_target_shape() -> None:
    attention = np.ones((8, 8), dtype=np.float32)
    resized = resize_heatmap_slice(attention, target_shape=(512, 512))
    assert resized.shape == (512, 512)
    assert float(resized.max()) <= 1.0
```

- [ ] **Step 2: Run the new shared-visualization tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_heatmap_visualization.py
```

Expected:
- FAIL because the new mapping / resizing helpers do not exist yet

- [ ] **Step 3: Implement minimal generic helpers in `heatmaps.py`**

Add generic utilities such as:

- `map_slice_index_between_depths(...)`
- `resize_heatmap_slice(...)`
- `render_overlay_artifact(...)`
- `build_rendering_metadata(...)`

Implementation rules:

- accept plain arrays and shapes
- normalize safely
- avoid any dataset-specific naming
- keep existing CAM helpers intact

- [ ] **Step 4: Run the shared-visualization tests to verify they pass**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_heatmap_visualization.py
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/shared/visualization/heatmaps.py \
  tests/test_heatmap_visualization.py
git commit -m "feat: add reusable original-slice heatmap helpers"
```

## Task 2: Expose native rendering context from the three-phase dataset without changing training samples

**Files:**
- Modify: `med_core/datasets/three_phase_ct.py`
- Test: `tests/test_three_phase_ct_dataset.py`

- [ ] **Step 1: Write the failing dataset tests**

Extend `tests/test_three_phase_ct_dataset.py` to assert that postprocessing can request a phase rendering context containing:

- native source volume shape
- resampled model-input volume shape
- a native slice array for a mapped index

Example expectation:

```python
def test_dataset_can_return_native_phase_render_context(tmp_path: Path) -> None:
    dataset = build_dataset_with_one_case(tmp_path)
    context = dataset.get_phase_render_context(index=0, phase_name="arterial")
    assert context["original_shape"] == (6, 16, 16)
    assert context["model_shape"] == (4, 8, 8)
```

- [ ] **Step 2: Run the focused dataset tests to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_three_phase_ct_dataset.py
```

Expected:
- FAIL because there is no rendering-context API yet

- [ ] **Step 3: Add a narrow dataset-side rendering API**

Modify `med_core/datasets/three_phase_ct.py` to add a postprocessing-only helper such as:

- `get_phase_render_context(index: int, phase_name: str) -> dict[str, Any]`

The method should:

- load the native source volume
- expose original depth/height/width
- expose the already-resampled model-input shape
- provide access to a requested native slice after mapping

Constraints:

- do not alter `__getitem__`
- do not change the tensors used for training
- keep phase naming aligned with the existing three-phase config

- [ ] **Step 4: Run the dataset tests to verify they pass**

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
git commit -m "feat: expose native render context for three-phase heatmaps"
```

## Task 3: Export model-space and original-slice heatmap artifacts from build-results

**Files:**
- Modify: `med_core/postprocessing/results.py`
- Modify: `tests/test_three_phase_mainline_build_results.py`

- [ ] **Step 1: Write the failing build-results integration test**

Extend `tests/test_three_phase_mainline_build_results.py` so the heatmap artifact test asserts:

- both model-space and original-space renderings are exported
- `original_slice_overlay.png` exists for each phase
- `original_slice.png` exists for each phase
- manifest items include:
  - `renderings`
  - `mapping`
  - `original_image` rendering metadata
- `case_explanations.json` includes the richer heatmap artifact payload

Example assertion shape:

```python
first_item = manifest["cases"][0]["heatmaps"][0]
assert any(view["space"] == "original_image" for view in first_item["renderings"])
assert first_item["mapping"]["strategy"] == "proportional_depth"
```

- [ ] **Step 2: Run the focused build-results test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_build_results.py::test_three_phase_build_results_emits_heatmap_artifacts_when_enabled
```

Expected:
- FAIL because only model-input overlays are exported today

- [ ] **Step 3: Implement original-slice overlay export in `results.py`**

Update `_generate_three_phase_heatmap_artifacts(...)` to:

- keep the current model-space overlay export
- use the shared helpers to map the selected slice to native depth
- resize the heatmap slice to the native slice resolution
- save:
  - `model_input_overlay.png`
  - `original_slice_overlay.png`
  - `original_slice.png`
- populate rendering provenance in manifest entries

Implementation notes:

- keep the current top-level `image_path` pointing at the model-space overlay for compatibility
- add new `renderings` and `mapping` fields instead of replacing the existing schema outright
- ensure report text says this is a "关注热区叠加图" / "原始切片叠加图", not a lesion contour

- [ ] **Step 4: Run the focused build-results test to verify it passes**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_build_results.py::test_three_phase_build_results_emits_heatmap_artifacts_when_enabled
```

Expected:
- PASS

- [ ] **Step 5: Run the broader three-phase regression tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_heatmap_visualization.py \
  tests/test_three_phase_ct_dataset.py \
  tests/test_three_phase_mainline_build_results.py
```

Expected:
- PASS

- [ ] **Step 6: Commit**

```bash
git add \
  med_core/postprocessing/results.py \
  med_core/datasets/three_phase_ct.py \
  med_core/shared/visualization/heatmaps.py \
  tests/test_heatmap_visualization.py \
  tests/test_three_phase_ct_dataset.py \
  tests/test_three_phase_mainline_build_results.py
git commit -m "feat: export original-slice overlays for three-phase heatmaps"
```

## Task 4: Tighten reporting language and artifact discoverability for clinician-facing review

**Files:**
- Modify: `med_core/postprocessing/results.py`
- Modify: `tests/test_three_phase_mainline_build_results.py`

- [ ] **Step 1: Write the failing report/summary assertions**

Extend existing assertions so report and summary mention:

- original-slice overlay availability
- mapping strategy disclosure
- clinician-safe wording that avoids overclaiming

Concrete expectations:

- report contains `原始切片叠加图`
- report contains `映射策略`
- summary artifact payload includes the enriched heatmap manifest path only once, without duplicating paths elsewhere

- [ ] **Step 2: Run the focused assertions to verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_three_phase_mainline_build_results.py::test_three_phase_build_results_emits_heatmap_artifacts_when_enabled
```

Expected:
- FAIL because current report text only describes generic imaging heatmaps

- [ ] **Step 3: Update the report and summary text**

In `med_core/postprocessing/results.py`, adjust the heatmap/report wording so it reads like a clinician-facing explanation:

- `影像关注热区热图`
- `原始切片叠加图`
- `映射策略：按重采样前后层面比例回映`

Keep the wording generic enough for other modalities and datasets.

- [ ] **Step 4: Re-run the focused assertion and the broader regression tests**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_heatmap_visualization.py \
  tests/test_three_phase_ct_dataset.py \
  tests/test_three_phase_mainline_build_results.py
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add \
  med_core/postprocessing/results.py \
  tests/test_three_phase_mainline_build_results.py
git commit -m "docs: clarify original-slice heatmap reporting"
```

## Verification Checklist

Before calling the work complete:

- [ ] `tests/test_heatmap_visualization.py` passes
- [ ] `tests/test_three_phase_ct_dataset.py` passes
- [ ] `tests/test_three_phase_mainline_build_results.py` passes
- [ ] Existing three-phase config validation/tests still pass if touched
- [ ] Real demo run still exports:
  - ROC
  - confusion matrix
  - SHAP
  - phase importance
  - case explanations
  - model-input heatmaps
  - original-slice overlays
- [ ] Manifest entries explicitly disclose mapping strategy and source/target spaces
- [ ] No shared helper contains demo/business naming

## Non-Goals

This plan does not add:

- voxel-accurate multi-phase registration
- lesion segmentation masks
- a promise that the heatmap is a contour
- Web UI heatmap gallery changes

Those can be planned later after the artifact contract stabilizes.
