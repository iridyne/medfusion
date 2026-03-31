from pathlib import Path

import numpy as np

from med_core.shared.visualization.heatmaps import (
    build_rendering_metadata,
    map_slice_index_between_depths,
    render_overlay_artifact,
    resize_heatmap_slice,
)


def test_map_slice_index_between_depths_preserves_endpoints() -> None:
    assert map_slice_index_between_depths(
        source_index=0,
        source_depth=16,
        target_depth=118,
    ) == 0
    assert map_slice_index_between_depths(
        source_index=15,
        source_depth=16,
        target_depth=118,
    ) == 117
    assert map_slice_index_between_depths(
        source_index=7,
        source_depth=16,
        target_depth=118,
    ) == 55


def test_resize_heatmap_slice_to_target_shape() -> None:
    attention = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

    resized = resize_heatmap_slice(attention, target_shape=(32, 24))

    assert resized.shape == (32, 24)
    assert resized.dtype == np.float32
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 1.0


def test_build_rendering_metadata_records_render_space() -> None:
    metadata = build_rendering_metadata(
        space="original_image",
        kind="overlay",
        image_path="/tmp/case_001_overlay.png",
        slice_index=58,
        image_shape=(512, 512),
    )

    assert metadata == {
        "space": "original_image",
        "kind": "overlay",
        "image_path": "/tmp/case_001_overlay.png",
        "slice_index": 58,
        "image_shape": [512, 512],
    }


def test_render_overlay_artifact_writes_output_and_metadata(tmp_path: Path) -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    attention = np.eye(8, dtype=np.float32)
    save_path = tmp_path / "overlay.png"

    metadata = render_overlay_artifact(
        image_slice=image,
        attention_slice=attention,
        save_path=save_path,
        space="original_image",
        kind="overlay",
        slice_index=3,
        title="Case 001",
    )

    assert save_path.exists()
    assert metadata["space"] == "original_image"
    assert metadata["kind"] == "overlay"
    assert metadata["image_path"] == str(save_path)
    assert metadata["slice_index"] == 3
    assert metadata["image_shape"] == [8, 8]
