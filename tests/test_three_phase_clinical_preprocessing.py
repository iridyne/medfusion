import numpy as np

from med_core.shared.preprocessing.clinical import ClinicalFeaturePreprocessor


def test_zero_with_mask_outputs_scaled_values_and_missing_mask() -> None:
    rows = [[10.0, None], [14.0, 2.0]]
    preprocessor = ClinicalFeaturePreprocessor(
        strategy="zero_with_mask",
        normalize=True,
    )

    payload = preprocessor.fit_transform(rows)

    assert payload.values.shape == (2, 2)
    assert payload.missing_mask.shape == (2, 2)
    assert payload.missing_mask.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert np.isclose(payload.values[0, 0], -1.0)
    assert np.isclose(payload.values[1, 0], 1.0)
    assert payload.values[0, 1] == 0.0


def test_fit_ignores_missing_values_when_estimating_stats() -> None:
    rows = [[10.0, None], [14.0, 2.0], [None, 6.0]]
    preprocessor = ClinicalFeaturePreprocessor(
        strategy="zero_with_mask",
        normalize=True,
    )

    preprocessor.fit(rows)

    assert np.allclose(preprocessor.means, np.asarray([12.0, 4.0], dtype=np.float32))
    assert np.allclose(preprocessor.stds, np.asarray([2.0, 2.0], dtype=np.float32))


def test_preprocessor_roundtrips_via_dict_payload() -> None:
    rows = [[10.0, None], [14.0, 2.0], [12.0, 6.0]]
    fitted = ClinicalFeaturePreprocessor(
        strategy="zero_with_mask",
        normalize=True,
    )
    fitted.fit(rows)

    restored = ClinicalFeaturePreprocessor.from_dict(fitted.to_dict())
    transformed = restored.transform([[12.0, None], [14.0, 6.0]])

    assert transformed.values.shape == (2, 2)
    assert transformed.missing_mask.tolist() == [[0.0, 1.0], [0.0, 0.0]]
    assert np.isclose(transformed.values[0, 0], 0.0)
