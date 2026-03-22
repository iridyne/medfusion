import pytest

from med_core.web.services import RemovedTrainingServiceError, TrainingService


def test_legacy_training_service_raises_clear_guidance() -> None:
    with pytest.raises(RemovedTrainingServiceError) as exc_info:
        TrainingService()

    message = str(exc_info.value)
    assert "/api/training/start" in message
    assert "medfusion train" in message
