"""Tests for train CLI config-driven runtime helpers."""

from torch import nn, optim

from med_core.cli.train import _build_optimizer, _build_scheduler


def test_build_optimizer_respects_requested_type() -> None:
    model = nn.Linear(4, 2)

    adam = _build_optimizer(model, "adam", 1e-3, 0.0, 0.9)
    adamw = _build_optimizer(model, "adamw", 1e-3, 1e-2, 0.9)
    sgd = _build_optimizer(model, "sgd", 1e-2, 0.0, 0.8)

    assert isinstance(adam, optim.Adam)
    assert isinstance(adamw, optim.AdamW)
    assert isinstance(sgd, optim.SGD)
    assert sgd.param_groups[0]["momentum"] == 0.8


def test_build_scheduler_respects_requested_type() -> None:
    model = nn.Linear(4, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    step = _build_scheduler(
        optimizer=optimizer,
        scheduler_name="step",
        num_epochs=5,
        train_loader_length=3,
        min_lr=1e-6,
        step_size=2,
        gamma=0.5,
        patience=3,
        factor=0.5,
        mode="max",
    )
    plateau = _build_scheduler(
        optimizer=optimizer,
        scheduler_name="plateau",
        num_epochs=5,
        train_loader_length=3,
        min_lr=1e-6,
        step_size=2,
        gamma=0.5,
        patience=3,
        factor=0.5,
        mode="max",
    )
    onecycle = _build_scheduler(
        optimizer=optimizer,
        scheduler_name="onecycle",
        num_epochs=5,
        train_loader_length=3,
        min_lr=1e-6,
        step_size=2,
        gamma=0.5,
        patience=3,
        factor=0.5,
        mode="max",
    )
    none_scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name="none",
        num_epochs=5,
        train_loader_length=3,
        min_lr=1e-6,
        step_size=2,
        gamma=0.5,
        patience=3,
        factor=0.5,
        mode="max",
    )

    assert isinstance(step, optim.lr_scheduler.StepLR)
    assert isinstance(plateau, optim.lr_scheduler.ReduceLROnPlateau)
    assert isinstance(onecycle, optim.lr_scheduler.OneCycleLR)
    assert none_scheduler is None
