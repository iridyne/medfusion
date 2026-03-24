#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from med_core.models import MultiModalModelBuilder
from med_core.postprocessing.advanced_analysis import (
    build_shap_artifacts,
    build_survival_artifacts as build_posthoc_survival_artifacts,
)


@dataclass
class RunPaths:
    root: Path
    output_dir: Path
    checkpoints_dir: Path
    reports_dir: Path


@dataclass
class TabularPack:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    feature_names: list[str]


def parse_pathology_encoder_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """统一解析 pathology_encoder 配置。"""
    raw = model_cfg.get("pathology_encoder", "patch_mil")

    if isinstance(raw, dict):
        encoder_type = str(raw.get("type", "patch_mil")).strip().lower()
        embedding_dim = int(
            raw.get(
                "embedding_dim",
                model_cfg.get("hipt_embedding_dim", 192),
            )
        )
    else:
        encoder_type = str(raw).strip().lower()
        embedding_dim = int(model_cfg.get("hipt_embedding_dim", 192))

    if encoder_type not in {"patch_mil", "hipt"}:
        raise ValueError(
            f"Unsupported pathology_encoder={encoder_type}, expected patch_mil|hipt",
        )

    return {
        "type": encoder_type,
        "embedding_dim": embedding_dim,
    }


def _mock_hipt_projection_matrix(embedding_dim: int) -> np.ndarray:
    rng = np.random.default_rng(20260324)
    return rng.normal(0.0, 1.0, size=(8, embedding_dim)).astype(np.float32) / np.sqrt(8.0)


def compute_mock_hipt_embedding(image_or_patch: np.ndarray, embedding_dim: int) -> np.ndarray:
    """用轻量统计特征 + 固定随机投影，模拟离线 HIPT embedding。"""
    arr = np.asarray(image_or_patch, dtype=np.float32)
    if arr.ndim == 3:
        # [C,H,W] -> [H,W] 近似
        arr = arr.mean(axis=0)

    flat = arr.reshape(-1)
    stats = np.array(
        [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.quantile(flat, 0.25)),
            float(np.quantile(flat, 0.5)),
            float(np.quantile(flat, 0.75)),
            float(np.mean(np.square(flat))),
        ],
        dtype=np.float32,
    )

    proj = _mock_hipt_projection_matrix(embedding_dim)
    embedding = stats @ proj
    return embedding.astype(np.float32)


class MultiRegionSmurfDataset(Dataset):
    """输出格式:
    - survival 开启: (inputs_dict, label, time, event)
    - survival 关闭: (inputs_dict, label)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tabular_matrix: np.ndarray,
        data_cfg: dict[str, Any],
        project_root: Path,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.tab = tabular_matrix.astype(np.float32)
        self.cfg = data_cfg
        self.root = project_root
        self.max_instances = int(data_cfg.get("max_instances_per_modality", 3))
        self.image_size = int(data_cfg.get("image_size", 64))

        self.enable_survival = bool(data_cfg.get("enable_survival", True))
        self.time_col = str(data_cfg.get("survival_time_column", "survival_time"))
        self.event_col = str(data_cfg.get("survival_event_column", "event"))

        self.pathology_encoder_type = str(
            data_cfg.get("pathology_encoder_type", "patch_mil")
        ).lower()
        self.hipt_embedding_dim = int(data_cfg.get("hipt_embedding_dim", 192))

        # 单分支模式：只使用 clinical + ct
        self.single_branch_mode = bool(data_cfg.get("single_branch_mode", False))
        self.ct_column = str(data_cfg.get("ct_column", "ct_paths"))

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, value: str) -> Path:
        p = Path(value)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()

    def _parse_sequence_field(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]
        if value is None:
            return []

        s = str(value).strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr if str(x).strip()]
            except json.JSONDecodeError:
                pass

        if "|" in s:
            return [x.strip() for x in s.split("|") if x.strip()]

        return [s]

    def _normalize_to_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"expected 2D/3D array, got shape={arr.shape}")

        c, h, w = arr.shape
        if c == 1:
            arr = np.repeat(arr, 3, axis=0)
        elif c != 3:
            if c > 3:
                arr = arr[:3]
            else:
                pad = np.repeat(arr[-1:, ...], 3 - c, axis=0)
                arr = np.concatenate([arr, pad], axis=0)

        if h != self.image_size or w != self.image_size:
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            arr = tensor.squeeze(0).numpy()

        return arr.astype(np.float32)

    def _select_non_contiguous(self, paths: list[str]) -> list[str]:
        if not paths:
            return []
        if len(paths) <= self.max_instances:
            return paths
        idx = np.linspace(0, len(paths) - 1, self.max_instances, dtype=int).tolist()
        return [paths[i] for i in idx]

    def _normalize_embedding_vector(self, arr: np.ndarray) -> np.ndarray:
        vec = np.asarray(arr, dtype=np.float32).reshape(-1)
        if vec.size == self.hipt_embedding_dim:
            return vec

        if vec.size > self.hipt_embedding_dim:
            return vec[: self.hipt_embedding_dim]

        out = np.zeros((self.hipt_embedding_dim,), dtype=np.float32)
        out[: vec.size] = vec
        return out

    def _load_embedding_sequence(self, cell_value: Any) -> torch.Tensor:
        paths = self._parse_sequence_field(cell_value)
        paths = self._select_non_contiguous(paths)

        if not paths:
            return torch.zeros(
                (self.max_instances, self.hipt_embedding_dim),
                dtype=torch.float32,
            )

        embeddings: list[np.ndarray] = []
        for p in paths:
            arr = np.load(self._resolve_path(p)).astype(np.float32)
            vec = self._normalize_embedding_vector(arr)
            embeddings.append(vec)

        while len(embeddings) < self.max_instances:
            embeddings.append(embeddings[-1].copy())

        stack = np.stack(embeddings[: self.max_instances], axis=0)  # [N,D]
        return torch.from_numpy(stack)

    def _load_sequence(self, cell_value: Any, modality_name: str) -> torch.Tensor:
        if self.pathology_encoder_type == "hipt" and "pathology" in modality_name:
            return self._load_embedding_sequence(cell_value)

        paths = self._parse_sequence_field(cell_value)
        paths = self._select_non_contiguous(paths)

        if not paths:
            dummy = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            stack = np.stack([dummy] * self.max_instances, axis=0)
            return torch.from_numpy(stack)

        images: list[np.ndarray] = []
        for p in paths:
            arr = np.load(self._resolve_path(p)).astype(np.float32)
            arr = self._normalize_to_rgb(arr)
            images.append(arr)

        while len(images) < self.max_instances:
            images.append(images[-1].copy())

        stack = np.stack(images[: self.max_instances], axis=0)  # [N,3,H,W]
        return torch.from_numpy(stack)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        inputs = {
            "clinical": torch.from_numpy(self.tab[idx]),
        }

        if self.single_branch_mode:
            inputs["ct"] = self._load_sequence(row[self.ct_column], "ct")
        else:
            inputs.update(
                {
                    "region1_ct": self._load_sequence(
                        row[self.cfg["region1_ct_column"]], "region1_ct"
                    ),
                    "region1_pathology": self._load_sequence(
                        row[self.cfg["region1_pathology_column"]], "region1_pathology"
                    ),
                    "region2_ct": self._load_sequence(
                        row[self.cfg["region2_ct_column"]], "region2_ct"
                    ),
                    "region2_pathology": self._load_sequence(
                        row[self.cfg["region2_pathology_column"]], "region2_pathology"
                    ),
                }
            )

        label = torch.tensor(int(row[self.cfg["label_column"]]), dtype=torch.long)

        if self.enable_survival:
            t = torch.tensor(float(row[self.time_col]), dtype=torch.float32)
            e = torch.tensor(float(row[self.event_col]), dtype=torch.float32)
            return inputs, label, t, e

        return inputs, label


class MultiTaskSmurfModel(nn.Module):
    """共享 backbone/fusion, 双头输出:
    - classification logits
    - continuous risk score (SMuRF-score 风格)
    """

    def __init__(self, base_model: nn.Module, fusion_dim: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.risk_head = nn.Linear(fusion_dim, 1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features = self.base_model(inputs, return_features=True)
        fused = features["fused_features"]
        risk = self.risk_head(fused).squeeze(1)
        return logits, risk


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Cox partial likelihood loss (negative log)."""
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]

    log_cumsum_exp = torch.logcumsumexp(r, dim=0)
    observed = e > 0.5

    if torch.sum(observed) == 0:
        return torch.zeros((), device=risk.device, dtype=risk.dtype)

    loss = -(r[observed] - log_cumsum_exp[observed]).mean()
    return loss


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    n = len(time)
    num = 0.0
    den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                continue

            # comparable if earlier time has event
            if time[i] < time[j] and event[i] == 1:
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
            elif time[j] < time[i] and event[j] == 1:
                den += 1
                if risk[j] > risk[i]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5

    if den == 0:
        return 0.0
    return float(num / den)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_root(script_path: Path) -> Path:
    return script_path.resolve().parents[2]


def resolve_paths(config: dict[str, Any], project_root: Path) -> RunPaths:
    output_dir = (project_root / config.get("output_dir", "demo/smurf_e2e/outputs")).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=project_root,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        reports_dir=reports_dir,
    )


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _build_tabular_pack(df: pd.DataFrame, data_cfg: dict[str, Any]) -> TabularPack:
    split_col = data_cfg.get("split_column", "split")
    num_cols = list(data_cfg.get("tabular_numerical_columns", []))
    cat_cols = list(data_cfg.get("tabular_categorical_columns", []))

    train_df = df[df[split_col].str.lower() == "train"].copy()
    val_df = df[df[split_col].str.lower() == "val"].copy()
    test_df = df[df[split_col].str.lower() == "test"].copy()

    def transform_block(block: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=block.index)
        for c in num_cols:
            out[c] = pd.to_numeric(block[c], errors="coerce").fillna(0.0)
        if cat_cols:
            dummies = pd.get_dummies(block[cat_cols].astype(str), prefix=cat_cols)
            out = pd.concat([out, dummies], axis=1)
        return out

    train_x = transform_block(train_df)
    val_x = transform_block(val_df)
    test_x = transform_block(test_df)

    all_cols = train_x.columns.tolist()
    val_x = val_x.reindex(columns=all_cols, fill_value=0.0)
    test_x = test_x.reindex(columns=all_cols, fill_value=0.0)

    return TabularPack(
        train=train_x.to_numpy(dtype=np.float32),
        val=val_x.to_numpy(dtype=np.float32),
        test=test_x.to_numpy(dtype=np.float32),
        feature_names=all_cols,
    )


def load_split_frames(config: dict[str, Any], project_root: Path):
    data_cfg = config["data"]
    csv_path = (project_root / data_cfg["csv_path"]).resolve()
    split_col = data_cfg.get("split_column", "split")

    if not csv_path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for needed in ["train", "val", "test"]:
        if needed not in set(df[split_col].astype(str).str.lower().unique()):
            raise ValueError(f"split '{needed}' missing in column '{split_col}'")

    pack = _build_tabular_pack(df, data_cfg)

    train_df = df[df[split_col].str.lower() == "train"].copy()
    val_df = df[df[split_col].str.lower() == "val"].copy()
    test_df = df[df[split_col].str.lower() == "test"].copy()
    return train_df, val_df, test_df, pack


def build_backbone_model(config: dict[str, Any], tabular_input_dim: int) -> nn.Module:
    mcfg = config["model"]
    data_cfg = config["data"]
    encoder_cfg = parse_pathology_encoder_config(mcfg)

    single_branch_mode = bool(data_cfg.get("single_branch_mode", False))

    builder = MultiModalModelBuilder()
    builder.add_modality(
        "clinical",
        backbone="mlp",
        modality_type="tabular",
        input_dim=tabular_input_dim,
        feature_dim=int(mcfg.get("tabular_feature_dim", 64)),
    )

    vision_backbone = str(mcfg.get("vision_backbone", "mobilenetv3_small"))
    vision_dim = int(mcfg.get("vision_feature_dim", 96))

    if single_branch_mode:
        builder.add_modality(
            "ct",
            backbone=vision_backbone,
            modality_type="vision",
            feature_dim=vision_dim,
        )
        builder.add_mil_aggregation(
            "ct",
            strategy="attention",
            attention_dim=max(vision_dim // 2, 16),
        )
    else:
        pathology_feature_dim = int(mcfg.get("pathology_feature_dim", vision_dim))
        ct_modalities = ["region1_ct", "region2_ct"]
        pathology_modalities = ["region1_pathology", "region2_pathology"]

        for modality_name in ct_modalities:
            builder.add_modality(
                modality_name,
                backbone=vision_backbone,
                modality_type="vision",
                feature_dim=vision_dim,
            )
            builder.add_mil_aggregation(
                modality_name,
                strategy="attention",
                attention_dim=max(vision_dim // 2, 16),
            )

        for modality_name in pathology_modalities:
            if encoder_cfg["type"] == "hipt":
                builder.add_modality(
                    modality_name,
                    backbone="hipt",
                    modality_type="embedding",
                    input_dim=int(encoder_cfg["embedding_dim"]),
                    feature_dim=pathology_feature_dim,
                    dropout=0.1,
                )
            else:
                builder.add_modality(
                    modality_name,
                    backbone=vision_backbone,
                    modality_type="vision",
                    feature_dim=pathology_feature_dim,
                )

            builder.add_mil_aggregation(
                modality_name,
                strategy="attention",
                attention_dim=max(pathology_feature_dim // 2, 16),
            )

    builder.set_fusion(
        strategy=str(mcfg.get("fusion_strategy", "fused_attention")),
        output_dim=int(mcfg.get("fusion_output_dim", 128)),
        num_heads=int(mcfg.get("fusion_num_heads", 4)),
        dropout=0.1,
    )

    builder.set_head(
        task_type="classification",
        num_classes=int(mcfg.get("num_classes", 2)),
        dropout=0.2,
    )
    return builder.build()


def make_model(config: dict[str, Any], tabular_input_dim: int) -> MultiTaskSmurfModel:
    backbone = build_backbone_model(config, tabular_input_dim)
    fusion_dim = int(config["model"].get("fusion_output_dim", 128))
    return MultiTaskSmurfModel(backbone, fusion_dim)


def _build_dataset_cfg(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(config["data"])
    encoder_cfg = parse_pathology_encoder_config(config["model"])
    data_cfg["pathology_encoder_type"] = encoder_cfg["type"]
    data_cfg["hipt_embedding_dim"] = int(encoder_cfg["embedding_dim"])

    if bool(data_cfg.get("single_branch_mode", False)):
        data_cfg["ct_column"] = str(
            data_cfg.get("ct_column", data_cfg.get("region1_ct_column", "ct_paths"))
        )

    return data_cfg


def make_loaders(config: dict[str, Any], project_root: Path):
    train_df, val_df, test_df, pack = load_split_frames(config, project_root)
    data_cfg = _build_dataset_cfg(config)

    train_ds = MultiRegionSmurfDataset(train_df, pack.train, data_cfg, project_root)
    val_ds = MultiRegionSmurfDataset(val_df, pack.val, data_cfg, project_root)
    test_ds = MultiRegionSmurfDataset(test_df, pack.test, data_cfg, project_root)

    batch_size = int(config["training"].get("batch_size", 2))
    num_workers = int(config["training"].get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, pack.train.shape[1]


def make_datasets(config: dict[str, Any], project_root: Path):
    train_df, val_df, test_df, pack = load_split_frames(config, project_root)
    data_cfg = _build_dataset_cfg(config)
    train_ds = MultiRegionSmurfDataset(train_df, pack.train, data_cfg, project_root)
    val_ds = MultiRegionSmurfDataset(val_df, pack.val, data_cfg, project_root)
    test_ds = MultiRegionSmurfDataset(test_df, pack.test, data_cfg, project_root)
    return (train_df, val_df, test_df), (train_ds, val_ds, test_ds), pack


def run_epoch(
    model: MultiTaskSmurfModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    enable_survival: bool,
    survival_loss_weight: float,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_surv = 0.0
    preds_all: list[int] = []
    labels_all: list[int] = []

    for batch in loader:
        if len(batch) == 4:
            inputs, labels, time, event = batch
            time = time.to(device)
            event = event.to(device)
        else:
            inputs, labels = batch
            time = None
            event = None

        labels = labels.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits, risk = model(inputs)
            cls_loss = criterion(logits, labels)
            surv_loss = torch.zeros((), device=device)

            if enable_survival and time is not None and event is not None:
                surv_loss = cox_ph_loss(risk, time, event)

            loss = cls_loss + survival_loss_weight * surv_loss

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_cls += float(cls_loss.item()) * labels.size(0)
        total_surv += float(surv_loss.item()) * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    n = max(len(loader.dataset), 1)
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls / n,
        "surv_loss": total_surv / n,
        "acc": float(accuracy_score(labels_all, preds_all)) if labels_all else 0.0,
    }


def evaluate_loader(model: MultiTaskSmurfModel, loader: DataLoader, device: torch.device):
    model.eval()
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    risk_list: list[np.ndarray] = []
    time_list: list[np.ndarray] = []
    event_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, labels, time, event = batch
                time_list.append(time.numpy())
                event_list.append(event.numpy())
            else:
                inputs, labels = batch

            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits, risk = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            risk_list.append(risk.cpu().numpy())

    probs_all = np.concatenate(probs_list, axis=0) if probs_list else np.zeros((0, 2))
    preds_all = np.concatenate(preds_list, axis=0) if preds_list else np.zeros((0,), dtype=int)
    labels_all = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0,), dtype=int)
    risk_all = np.concatenate(risk_list, axis=0) if risk_list else np.zeros((0,), dtype=float)

    time_all = np.concatenate(time_list, axis=0) if time_list else np.zeros((0,), dtype=float)
    event_all = np.concatenate(event_list, axis=0) if event_list else np.zeros((0,), dtype=float)

    return probs_all, preds_all, labels_all, risk_all, time_all, event_all


def _extract_attention_payload(
    model: MultiTaskSmurfModel,
    dataset: MultiRegionSmurfDataset,
    device: torch.device,
    sample_index: int,
) -> dict[str, Any]:
    raw = dataset[sample_index]
    if len(raw) == 4:
        inputs, label, time_value, event_value = raw
        time_scalar = float(time_value.item())
        event_scalar = float(event_value.item())
    else:
        inputs, label = raw
        time_scalar = None
        event_scalar = None

    batch_inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits, features = model.base_model(batch_inputs, return_features=True)
        fused = features["fused_features"]
        risk = model.risk_head(fused).squeeze(1)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    mil_attention: dict[str, list[float]] = {}
    for modality_name, weights in features.get("mil_attention_weights", {}).items():
        mil_attention[modality_name] = (
            weights.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(float).tolist()
        )

    fusion_aux = features.get("fusion_aux") or {}
    modality_attention = None
    if isinstance(fusion_aux, dict):
        if "multimodal_attention_weights" in fusion_aux:
            modality_attention = (
                fusion_aux["multimodal_attention_weights"]
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
            )
        elif "attention_weights" in fusion_aux and isinstance(fusion_aux["attention_weights"], torch.Tensor):
            arr = fusion_aux["attention_weights"].detach().cpu().numpy()
            modality_attention = (
                arr.mean(axis=tuple(range(arr.ndim - 1))).astype(float).tolist()
                if arr.ndim > 1
                else arr.astype(float).tolist()
            )

    return {
        "sample_index": sample_index,
        "true_label": int(label.item()),
        "pred_label": int(pred.item()),
        "probabilities": probs.squeeze(0).detach().cpu().numpy().astype(float).tolist(),
        "risk_logit": float(risk.item()),
        "smurf_score": float(torch.sigmoid(risk).item()),
        "time": time_scalar,
        "event": event_scalar,
        "mil_attention": mil_attention,
        "modality_attention": modality_attention,
    }


def _plot_attention_vector(values: list[float], title: str, save_path: Path, color: str = "#4c78a8") -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(values)), values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Attention")
    ax.set_ylim(0.0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_attention_artifacts(
    model: MultiTaskSmurfModel,
    dataset: MultiRegionSmurfDataset,
    device: torch.device,
    output_dir: Path,
    sample_limit: int = 4,
) -> dict[str, Any]:
    attention_dir = output_dir / "visualizations" / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = list(range(min(sample_limit, len(dataset))))
    manifest_items: list[dict[str, Any]] = []

    for sample_index in selected_indices:
        payload = _extract_attention_payload(model, dataset, device, sample_index)

        for modality_name, weights in payload.get("mil_attention", {}).items():
            plot_name = f"sample_{sample_index}_{modality_name}_mil_attention.png"
            plot_path = attention_dir / plot_name
            _plot_attention_vector(
                weights,
                f"MIL Attention | {modality_name} | sample {sample_index}",
                plot_path,
            )
            manifest_items.append(
                {
                    "artifact_key": f"sample_{sample_index}_{modality_name}_mil_attention",
                    "title": f"MIL Attention | {modality_name} | sample {sample_index}",
                    "image_path": str(plot_path),
                    "sample_index": sample_index,
                    "modality": modality_name,
                    "type": "mil_attention",
                    "true_label": payload["true_label"],
                    "pred_label": payload["pred_label"],
                    "smurf_score": payload["smurf_score"],
                }
            )

        modality_attention = payload.get("modality_attention")
        if modality_attention:
            plot_name = f"sample_{sample_index}_fusion_attention.png"
            plot_path = attention_dir / plot_name
            _plot_attention_vector(
                modality_attention,
                f"Fusion Attention | sample {sample_index}",
                plot_path,
                color="#d62728",
            )
            manifest_items.append(
                {
                    "artifact_key": f"sample_{sample_index}_fusion_attention",
                    "title": f"Fusion Attention | sample {sample_index}",
                    "image_path": str(plot_path),
                    "sample_index": sample_index,
                    "type": "fusion_attention",
                    "true_label": payload["true_label"],
                    "pred_label": payload["pred_label"],
                    "smurf_score": payload["smurf_score"],
                }
            )

    manifest = {"items": manifest_items}
    manifest_path = attention_dir / "attention_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "items": manifest_items}


def _parse_sequence_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None:
        return []

    s = str(value).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x) for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass

    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]

    return [s]


def _resolve_path_from_project(project_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def prepare_hipt_embeddings(config: dict[str, Any], project_root: Path) -> tuple[Path, int]:
    """离线生成 pathology HIPT embedding，并把 CSV 病理列改写为 embedding 路径。"""
    data_cfg = config["data"]
    model_cfg = config["model"]
    encoder_cfg = parse_pathology_encoder_config(model_cfg)

    if encoder_cfg["type"] != "hipt":
        return (project_root / data_cfg["csv_path"]).resolve(), 0

    csv_path = (project_root / data_cfg["csv_path"]).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata CSV not found: {csv_path}")

    hipt_dir = (
        project_root
        / data_cfg.get("hipt_embeddings_dir", "demo/smurf_e2e/data/mock/hipt_embeddings")
    ).resolve()
    hipt_dir.mkdir(parents=True, exist_ok=True)

    pathology_columns = [
        str(data_cfg["region1_pathology_column"]),
        str(data_cfg["region2_pathology_column"]),
    ]

    df = pd.read_csv(csv_path)
    converted = 0

    for col in pathology_columns:
        if col not in df.columns:
            continue

        modality_dir = hipt_dir / col
        modality_dir.mkdir(parents=True, exist_ok=True)

        rewritten_cells: list[str] = []
        for row_idx, cell in enumerate(df[col].tolist()):
            raw_paths = _parse_sequence_cell(cell)
            new_paths: list[str] = []

            for seq_idx, raw_path in enumerate(raw_paths):
                src = _resolve_path_from_project(project_root, raw_path)
                if not src.exists():
                    continue

                arr = np.load(src).astype(np.float32)
                emb = compute_mock_hipt_embedding(arr, int(encoder_cfg["embedding_dim"]))

                out_name = f"r{row_idx:05d}_{seq_idx:03d}_hipt.npy"
                out_path = modality_dir / out_name
                np.save(out_path, emb)
                new_paths.append(str(out_path.relative_to(project_root)))
                converted += 1

            rewritten_cells.append("|".join(new_paths))

        df[col] = rewritten_cells

    df.to_csv(csv_path, index=False)
    return csv_path, converted


def cmd_prepare_mock(config_path: Path) -> None:
    config = load_yaml(config_path)
    project_root = resolve_project_root(Path(__file__))
    set_seed(int(config.get("seed", 42)))

    data_cfg = config["data"]
    mock_cfg = config["mock_data"]

    csv_path = (project_root / data_cfg["csv_path"]).resolve()
    data_dir = csv_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    modalities = {
        "region1_ct": data_cfg["region1_ct_column"],
        "region1_pathology": data_cfg["region1_pathology_column"],
        "region2_ct": data_cfg["region2_ct_column"],
        "region2_pathology": data_cfg["region2_pathology_column"],
    }

    for name in modalities:
        (data_dir / name).mkdir(parents=True, exist_ok=True)

    n = int(mock_cfg.get("num_samples", 96))
    min_instances = int(mock_cfg.get("min_instances", 2))
    max_instances = int(mock_cfg.get("max_instances", 6))
    h, w = [int(x) for x in mock_cfg.get("image_shape", [64, 64])]

    time_col = str(data_cfg.get("survival_time_column", "survival_time"))
    event_col = str(data_cfg.get("survival_event_column", "event"))

    rows: list[dict[str, Any]] = []
    for i in range(n):
        label = 0 if i < n // 2 else 1
        pid = f"case_{i:04d}"

        row: dict[str, Any] = {"patient_id": pid}
        row["age"] = float(np.random.normal(55 + 8 * label, 10))
        row["bmi"] = float(np.random.normal(23 + 2.5 * label, 3))
        row["crp"] = float(max(0.1, np.random.normal(2 + 1.2 * label, 1.5)))
        row["sex"] = random.choice(["M", "F"])
        row["smoking"] = random.choice(["yes", "no"]) if label == 1 else random.choice(["no", "no", "yes"])

        # 生成生存信息：高风险类 => 更短生存时间、更高事件概率
        if label == 1:
            row[time_col] = float(max(1.0, np.random.exponential(scale=14.0)))
            row[event_col] = float(np.random.binomial(1, 0.78))
        else:
            row[time_col] = float(max(1.0, np.random.exponential(scale=28.0)))
            row[event_col] = float(np.random.binomial(1, 0.42))

        for modality_name, col_name in modalities.items():
            k = random.randint(min_instances, max_instances)
            seq_paths: list[str] = []
            for j in range(k):
                img = np.random.normal(0, 1, size=(h, w)).astype(np.float32)
                if label == 1:
                    if "region1" in modality_name:
                        img[h // 5 : h // 2, w // 5 : w // 2] += 0.7
                    else:
                        img[h // 2 : h * 4 // 5, w // 2 : w * 4 // 5] += 0.55

                rel = Path("demo/smurf_e2e/data/mock") / modality_name / f"{pid}_{j:02d}.npy"
                np.save(project_root / rel, img)
                seq_paths.append(str(rel))

            row[col_name] = "|".join(seq_paths)

        row[data_cfg["label_column"]] = int(label)
        r = random.random()
        row[data_cfg.get("split_column", "split")] = "train" if r < 0.7 else ("val" if r < 0.85 else "test")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    encoder_cfg = parse_pathology_encoder_config(config["model"])
    if encoder_cfg["type"] == "hipt":
        _, converted = prepare_hipt_embeddings(config, project_root)
        print(
            f"[prepare-mock] done: {csv_path} ({len(df)} samples), "
            f"hipt_embeddings={converted}",
        )
    else:
        print(f"[prepare-mock] done: {csv_path} ({len(df)} samples)")


def cmd_prepare_hipt(config_path: Path) -> None:
    config = load_yaml(config_path)
    project_root = resolve_project_root(Path(__file__))
    csv_path, converted = prepare_hipt_embeddings(config, project_root)
    print(f"[prepare-hipt] csv: {csv_path}")
    print(f"[prepare-hipt] converted_embeddings: {converted}")


def cmd_train(config_path: Path) -> None:
    config = load_yaml(config_path)
    project_root = resolve_project_root(Path(__file__))
    paths = resolve_paths(config, project_root)
    set_seed(int(config.get("seed", 42)))

    enable_survival = bool(config["data"].get("enable_survival", True))
    surv_w = float(config["training"].get("survival_loss_weight", 0.3))

    device = resolve_device(str(config.get("device", "auto")))
    train_loader, val_loader, _, tab_dim = make_loaders(config, project_root)

    model = make_model(config, tab_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-2)),
    )

    epochs = int(config["training"].get("epochs", 4))
    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_ckpt = paths.checkpoints_dir / "best_smurf_multiregion.pth"

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, device, criterion, optimizer, enable_survival, surv_w)
        va = run_epoch(model, val_loader, device, criterion, None, enable_survival, surv_w)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_cls_loss": tr["cls_loss"],
                "train_surv_loss": tr["surv_loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_cls_loss": va["cls_loss"],
                "val_surv_loss": va["surv_loss"],
                "val_acc": va["acc"],
            }
        )

        print(
            f"[train] epoch={epoch:03d} "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
            f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}"
        )

        if va["acc"] >= best_val_acc:
            best_val_acc = va["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": config["model"],
                    "training_config": config["training"],
                    "data_config": config["data"],
                    "best_val_acc": best_val_acc,
                },
                best_ckpt,
            )

    history_path = paths.output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump({"entries": history, "best_val_acc": best_val_acc}, f, ensure_ascii=False, indent=2)

    print(f"[train] best checkpoint: {best_ckpt}")
    print(f"[train] history: {history_path}")


def cmd_evaluate(config_path: Path, checkpoint: Path | None = None) -> None:
    config = load_yaml(config_path)
    project_root = resolve_project_root(Path(__file__))
    paths = resolve_paths(config, project_root)

    enable_survival = bool(config["data"].get("enable_survival", True))

    device = resolve_device(str(config.get("device", "auto")))
    _, _, test_loader, tab_dim = make_loaders(config, project_root)

    ckpt_path = checkpoint or (paths.checkpoints_dir / "best_smurf_multiregion.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model = make_model(config, tab_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    probs, preds, labels, risk, time, event = evaluate_loader(model, test_loader, device)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) > 0 else 0.0,
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)) if len(labels) > 0 else 0.0,
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)) if len(labels) > 0 else 0.0,
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)) if len(labels) > 0 else 0.0,
        "num_samples": int(len(labels)),
    }

    if probs.shape[1] == 2 and len(labels) > 0:
        from sklearn.metrics import roc_auc_score

        metrics["auc"] = float(roc_auc_score(labels, probs[:, 1]))

    # 连续风险分数（SMuRF-score 风格）
    risk_score = 1.0 / (1.0 + np.exp(-risk)) if len(risk) > 0 else np.array([])
    metrics["risk_score_mean"] = float(risk_score.mean()) if len(risk_score) > 0 else 0.0

    if enable_survival and len(time) > 0:
        metrics["c_index"] = concordance_index(time, event, risk)

    metrics_path = paths.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    n_samples = int(len(labels))
    survival_time_col = time.tolist() if len(time) == n_samples else [None] * n_samples
    event_col = event.tolist() if len(event) == n_samples else [None] * n_samples

    pred_df = pd.DataFrame(
        {
            "label": labels.tolist(),
            "pred": preds.tolist(),
            "confidence": probs.max(axis=1).tolist() if len(probs) > 0 else [],
            "smurf_score": risk_score.tolist(),
            "risk_logit": risk.tolist(),
            "survival_time": survival_time_col,
            "event": event_col,
        }
    )
    pred_path = paths.output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Reuse mainline post-hoc analysis helpers
    (_, _, test_df), (train_ds, _, test_ds), pack = make_datasets(config, project_root)
    batch_size = int(config["training"].get("batch_size", 2))
    num_workers = int(config["training"].get("num_workers", 0))

    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train_probs, train_preds, train_labels, train_risk, train_time, train_event = evaluate_loader(model, train_eval_loader, device)
    train_risk_score = 1.0 / (1.0 + np.exp(-train_risk)) if len(train_risk) > 0 else np.array([])
    risk_cutoff = float(np.median(train_risk_score)) if len(train_risk_score) > 0 else None

    survival_payload = None
    shap_payload = None
    if enable_survival and len(time) > 0:
        survival_payload, _survival_artifacts, survival_metric_updates = build_posthoc_survival_artifacts(
            output_dir=paths.output_dir,
            metadata_frame=test_df.reset_index(drop=True),
            risk_scores=risk_score,
            time_column=str(config["data"].get("survival_time_column", "survival_time")),
            event_column=str(config["data"].get("survival_event_column", "event")),
            split="test",
            cutoff=risk_cutoff,
            cutoff_source="train_split_median" if risk_cutoff is not None else "current_split_median",
        )
        metrics.update({k: v for k, v in survival_metric_updates.items() if v is not None})

    shap_payload, _shap_artifacts, _ = build_shap_artifacts(
        output_dir=paths.output_dir,
        feature_names=pack.feature_names,
        tabular_data=pack.test,
        model_scores=risk_score,
        y_true=labels,
        positive_class_label="1",
        max_display=min(10, max(1, len(pack.feature_names) + 1)),
        max_samples=max(32, min(len(labels), 200)),
    )

    attention_summary = generate_attention_artifacts(
        model=model,
        dataset=test_ds,
        device=device,
        output_dir=paths.output_dir,
        sample_limit=min(4, len(test_ds)),
    )

    if shap_payload and shap_payload.get("available"):
        metrics["shap_method"] = shap_payload.get("method")
        if shap_payload.get("features"):
            metrics["top_global_feature"] = shap_payload["features"][0]["feature"]

    analysis_summary = {
        "survival": survival_payload,
        "global_feature_importance": shap_payload,
        "attention": attention_summary,
    }
    analysis_path = paths.output_dir / "analysis_summary.json"
    analysis_path.write_text(json.dumps(analysis_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[evaluate] metrics: {metrics_path}")
    print(f"[evaluate] predictions: {pred_path}")
    print(f"[evaluate] analysis: {analysis_path}")
    print(f"[evaluate] summary: {metrics}")


def cmd_report(config_path: Path) -> None:
    config = load_yaml(config_path)
    project_root = resolve_project_root(Path(__file__))
    paths = resolve_paths(config, project_root)

    metrics_path = paths.output_dir / "metrics.json"
    history_path = paths.output_dir / "history.json"
    analysis_path = paths.output_dir / "analysis_summary.json"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"entries": []}
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
    survival_payload = analysis.get("survival") or {}
    shap_payload = analysis.get("global_feature_importance") or {}
    attention_payload = analysis.get("attention") or {}

    lines = [
        "# SMuRF E2E Demo Report (Multi-Region + Risk Score)",
        "",
        "## 1) Output Heads",
        "",
        "- classification head: 类别预测",
        "- risk head: 连续风险分（smurf_score）",
        "",
        "## 2) Config",
        "",
        f"- fusion_strategy: `{config['model'].get('fusion_strategy')}`",
        f"- num_classes: `{config['model'].get('num_classes')}`",
        f"- epochs: `{config['training'].get('epochs')}`",
        f"- enable_survival: `{config['data'].get('enable_survival', True)}`",
        f"- survival_loss_weight: `{config['training'].get('survival_loss_weight', 0.3)}`",
        "",
        "## 3) Metrics (test split)",
        "",
    ]

    for k, v in metrics.items():
        lines.append(f"- {k}: **{v}**")

    if survival_payload:
        lines.extend(
            [
                "",
                "## 4) Survival Post-hoc",
                "",
                f"- c_index: **{survival_payload.get('c_index')}**",
                f"- cutoff: **{survival_payload.get('cutoff')}**",
                f"- cutoff_source: **{survival_payload.get('cutoff_source')}**",
                f"- sample_count: **{survival_payload.get('sample_count')}**",
                "",
                "![Kaplan-Meier](../visualizations/survival/kaplan_meier.png)",
                "",
                "![Risk Score Distribution](../visualizations/survival/risk_score_distribution.png)",
            ]
        )

    if shap_payload:
        lines.extend(
            [
                "",
                "## 5) Global Feature Importance",
                "",
                f"- method: **{shap_payload.get('method')}**",
                f"- feature_count: **{shap_payload.get('feature_count')}**",
                "",
                "![Importance Bar](../visualizations/shap/shap_bar.png)",
                "",
                "![Importance Beeswarm](../visualizations/shap/shap_beeswarm.png)",
            ]
        )

    if attention_payload:
        lines.extend(
            [
                "",
                "## 6) Attention Interpretability",
                "",
                f"- exported_items: **{len(attention_payload.get('items', []))}**",
            ]
        )
        for item in attention_payload.get("items", [])[:6]:
            image_rel = Path(item["image_path"]).relative_to(paths.reports_dir.parent)
            lines.extend(["", f"### {item['title']}", "", f"![{item['title']}]({image_rel.as_posix()})"])

    lines.extend(["", "## 7) Training History", ""])
    for row in history.get("entries", []):
        lines.append(
            f"- epoch {row['epoch']}: train_acc={row['train_acc']:.4f}, val_acc={row['val_acc']:.4f}, "
            f"train_loss={row['train_loss']:.4f}, val_loss={row['val_loss']:.4f}, "
            f"train_surv_loss={row['train_surv_loss']:.4f}, val_surv_loss={row['val_surv_loss']:.4f}"
        )

    report_path = paths.reports_dir / "smurf_e2e_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SMuRF E2E multi-region demo pipeline")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to config YAML",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-mock", help="Generate mock multi-region data")
    sub.add_parser("prepare-hipt", help="Offline-generate HIPT embeddings from pathology paths")
    sub.add_parser("train", help="Train model")

    eval_p = sub.add_parser("evaluate", help="Evaluate checkpoint on test split")
    eval_p.add_argument("--checkpoint", default=None, help="Checkpoint path")

    sub.add_parser("report", help="Generate markdown report")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    if args.command == "prepare-mock":
        cmd_prepare_mock(config_path)
    elif args.command == "prepare-hipt":
        cmd_prepare_hipt(config_path)
    elif args.command == "train":
        cmd_train(config_path)
    elif args.command == "evaluate":
        ckpt = Path(args.checkpoint).resolve() if args.checkpoint else None
        cmd_evaluate(config_path, ckpt)
    elif args.command == "report":
        cmd_report(config_path)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
