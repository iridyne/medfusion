#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    source_root = Path('/home/yixian/Research/MedML/medical_demo_data/elbow_multimodal')
    csv_path = source_root / 'original_data.csv'

    out_root = Path('demo/smurf_e2e/data/elbow_dryrun').resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    modality_dirs = {
        'region1_ct': out_root / 'region1_ct',
        'region1_pathology': out_root / 'region1_pathology',
        'region2_ct': out_root / 'region2_ct',
        'region2_pathology': out_root / 'region2_pathology',
    }
    for d in modality_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    img_dirs = [
        source_root / 'imgs' / 'cropped_748' / 'coronal',
        source_root / 'imgs' / 'manual_preprocessed' / 'coronal',
    ]

    # build image index by ID (prefer cropped_748)
    image_map: dict[str, Path] = {}
    for d in img_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob('*.jpg')):
            image_map.setdefault(p.stem, p)

    df = pd.read_csv(csv_path)
    df['id_str'] = df['ID号'].astype(str).str.replace('.0', '', regex=False)
    df = df[df['id_str'].isin(image_map.keys())].copy().reset_index(drop=True)

    # keep a manageable subset for quick dry-run
    max_samples = 120
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        pid = str(row['id_str'])
        img_path = image_map[pid]

        img = Image.open(img_path).convert('L').resize((64, 64))
        arr = np.asarray(img, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)

        # 通用序列（兼容旧流程）
        def save_seq(modality: str, boosts: list[float]) -> str:
            seq: list[str] = []
            for j, b in enumerate(boosts):
                frame = arr.copy()
                if b != 0:
                    h, w = frame.shape
                    frame[h // 4 : h // 2, w // 4 : w // 2] += b
                frame += np.random.normal(0, 0.02, size=frame.shape).astype(np.float32)
                out = modality_dirs[modality] / f'{pid}_{j:02d}.npy'
                np.save(out, frame.astype(np.float32))
                seq.append(str(out.relative_to(Path.cwd())))
            return '|'.join(seq)

        # 单CT分支：真实 patch 序列做 MIL（整图+四个象限）
        def save_ct_patch_seq() -> str:
            h, w = arr.shape
            patch_boxes = [
                (0, 0, h, w),
                (0, 0, h // 2, w // 2),
                (0, w // 2, h // 2, w),
                (h // 2, 0, h, w // 2),
                (h // 2, w // 2, h, w),
            ]
            seq: list[str] = []
            for j, (r0, c0, r1, c1) in enumerate(patch_boxes):
                patch = arr[r0:r1, c0:c1].copy()
                patch += np.random.normal(0, 0.01, size=patch.shape).astype(np.float32)
                out = modality_dirs['region1_ct'] / f'{pid}_mil_{j:02d}.npy'
                np.save(out, patch.astype(np.float32))
                seq.append(str(out.relative_to(Path.cwd())))
            return '|'.join(seq)

        label = int(row['是否原发（原发=0，继发=1）(因变量Y)']) if not pd.isna(row['是否原发（原发=0，继发=1）(因变量Y)']) else 0
        age = float(row['年龄']) if not pd.isna(row['年龄']) else 55.0
        bmi = float(row['腋隐窝宽度（mm）']) if '腋隐窝宽度（mm）' in row and not pd.isna(row['腋隐窝宽度（mm）']) else 22.0
        crp = float(row['腋囊厚度']) if '腋囊厚度' in row and not pd.isna(row['腋囊厚度']) else 2.0

        sex_raw = int(row['性别（0=男，1=女）']) if not pd.isna(row['性别（0=男，1=女）']) else 0
        sex = 'F' if sex_raw == 1 else 'M'
        smoke_raw = int(row['喙突下滑囊炎']) if '喙突下滑囊炎' in row and not pd.isna(row['喙突下滑囊炎']) else 0
        smoking = 'yes' if smoke_raw == 1 else 'no'

        r = random.random()
        split = 'train' if r < 0.7 else ('val' if r < 0.85 else 'test')

        region1_ct_paths = save_ct_patch_seq()
        rows.append(
            {
                'patient_id': pid,
                'age': age,
                'bmi': bmi,
                'crp': crp,
                'sex': sex,
                'smoking': smoking,
                'label': label,
                'split': split,
                # survival is disabled in dry-run config, keep placeholders
                'survival_time': float(max(1.0, np.random.exponential(18.0 if label == 1 else 30.0))),
                'event': float(np.random.binomial(1, 0.7 if label == 1 else 0.4)),
                'region1_ct_paths': region1_ct_paths,
                'region1_pathology_paths': save_seq('region1_pathology', [0.10, 0.02]),
                'region2_ct_paths': save_seq('region2_ct', [0.00, 0.06]),
                'region2_pathology_paths': save_seq('region2_pathology', [0.05, -0.02, 0.01]),
                'ct_paths': region1_ct_paths,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = out_root / 'metadata.csv'
    out_df.to_csv(out_csv, index=False)

    print(f'[build-elbow-dryrun] rows={len(out_df)} saved={out_csv}')
    print('[build-elbow-dryrun] split counts:', out_df['split'].value_counts().to_dict())
    print('[build-elbow-dryrun] label counts:', out_df['label'].value_counts().to_dict())


if __name__ == '__main__':
    main()
