"""Minimal 2D slice dataset read from one folder per split:

    <split_dir>/images.npy    float [N, C, H, W]   (already normalised; H, W divisible by 16)
    <split_dir>/masks.npy     int   [N, H, W]      labels 0..K-1 (0 = background)
    <split_dir>/patients.txt  N lines, the patient id of each slice

Calibration and evaluation are patient-level, so every slice must carry its patient id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class SliceDataset:
    def __init__(self, split_dir: str | Path, in_ram: bool = True):
        d = Path(split_dir)
        mode = None if in_ram else "r"
        self.images = np.load(d / "images.npy", mmap_mode=mode)
        self.masks = np.load(d / "masks.npy", mmap_mode=mode)
        self.patients = (d / "patients.txt").read_text().split()
        if not (len(self.images) == len(self.masks) == len(self.patients)):
            raise ValueError(f"{d}: images, masks and patients.txt have different lengths")

    def __len__(self) -> int:
        return len(self.patients)

    def get_batch(self, idxs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        idxs = np.asarray(idxs)  # order kept: the first rows of a J-CRT batch are D_cal
        img = torch.from_numpy(np.asarray(self.images[idxs], dtype=np.float32))
        msk = torch.from_numpy(np.asarray(self.masks[idxs], dtype=np.int64))
        return img, msk

    def patient_slices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, p in enumerate(self.patients):
            out.setdefault(p, []).append(i)
        return out

    def slice_classes(self) -> list[set[int]]:
        return [set(np.unique(np.asarray(m)).tolist()) - {0} for m in self.masks]
