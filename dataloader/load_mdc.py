"""MDC NPY loader with optional native-resolution LR-HSI output."""

from __future__ import annotations

import json
import os

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class MDCDataset(Dataset):
    """Load the existing MDC train/val/test NPY split.

    Existing callers receive ``(LRHSI_up, HRMSI, HRHSI, LRMSI_up)``. Setting
    ``return_native_lr=True`` prepends the native LR-HSI produced before the
    bicubic upsampling step.
    """

    def __init__(self, root_dir, dataset_type=None, transform=None,
                 downscale_factor=4, de=True, return_native_lr=False):
        if dataset_type not in ("train", "val", "test"):
            raise ValueError('dataset_type must be in ["train", "val", "test"]')
        if downscale_factor <= 0:
            raise ValueError("downscale_factor must be positive")
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.downscale_factor = downscale_factor
        self.de = de  # Retained for backward-compatible construction.
        self.return_native_lr = return_native_lr
        self.dataset_info_path = os.path.join(root_dir, "dataset_info.json")
        with open(self.dataset_info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
        if dataset_type not in info:
            raise KeyError(f"Split {dataset_type!r} is absent from {self.dataset_info_path}")
        self.dataset_paths = info[dataset_type]

    def __len__(self):
        return len(self.dataset_paths)

    def __getitem__(self, idx):
        paths = self.dataset_paths[idx]
        hsi_path = os.path.join(self.root_dir, self.dataset_type, paths["hr_hsi"])
        msi_path = os.path.join(self.root_dir, self.dataset_type, paths["hr_msi"])
        hsi_hwc = np.load(hsi_path).astype(np.float32, copy=False)
        msi_hwc = np.load(msi_path).astype(np.float32, copy=False)
        if hsi_hwc.ndim != 3 or msi_hwc.ndim != 3:
            raise ValueError(f"MDC arrays must be HWC; HSI={hsi_hwc.shape}, MSI={msi_hwc.shape}")
        if hsi_hwc.shape[:2] != msi_hwc.shape[:2]:
            raise ValueError(f"MDC spatial mismatch: HSI={hsi_hwc.shape}, MSI={msi_hwc.shape}")
        hr_hsi = hsi_hwc.transpose(2, 0, 1)
        hr_msi = msi_hwc.transpose(2, 0, 1)

        native = self.synthesize_lr_hsi_native(hr_hsi, downscale_factor=self.downscale_factor)
        lr_up = self.upsample_lr_hsi(native, hr_hsi.shape[-2:])
        lrmsi_up = self.synthesize_lr_msi_gt(hr_msi, self.downscale_factor)
        tensors = [torch.from_numpy(value.copy()).float()
                   for value in (native, lr_up, hr_msi, hr_hsi, lrmsi_up)]
        if self.transform:
            tensors = [self.transform(value) for value in tensors]
        native_t, lr_up_t, msi_t, hsi_t, lrmsi_t = tensors
        if self.return_native_lr:
            return native_t, lr_up_t, msi_t, hsi_t, lrmsi_t
        return lr_up_t, msi_t, hsi_t, lrmsi_t

    @staticmethod
    def synthesize_lr_hsi_native(hr_hsi, sigma=2, downscale_factor=4):
        """7x7 Gaussian blur (sigma=2), then bicubic xK downsampling."""
        hsi_hwc = hr_hsi.transpose(1, 2, 0)
        blurred = cv.GaussianBlur(hsi_hwc, (7, 7), sigma)
        native = cv.resize(blurred, None, fx=1 / downscale_factor,
                           fy=1 / downscale_factor, interpolation=cv.INTER_CUBIC)
        if native.ndim == 2:
            native = native[:, :, None]
        return native.transpose(2, 0, 1).astype(np.float32)

    @staticmethod
    def upsample_lr_hsi(native, output_hw):
        height, width = output_hw
        up = cv.resize(native.transpose(1, 2, 0), (width, height),
                       interpolation=cv.INTER_CUBIC).clip(0, 1)
        if up.ndim == 2:
            up = up[:, :, None]
        return up.transpose(2, 0, 1).astype(np.float32)

    def synthesize_lr_hsi(self, hr_hsi, sigma=2, downscale_factor=4):
        """Backward-compatible interface returning LR-HSI on the HR grid."""
        native = self.synthesize_lr_hsi_native(hr_hsi, sigma, downscale_factor)
        return self.upsample_lr_hsi(native, hr_hsi.shape[-2:])

    @staticmethod
    def synthesize_lr_msi_gt(hr_msi, downscale_factor=4):
        msi_hwc = hr_msi.transpose(1, 2, 0)
        height, width = msi_hwc.shape[:2]
        small = cv.resize(msi_hwc, (width // downscale_factor, height // downscale_factor),
                          interpolation=cv.INTER_CUBIC)
        up = cv.resize(small, (width, height), interpolation=cv.INTER_CUBIC).clip(0, 1)
        if up.ndim == 2:
            up = up[:, :, None]
        return up.transpose(2, 0, 1).astype(np.float32)


if __name__ == "__main__":
    dataset = MDCDataset("/mnt/16T/data/FJL/MDC_New/", "train", downscale_factor=4)
    print([tuple(value.shape) for value in dataset[0]])
