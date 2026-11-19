# dataloader.py

import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import cv2 as cv


class PaviaDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.data_path = self.config["data_path"]
        self.patch_size = self.config.get("patch_size", 64)
        self.stride = self.config.get("stride", 16)
        self.augmentation = self.config.get("augmentation", True)
        self.downscale_factor = self.config.get("downscale_factor", 4)

        self.split = self.config.get("split", "train" if is_train else "val")
        self.gap = int(self.config.get("gap", 8))
        if self.split not in ("train", "val", "test"):
            raise ValueError("Pavia split must be 'train', 'val', or 'test'")
        if self.gap < 0:
            raise ValueError("Pavia split gap must be non-negative")
        self.is_train = self.split == "train"

        mat = scipy.io.loadmat(self.data_path)
        requested_key = self.config.get("data_key")
        candidate_keys = [requested_key] if requested_key else ["pavia", "paviaU"]
        data = None
        used_key = None
        for key in candidate_keys:
            if key and key in mat and isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
                data, used_key = mat[key], key
                break
        if data is None:
            arrays = [
                (key, value)
                for key, value in mat.items()
                if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 3
            ]
            if len(arrays) != 1:
                available = [key for key, _ in arrays]
                raise KeyError(
                    f"Cannot identify the Pavia HSI cube in {self.data_path}; "
                    f"3-D variables={available}. Set config['data_key'] explicitly."
                )
            used_key, data = arrays[0]
        data = np.asarray(data, dtype=np.float32)
        self.data_key = used_key

        # 归一化到 [0, 1]
        self.global_min = data.min()
        self.global_max = data.max()
        value_range = self.global_max - self.global_min
        if value_range <= 0:
            raise ValueError(f"Pavia cube {self.data_path} is constant")
        data = ((data - self.global_min) / value_range).astype(np.float32)

        # Paper split: test is the upper-left 512x128 block; validation is
        # the 128x128 block below it. Training uses the remaining image with
        # a spatial gap, preventing patch overlap and train/val leakage.
        if data.shape[0] < 640 or data.shape[1] < 128:
            raise ValueError(
                f"Pavia cube is too small for the paper split: {data.shape}; "
                "need at least 640x128 spatial pixels"
            )
        if self.split == "test":
            regions = [data[:512, :128]]
        elif self.split == "val":
            regions = [data[512:640, :128]]
        else:
            regions = []
            right_start = 128 + self.gap
            bottom_start = 640 + self.gap
            if right_start < data.shape[1]:
                regions.append(data[:, right_start:])
            if bottom_start < data.shape[0]:
                regions.append(data[bottom_start:, :128])
            if not regions:
                raise ValueError(f"No Pavia training area remains with gap={self.gap}")

        self.data = regions[0] if len(regions) == 1 else tuple(regions)
        self.patches = []
        for region in regions:
            self.patches.extend(self.create_patches(region))
        if not self.patches:
            raise ValueError(
                f"No {self.split} patches for patch_size={self.patch_size}, stride={self.stride}"
            )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        # ------------------ Ground Truth (GT) ------------------
        # HRHSI_gt 就是原始patch
        hrhsi_gt = np.array(self.patches[index], dtype=np.float32, copy=True)
        if self.is_train and self.augmentation:
            if random.random() > 0.5:
                hrhsi_gt = np.flip(hrhsi_gt, axis=0)
            if random.random() > 0.5:
                hrhsi_gt = np.flip(hrhsi_gt, axis=1)
            hrhsi_gt = np.rot90(hrhsi_gt, random.randint(0, 3), axes=(0, 1))
            hrhsi_gt = np.ascontiguousarray(hrhsi_gt)

        # ------------------ 模型输入 ------------------
        # 1. 生成高分辨率多光谱图像 (HRMSI)
        hrmsi = self.synthesize_msi(hrhsi_gt)
        # 2. 生成低分辨率高光谱图像 (LRHSI)
        lrhsi = self.synthesize_lr_hsi(hrhsi_gt, downscale_factor=self.downscale_factor)

        # ------------------ 辅助损失的GT ------------------
        # 1. 生成低分辨率多光谱图像 (LRMSI_gt)
        # 我们可以通过对HRMSI进行与HSI相同的降采样过程来模拟它
        lrmsi_gt = cv.resize(hrmsi,
                             (self.patch_size // self.downscale_factor, self.patch_size // self.downscale_factor),
                             interpolation=cv.INTER_CUBIC)
        lrmsi_gt = cv.resize(lrmsi_gt, (self.patch_size, self.patch_size), interpolation=cv.INTER_CUBIC)

        # 数据增强 (如果需要)
        # 转换为 PyTorch Tensors 并调整维度 (C, H, W)
        hrhsi_gt = torch.from_numpy(hrhsi_gt.transpose(2, 0, 1)).float()
        lrhsi = torch.from_numpy(lrhsi.transpose(2, 0, 1)).float()
        hrmsi = torch.from_numpy(hrmsi.transpose(2, 0, 1)).float()
        lrmsi_gt = torch.from_numpy(lrmsi_gt.transpose(2, 0, 1)).float()

        return lrhsi, hrmsi, hrhsi_gt, lrmsi_gt

    def create_patches(self, data):
        h, w, c = data.shape
        if h < self.patch_size or w < self.patch_size:
            return []
        return [
            data[y:y + self.patch_size, x:x + self.patch_size]
            for y in range(0, h - self.patch_size + 1, self.stride)
            for x in range(0, w - self.patch_size + 1, self.stride)
        ]

    def synthesize_msi(self, hsi):
        """
        模拟MSI，对HSI中的特定光谱段取平均
        Args:
        - hsi (numpy.ndarray): 原始高光谱图像
        - num_bands (int): 生成的 MSI 的波段数

        Returns:
        - numpy.ndarray: 模拟的 MSI
        """
        # B_bands 3-22 G_bands 19-41 R_bands 50-66 NIR_bands 80-102 在特定光谱段上进行平均
        split_bands = [hsi[:, :, 2:22], hsi[:, :, 18:41], hsi[:, :, 49:66], hsi[:, :, 79:]]
        # 计算每个子数组的平均值
        averaged_bands = [np.mean(band, axis=-1) for band in split_bands]
        # 将计算出的平均值堆叠成一个新的数组
        msi = np.stack(averaged_bands, axis=-1)
        return msi

    @staticmethod
    def synthesize_lr_hsi_native(hsi, sigma=2, downscale_factor=4):
        blurred = cv.GaussianBlur(hsi, (7, 7), sigma)
        native = cv.resize(blurred, dsize=None, fx=1 / downscale_factor,
                           fy=1 / downscale_factor, interpolation=cv.INTER_CUBIC)
        if native.ndim == 2:
            native = native[:, :, None]
        return native.astype(np.float32)

    def synthesize_lr_hsi(self, hsi, sigma=2, downscale_factor=4):
        h, w = hsi.shape[:2]
        native = self.synthesize_lr_hsi_native(hsi, sigma, downscale_factor)
        lr_hsi = cv.resize(native, (w, h), interpolation=cv.INTER_CUBIC)
        if lr_hsi.ndim == 2:
            lr_hsi = lr_hsi[:, :, None]
        return lr_hsi.clip(0, 1)
