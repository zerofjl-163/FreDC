# dataloader/load_pavia_test.py

import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv


class PaviaTestDataset(Dataset):
    """
    Pavia数据集的测试加载器。
    与训练/验证不同，此加载器不将测试图像切分成块(patch)，
    而是直接加载完整的测试图像进行评估。
    """

    def __init__(self, data_path, downscale_factor=4):
        self.data_path = data_path
        self.downscale_factor = downscale_factor

        # 加载完整数据
        data = scipy.io.loadmat(self.data_path)['pavia']

        # 归一化到 [0, 1]
        global_min = data.min()
        global_max = data.max()
        data = (data - global_min) / (global_max - global_min)

        # 定义测试集区域 (不切分)
        self.hrhsi_gt = data[:512, :128]

        # 获取图像尺寸
        h, w, _ = self.hrhsi_gt.shape
        self.height = h
        self.width = w

    def __len__(self):
        # 数据集只包含一张完整的测试图像
        return 1

    def __getitem__(self, index):
        # ------------------ Ground Truth (GT) ------------------
        hrhsi_gt_np = self.hrhsi_gt

        # ------------------ 模型输入 ------------------
        # 1. 生成高分辨率多光谱图像 (HRMSI)
        hrmsi_np = self.synthesize_msi(hrhsi_gt_np)
        # 2. 生成低分辨率高光谱图像 (LRHSI)
        lrhsi_np = self.synthesize_lr_hsi(hrhsi_gt_np, downscale_factor=self.downscale_factor)

        # 转换为 PyTorch Tensors 并调整维度 (C, H, W)
        hrhsi_gt = torch.from_numpy(hrhsi_gt_np.transpose(2, 0, 1)).float()
        lrhsi = torch.from_numpy(lrhsi_np.transpose(2, 0, 1)).float()
        hrmsi = torch.from_numpy(hrmsi_np.transpose(2, 0, 1)).float()

        # 返回一个元组，包含所有需要的数据
        return lrhsi, hrmsi, hrhsi_gt

    def synthesize_msi(self, hsi):
        """
        模拟MSI，对HSI中的特定光谱段取平均。
        """
        # B_bands 3-22, G_bands 19-41, R_bands 50-66, NIR_bands 80-102
        split_bands = [hsi[:, :, 2:22], hsi[:, :, 18:41], hsi[:, :, 49:66], hsi[:, :, 79:]]
        averaged_bands = [np.mean(band, axis=-1) for band in split_bands]
        msi = np.stack(averaged_bands, axis=-1)
        return msi

    def synthesize_lr_hsi(self, hsi, sigma=2, downscale_factor=4):
        """
        模拟LR-HSI。
        """
        h, w, c = hsi.shape
        blurred = cv.GaussianBlur(hsi, (7, 7), sigma)
        lr_hsi_small = cv.resize(blurred, dsize=None, fx=1 / downscale_factor, fy=1 / downscale_factor,
                                 interpolation=cv.INTER_CUBIC)
        lr_hsi = cv.resize(lr_hsi_small, (w, h), interpolation=cv.INTER_CUBIC)
        return lr_hsi.clip(0, 1)