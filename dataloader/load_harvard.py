# load_harvard.py

import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import scipy.io
from matplotlib import pyplot as plt
from utils.random_transform import random_transform


def synthesize_lr_hsi(hsi, sigma=2, downscale_factor=8):
    h, w, c = hsi.shape
    blurred = cv.GaussianBlur(hsi, (7, 7), sigma)
    lr_hsi_small = cv.resize(blurred, dsize=None, fx=1 / downscale_factor, fy=1 / downscale_factor,
                             interpolation=cv.INTER_CUBIC)
    lr_hsi_up = cv.resize(lr_hsi_small, (w, h), interpolation=cv.INTER_CUBIC).clip(0, 1)
    return lr_hsi_up


def synthesize_lr_msi(msi, downscale_factor=8):
    """
    生成低分辨率多光谱图像 (LRMSI)，作为辅助损失的GT。
    过程与Pavia数据加载器中的lrmsi_gt生成过程对齐。
    """
    h, w, c = msi.shape
    # 向下采样
    lr_msi_small = cv.resize(msi,
                             (w // downscale_factor, h // downscale_factor),
                             interpolation=cv.INTER_CUBIC)
    # 向上采样
    lr_msi_up = cv.resize(lr_msi_small, (w, h), interpolation=cv.INTER_CUBIC)
    return lr_msi_up.clip(0, 1)


class HarvardDataset(Dataset):
    def __init__(self, root_dir, dataset_type, downsacle_factor=8, transform=None):
        assert dataset_type in ['train', 'val', 'test', 'test_patch'], 'Invalid dataset type'
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.downscale_factor = downsacle_factor

        if self.dataset_type == 'train':
            self.file_paths = [os.path.join(root_dir, 'train', f) for f in os.listdir(os.path.join(root_dir, 'train'))
                               if os.path.isdir(os.path.join(root_dir, 'train', f))]
        elif self.dataset_type == 'val':
            self.file_paths = [os.path.join(root_dir, 'val', f) for f in os.listdir(os.path.join(root_dir, 'val'))
                               if os.path.isdir(os.path.join(root_dir, 'val', f))]
        elif self.dataset_type == 'test_patch':
            self.file_paths = [os.path.join(root_dir, 'test_patch', f) for f in
                               os.listdir(os.path.join(root_dir, 'test_patch'))
                               if os.path.isdir(os.path.join(root_dir, 'test_patch', f))]
        else:  # 'test'
            self.file_paths = [os.path.join(root_dir, 'test', f) for f in os.listdir(os.path.join(root_dir, 'test'))
                               if f.endswith('.npy')]
            # Optional: filter test files if needed
            # indices_to_keep = {16}
            # self.file_paths = [
            #     os.path.join(root_dir, 'test', f)
            #     for i, f in enumerate(os.listdir(os.path.join(root_dir, 'test')))
            #     if f.endswith('.npy') and i in indices_to_keep
            # ]

        self.reps = scipy.io.loadmat(os.path.join(root_dir, "resp.mat"))['resp']
        self.reps = np.transpose(self.reps, (1, 0))

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.file_paths) * 80
        elif self.dataset_type == 'val':
            return len(self.file_paths) * 10
        elif self.dataset_type == 'test_patch':
            return len(self.file_paths) * 20
        else:
            return len(self.file_paths)

    def __getitem__(self, idx):
        # --- 核心修改：所有分支都需要返回4个张量 ---
        if self.dataset_type == 'train' or self.dataset_type == 'val' or self.dataset_type == 'test_patch':
            if self.dataset_type == 'train':
                file_idx, patch_idx = idx // 80, idx % 80
            elif self.dataset_type == 'val':
                file_idx, patch_idx = idx // 10, idx % 10
            else:  # test_patch
                file_idx, patch_idx = idx // 20, idx % 20

            patch_folder = self.file_paths[file_idx]
            ref_patch_path = os.path.join(patch_folder, f'{patch_idx}.npy')
            ref_patch = np.load(ref_patch_path).astype(np.float32)

            hr_msi = np.tensordot(ref_patch, self.reps, (-1, 0))
            lr_hsi = synthesize_lr_hsi(ref_patch, downscale_factor=self.downscale_factor)
            # --- 新增 lrmsi_gt ---
            lrmsi_gt = synthesize_lr_msi(hr_msi, downscale_factor=self.downscale_factor)

            if self.transform:
                ref_patch, hr_msi, lr_hsi, lrmsi_gt = self.transform(ref_patch, hr_msi, lr_hsi, lrmsi_gt)
                # 注意: 变换通常不应用于lrmsi_gt，因为它应与原始hr_msi对应
                # 如果transform包含翻转/旋转，lrmsi_gt也应做同样变换，这里暂不处理

            return (torch.from_numpy(lr_hsi.transpose(2, 0, 1)).float(),
                    torch.from_numpy(hr_msi.transpose(2, 0, 1)).float(),
                    torch.from_numpy(ref_patch.transpose(2, 0, 1)).float(),
                    torch.from_numpy(lrmsi_gt.transpose(2, 0, 1)).float())
        else:  # 'test'
            ref_patch = np.load(self.file_paths[idx]).astype(np.float32)

            hr_msi = np.tensordot(ref_patch, self.reps, (-1, 0))
            lr_hsi = synthesize_lr_hsi(ref_patch, downscale_factor=self.downscale_factor)
            # --- 新增 lrmsi_gt ---
            lrmsi_gt = synthesize_lr_msi(hr_msi, downscale_factor=self.downscale_factor)

            return (torch.from_numpy(lr_hsi.transpose(2, 0, 1)).float(),
                    torch.from_numpy(hr_msi.transpose(2, 0, 1)).float(),
                    torch.from_numpy(ref_patch.transpose(2, 0, 1)).float(),
                    torch.from_numpy(lrmsi_gt.transpose(2, 0, 1)).float())


def visualize(img):
    # 确保图像数据范围在[0, 1]用于显示
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # 测试新的数据加载器
    dataset = HarvardDataset(root_dir='F:/HarvardDataset/npy', dataset_type='train', downsacle_factor=8)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 获取一个样本并检查维度
    lr_hsi, hr_msi, ref, lrmsi_gt = next(iter(dataloader))
    print(f"LR-HSI shape: {lr_hsi.shape}")
    print(f"HR-MSI shape: {hr_msi.shape}")
    print(f"Reference shape: {ref.shape}")
    print(f"LR-MSI GT shape: {lrmsi_gt.shape}")

    # 可视化一个样本
    visualize(hr_msi[0].cpu().numpy().transpose(1, 2, 0))
    visualize(lrmsi_gt[0].cpu().numpy().transpose(1, 2, 0))