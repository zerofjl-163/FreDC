# dataloader/load_pavia_patches.py

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class PaviaPatchesDataset(Dataset):
    """
    一个专门用于 t-SNE 分析的数据集。
    (新版本) 它从整个 Pavia University 影像中随机采样图像块。
    """

    def __init__(self, data_path: str, patch_size: int = 32, num_patches: int = 1000):
        """
        参数:
        data_path (str): 指向 .mat 文件的路径。
        patch_size (int): 采样的图像块大小。
        num_patches (int): 要从整个影像中随机采样的图像块总数。
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches  # 注意：此参数现在代表总采样数

        # 加载并预处理数据
        try:
            data = sio.loadmat(data_path)
            self.hsi = data.get('paviaU', None)
            if self.hsi is None:
                keys = [k for k in data if not k.startswith('__')]
                if keys:
                    self.hsi = data[keys[0]]
                else:
                    raise KeyError("在 .mat 文件中找不到高光谱数据。")

            self.hsi = self.hsi.astype(np.float32)
            self.hsi = (self.hsi - self.hsi.min()) / (self.hsi.max() - self.hsi.min())
            self.h, self.w, self.c = self.hsi.shape
            print(f"成功加载 Pavia 数据，尺寸: ({self.h}, {self.w}, {self.c})")

        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {data_path}")
        except Exception as e:
            raise RuntimeError(f"加载或处理数据时出错: {e}")

        # --- 核心修改：移除基于类别的 ROI，直接创建随机图像块 ---
        self.patches = []
        self._create_random_patches()

    def _create_random_patches(self):
        """从整个影像中随机采样图像块"""
        print(f"正在从整个影像中随机采样 {self.num_patches} 个图像块...")

        while len(self.patches) < self.num_patches:
            # 随机选择一个有效的左上角坐标
            r = np.random.randint(0, self.h - self.patch_size)
            c = np.random.randint(0, self.w - self.patch_size)

            patch = self.hsi[r:r + self.patch_size, c:c + self.patch_size, :]

            # (可选) 检查图像块是否几乎为空 (背景)，如果是则跳过
            # 这可以避免 t-SNE 图中出现大量无意义的背景点
            if patch.sum() < 1e-6:
                continue

            self.patches.append(patch)

        print(f"总共创建了 {len(self.patches)} 个随机图像块用于分析。")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]

        # 因为是随机采样，不再有具体的类别标签。
        # 我们返回一个统一的标签，以便在 t-SNE 图中将所有点视为一类。
        label = "Random Patch"

        # 将维度从 (H, W, C) 转换为 (C, H, W)
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)

        return patch_tensor, label