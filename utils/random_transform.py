import numpy as np
import random

def random_transform(ref_patch, hr_msi, lr_hsi, lr_msi):
    if random.random() > 0.5:
        ref_patch = np.flip(ref_patch, axis=0).copy()
        hr_msi = np.flip(hr_msi, axis=0).copy()
        lr_hsi = np.flip(lr_hsi, axis=0).copy()
        lr_msi = np.flip(lr_msi, axis=0).copy()

    if random.random() > 0.5:
        ref_patch = np.flip(ref_patch, axis=1).copy()
        hr_msi = np.flip(hr_msi, axis=1).copy()
        lr_hsi = np.flip(lr_hsi, axis=1).copy()
        lr_msi = np.flip(lr_msi, axis=1).copy()

    k = random.randint(0, 3)
    ref_patch = np.rot90(ref_patch, k=k, axes=(0, 1)).copy()
    hr_msi = np.rot90(hr_msi, k=k, axes=(0, 1)).copy()
    lr_hsi = np.rot90(lr_hsi, k=k, axes=(0, 1)).copy()
    lr_msi = np.rot90(lr_msi, k=k, axes=(0, 1)).copy()

    return ref_patch, hr_msi, lr_hsi, lr_msi
