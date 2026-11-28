A PyTorch implementation for hyperspectral image (HSI) super-resolution and HSI/MSI fusion. FreDC uses grouped spectral processing, stationary wavelet decomposition, disentanglement encoders, and lightweight cross-attention to reconstruct high-resolution hyperspectral images.

## Repository Structure

```text
ablation/          Hyperparameter experiments
dataloader/        Harvard, MDC, and Pavia dataset loaders
metrics/           PSNR, SSIM, SAM, ERGAS, RMSE, and QNR metrics
net/               Model, wavelet, and loss implementations
utils/             Visualization and data transformation utilities
train_*.py         Dataset-specific training scripts
test_*.py          Dataset-specific evaluation scripts
```

## Requirements

- Python 3.8+
- PyTorch
- CUDA-capable GPU recommended
- NumPy
- OpenCV
- einops
- tqdm
- Matplotlib
- WandB
- SciPy, required for the Pavia and Harvard loaders


## Dataset Setup

The scripts expect dataset-specific directory structures or files.

### MDC

Prepare `dataset_info.json` under the dataset root. It should contain `train`, `val`, and `test` entries referencing the corresponding HSI and MSI NPY files.

```text
MDC_New/
├── dataset_info.json
├── train/
├── val/
└── test/
```

## Citation

If you use this repository in your research, cite the associated paper or add the repository URL:

```text
https://github.com/zerofjl-163/FreDC
```

## License

No license is currently specified for this repository. Please contact the repository owner before commercial or redistribution use.
