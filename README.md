I’m inspecting the entry points and package structure so the README has accurate commands and dataset assumptions.I’ve identified the repository as a PyTorch hyperspectral fusion implementation with separate MDC, Pavia, and Harvard workflows. I’m narrowing the README to documented commands and structure, and will avoid presenting unverified dependencies or dataset downloads.# FreDC

A PyTorch implementation for hyperspectral image (HSI) super-resolution and HSI/MSI fusion. FreDC uses grouped spectral processing, stationary wavelet decomposition, disentanglement encoders, and lightweight cross-attention to reconstruct high-resolution hyperspectral images.

The repository includes training and evaluation pipelines for MDC, Harvard, and Pavia datasets.

## Features

- Grouped hyperspectral channel processing
- Stationary wavelet-based low/high-frequency decomposition
- Disentangled HSI and MSI feature representations
- Memory-efficient normalized linear cross-attention
- Auxiliary low-resolution MSI reconstruction
- L1, cosine disentanglement, SSIM, and spectral angle losses
- WandB experiment logging
- Validation visualization and per-image metrics
- Deterministic training support for MDC

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

## Model Overview

`GroupedDisentangleFusionNet` splits the input HSI into configurable spectral groups and processes each group with a dedicated subnet.

The main reconstruction path:

1. Applies the stationary wavelet transform to LR-HSI and HR-MSI.
2. Encodes low- and high-frequency features independently.
3. Separates each representation into relevant and irrelevant features.
4. Fuses MSI features with HSI features using lightweight cross-attention.
5. Reconstructs the spectral subgroups.
6. Combines the subgroups into the final HR-HSI output.

An auxiliary path reconstructs LR-MSI and helps regularize feature disentanglement.

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

Example installation:

```bash
pip install torch numpy opencv-python einops tqdm matplotlib scipy wandb
```

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

### Harvard

Point `--data_path` to the directory containing the Harvard NPY dataset.

### Pavia

Point `--data_path` to the Pavia `.mat` file.

## Training

### MDC

```bash
python train_mdc.py \
  --data_path /path/to/MDC_New \
  --save_dir ./experiments_mdc \
  --epochs 500 \
  --batch_size 16 \
  --lr 5e-4 \
  --downscale_factor 4 \
  --hsi_group_channels 25,25 \
  --msi_c 3 \
  --device auto
```

The MDC training script records:

- `training_config.json`
- `training_history.csv`
- `best_model.pth`
- `last_checkpoint.pth`
- Validation visualizations
- Best-model metrics

Resume from a saved model or checkpoint:

```bash
python train_mdc.py \
  --data_path /path/to/MDC_New \
  --resume /path/to/checkpoint.pth
```

### Harvard

```bash
python train_harvard.py \
  --data_path /path/to/HarvardDataset/npy \
  --save_dir ./experiments_harvard \
  --epochs 100 \
  --batch_size 2 \
  --downscale_factor 16 \
  --hsi_group_channels 17,14 \
  --msi_c 3
```

### Pavia

```bash
python train_pavia.py \
  --data_path /path/to/Pavia.mat \
  --save_dir ./experiments_pavia \
  --epochs 100 \
  --batch_size 4 \
  --downscale_factor 4 \
  --patch_size 64 \
  --stride 32 \
  --hsi_group_channels 35,35,32 \
  --msi_c 4
``>

## Evaluation

Evaluate an MDC model on the full test images:

```bash
python test_mdc.py \
  --data_path /path/to/MDC_New \
  --checkpoint_path /path/to/best_model.pth \
  --downscale_factor 4 \
  --hsi_group_channels 25,25 \
  --msi_c 3
```

The other dataset-specific test scripts follow a similar pattern:

```bash
python test_harvard.py --checkpoint_path /path/to/best_model.pth
python test_pavia.py --checkpoint_path /path/to/best_model.pth
``>

Check each script's `--help` output for dataset-specific options.

## Results

Test results include:

- PSNR
- SSIM
- Spectral Angle Mapper (SAM)
- ERGAS
- Aggregate mean and standard deviation
- Per-image metrics in CSV format
- Predicted HSI, ground-truth HSI, MSI input, LR-HSI input, and error maps

MDC test outputs are saved beside the model checkpoint in a directory named `test_results_full_images_mdc`.

## WandB

WandB is disabled by default. Enable it during training with:

```bash
python train_mdc.py --wandb_mode online
```

Supported modes are `online`, `offline`, and `disabled`.

## Citation

If you use this repository in your research, cite the associated paper or add the repository URL:

```text
https://github.com/zerofjl-163/FreDC
```

## License

No license is currently specified for this repository. Please contact the repository owner before commercial or redistribution use.
