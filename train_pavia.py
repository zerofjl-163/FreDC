# train_pavia_modified.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm
import numpy as np
import os
import argparse
from datetime import datetime

# --- 1. 修改导入路径 ---
# 确保从包含最终模型代码的文件导入
from net.network import GroupedDisentangleFusionNet
from net.samLoss import SAMLoss
from dataloader.load_pavia import PaviaDataset
from metrics.metric import psnr, ssim, sam, ergas
from utils.visualize_pavia import save_validation_results


def main(args):
    # --- 创建独立的实验结果保存路径 (保持不变) ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(args.save_dir, f"exp_{timestamp}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Results, logs, and checkpoints will be saved to: {save_path}")

    # --- 环境设置 (保持不变) ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 数据加载 (保持不变) ---
    print("Setting up data loaders...")
    train_config = {"data_path": args.data_path, "patch_size": args.patch_size, "stride": args.stride,
                    "downscale_factor": args.downscale_factor, "split": "train", "gap": args.gap}
    val_config = {"data_path": args.data_path, "patch_size": args.patch_size, "stride": args.patch_size,
                  "downscale_factor": args.downscale_factor, "split": "val", "gap": args.gap}

    train_dataset = PaviaDataset(train_config, is_train=True)
    val_dataset = PaviaDataset(val_config, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # --- 4. 模型初始化 (修改) ---
    print("Initializing model...")
    hsi_group_channels = [int(x) for x in args.hsi_group_channels.split(',')]
    print(f"HSI will be grouped into channels: {hsi_group_channels}")

    model = GroupedDisentangleFusionNet(
        hsi_group_channels=hsi_group_channels,
        msi_channels=args.msi_c,
        embed_dim=args.embed_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f} M")

    # --- 5. 优化器, 调度器, 损失函数 (保持不变) ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    l1_loss_fn = nn.L1Loss()
    sam_loss_fn = SAMLoss().to(device)

    # --- 6. 加载检查点 (修正逻辑) ---
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading model weights from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            # 尝试加载模型权重，允许部分加载或非严格匹配（以防万一）
            # 如果 checkpoint 直接是 state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Loaded model weights successfully from checkpoint['model_state_dict'].")
            # 如果 checkpoint 直接就是 state_dict (例如只保存了 model.state_dict())
            elif isinstance(checkpoint, dict) and not 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint, strict=False)
                print("Loaded model weights successfully from checkpoint (assumed state_dict).")
            else:
                print("Checkpoint format not recognized or model_state_dict not found.")
                print("Starting training from scratch.")

            # 不再加载 optimizer, scheduler, start_epoch, best_psnr

        except Exception as e:
            print(f"Could not load model weights: {e}")
            print("Starting training from scratch.")
            # start_epoch = 0 # 已在上面初始化
            # best_psnr = 0.0

    # --- 7. 训练与验证循环 (逻辑保持不变) ---
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Training]")
        epoch_main_loss, epoch_aux_loss, epoch_cos_loss, epoch_sam_loss, epoch_total_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        for lrhsi, hrmsi, hrhsi_gt, lrmsi_gt in progress_bar:
            lrhsi, hrmsi, hrhsi_gt, lrmsi_gt = lrhsi.to(device), hrmsi.to(device), hrhsi_gt.to(device), lrmsi_gt.to(device)
            optimizer.zero_grad()

            # 模型输出与之前的训练代码兼容
            pred_hrhsi, pred_lrmsi, hsi_ll_r, hsi_ll_u, msi_ll_r, msi_ll_u = model(lrhsi, hrmsi)
            # pred_hrhsi = pred_hrhsi.clamp(0, 1)
            # --- 计算各类损失 ---
            main_loss = l1_loss_fn(pred_hrhsi, hrhsi_gt)
            aux_loss = l1_loss_fn(pred_lrmsi, lrmsi_gt)

            # 计算解缠的余弦相似度损失 (B, C, H, W -> B, N, C)
            # 确保在计算相似度前处理可能的 NaN 或 Inf (虽然理论上不应出现)
            hsi_ll_r_flat = rearrange(hsi_ll_r, 'b c h w -> b (h w) c')
            hsi_ll_u_flat = rearrange(hsi_ll_u, 'b c h w -> b (h w) c')
            msi_ll_r_flat = rearrange(msi_ll_r, 'b c h w -> b (h w) c')
            msi_ll_u_flat = rearrange(msi_ll_u, 'b c h w -> b (h w) c')

            # 添加 eps 防止除零
            hsi_cos_sim = F.cosine_similarity(hsi_ll_r_flat, hsi_ll_u_flat, dim=2, eps=1e-8).mean()
            msi_cos_sim = F.cosine_similarity(msi_ll_r_flat, msi_ll_u_flat, dim=2, eps=1e-8).mean()

            # 如果相似度计算结果是 NaN (例如向量全零), 则设为 0
            hsi_cos_sim = torch.nan_to_num(hsi_cos_sim, nan=0.0)
            msi_cos_sim = torch.nan_to_num(msi_cos_sim, nan=0.0)

            cos_loss = (hsi_cos_sim + msi_cos_sim) / 2.0

            # --- 计算分组 SAM 损失 ---
            pred_hrhsi_groups = torch.split(pred_hrhsi, hsi_group_channels, dim=1)
            hrhsi_gt_groups = torch.split(hrhsi_gt, hsi_group_channels, dim=1)
            sam_losses = []
            for pred_group, gt_group in zip(pred_hrhsi_groups, hrhsi_gt_groups):
                sam_losses.append(sam_loss_fn(pred_group, gt_group))
            sam_loss = torch.mean(torch.stack(sam_losses))
            sam_loss = torch.nan_to_num(sam_loss, nan=0.0) # 处理可能的 NaN
            # sam_loss = sam_loss_fn(pred_hrhsi, hrhsi_gt)
            # --- 计算总损失 ---
            total_loss = main_loss + \
                         args.lambda_aux * aux_loss + \
                         args.lambda_cos * cos_loss + \
                         args.lambda_sam * sam_loss

            total_loss.backward()
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 记录 batch 损失
            epoch_main_loss += main_loss.item()
            epoch_aux_loss += aux_loss.item()
            epoch_cos_loss += cos_loss.item()
            epoch_sam_loss += sam_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                main=f"{main_loss.item():.4f}",
                aux=f"{aux_loss.item():.4f}",
                cos=f"{cos_loss.item():.4f}",
                sam=f"{sam_loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}"
            )

        scheduler.step()

        # 打印 epoch 平均损失
        print(f"\nEpoch {epoch + 1} Training Avg Loss: "
              f"Total={epoch_total_loss / num_batches:.4f}, "
              f"Main={epoch_main_loss / num_batches:.4f}, "
              f"Aux={epoch_aux_loss / num_batches:.4f}, "
              f"Cos={epoch_cos_loss / num_batches:.4f}, "
              f"SAM={epoch_sam_loss / num_batches:.4f}")

        # 验证 (逻辑保持不变)
        model.eval()
        val_psnr, val_ssim, val_sam, val_ergas = [], [], [], []
        with torch.no_grad():
            for i, (lrhsi, hrmsi, hrhsi_gt, lrmsi_gt) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Validation]")):
                lrhsi, hrmsi, hrhsi_gt = lrhsi.to(device), hrmsi.to(device), hrhsi_gt.to(device)
                # 模型输出与之前的验证代码兼容
                pred_hrhsi, _, _, _, _, _ = model(lrhsi, hrmsi)
                pred, gt = pred_hrhsi.clamp(0, 1), hrhsi_gt
                # 计算指标
                val_psnr.append(psnr(pred, gt))
                val_ssim.append(ssim(pred, gt))
                val_sam.append(sam(pred, gt))
                val_ergas.append(ergas(gt, pred))
                # 可视化第一个 batch
                if i == 0 and epoch % args.vis_every == 0: # 添加可视化频率控制
                    vis_path = os.path.join(save_path, "visualizations")
                    if not os.path.exists(vis_path): os.makedirs(vis_path)
                    save_validation_results(epoch + 1, lrhsi, hrmsi, pred_hrhsi, hrhsi_gt, save_dir=vis_path)

        avg_psnr, avg_ssim, avg_sam, avg_ergas = np.mean(val_psnr), np.mean(val_ssim), np.mean(val_sam), np.mean(val_ergas)
        print(
            f"--- Epoch {epoch + 1} Validation --- | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}")

        # 保存检查点 (逻辑保持不变)
        checkpoint_path = os.path.join(save_path, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'args': args # 保存命令行参数以便复现
        }, checkpoint_path)

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with PSNR: {best_psnr:.4f} to {best_model_path}\n")

    print("Training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train HSI Disentanglement and Fusion Network (Modified).")

    # --- 路径和实验管理参数 ---
    parser.add_argument('--data_path', type=str, default="E:/PF-MAWN/datasets/Pavia.mat",
                        help="Path to the Pavia dataset (.mat file).")
    parser.add_argument('--save_dir', type=str, default="./experiments",
                        help="Directory to save experiment logs and checkpoints.")
    parser.add_argument('--resume', type=str, default=None, # 默认为 None
                        help="Path to the checkpoint to resume training from.")

    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=3e-4, help="Initial learning rate.")

    # --- 数据和模型参数 ---
    parser.add_argument('--patch_size', type=int, default=64, help="Size of the image patches for training.")
    parser.add_argument('--stride', type=int, default=32, help="Stride for creating patches during training.")
    parser.add_argument('--gap', type=int, default=8, help="Spatial gap between Pavia train and val/test regions.")
    parser.add_argument('--downscale_factor', type=int, default=4, help="Downscale factor for LRHSI generation.")
    parser.add_argument('--msi_c', type=int, default=4, help="Number of channels for the MSI image.")
    parser.add_argument('--hsi_group_channels', type=str, default="35,35,32",
                        help="Comma-separated list of HSI channels for each subnet.")
    parser.add_argument('--embed_dim', type=int, default=48,
                        help="Embedding dimension of the model internal features.")

    # --- 损失权重 ---
    parser.add_argument('--lambda_aux', type=float, default=0, help="Weight for the auxiliary reconstruction loss.")
    parser.add_argument('--lambda_cos', type=float, default=0,
                        help="Weight for the cosine similarity disentanglement loss.")
    parser.add_argument('--lambda_sam', type=float, default=0, help="Weight for the SAM loss.")

    # --- 系统参数 ---
    parser.add_argument('--num_workers', type=int, default=0, help="Number of worker processes for data loading.")
    parser.add_argument('--vis_every', type=int, default=5, help="Frequency (epochs) for saving validation visualizations.") # 添加可视化频率参数


    args = parser.parse_args()
    main(args)
