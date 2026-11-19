# train_harvard.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import os
import argparse
from datetime import datetime
import wandb
import matplotlib.pyplot as plt

# --- 核心修改: 导入新的模型 ---
from net.network import GroupedDisentangleFusionNet
from net.samLoss import SAMLoss
from dataloader.load_harvard import HarvardDataset
from metrics.metric import psnr, ssim, sam, ergas
from utils.random_transform import random_transform

# 设置可复现的随机种子
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)

# --- 新增：内置的可微分SSIM损失函数 ---
class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    Adapted from https://github.com/po-hsun-chen/pytorch-ssim
    """
    def __init__(self, window_size=11, size_average=True, channel=31):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return nn.Parameter(window, requires_grad=False)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# --- 内置的可视化函数，专为Harvard数据集定制 ---
def save_harvard_validation_results(epoch, lrhsi, hrmsi, pred_hrhsi, gt_hrhsi, save_dir):
    """
    将针对Harvard数据集的验证结果可视化并保存为图像，包含输入、输出、真值以及误差图。
    此函数内嵌于训练脚本中，无需外部依赖。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lrhsi = lrhsi[0].cpu().numpy()
    hrmsi = hrmsi[0].cpu().numpy()
    pred_hrhsi = pred_hrhsi[0].cpu().detach().numpy()
    gt_hrhsi = gt_hrhsi[0].cpu().numpy()
    pred_hrhsi = np.clip(pred_hrhsi, 0, 1)
    hrmsi_rgb = hrmsi.transpose(1, 2, 0)
    rgb_bands = [20, 10, 5]

    def to_rgb(img):
        rgb_img = img[rgb_bands, :, :].transpose(1, 2, 0)
        max_val, min_val = rgb_img.max(), rgb_img.min()
        if max_val > min_val:
            rgb_img = (rgb_img - min_val) / (max_val - min_val)
        return rgb_img

    error_map = np.mean(np.abs(pred_hrhsi - gt_hrhsi), axis=0)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    title_fontsize = 12

    axes[0].imshow(to_rgb(lrhsi))
    axes[0].set_title("Input LR-HSI (upscaled)", fontsize=title_fontsize)
    axes[0].axis('off')

    axes[1].imshow(hrmsi_rgb)
    axes[1].set_title("Input HR-MSI (RGB)", fontsize=title_fontsize)
    axes[1].axis('off')

    axes[2].imshow(to_rgb(pred_hrhsi) * 2.5)
    axes[2].set_title("Predicted HR-HSI", fontsize=title_fontsize)
    axes[2].axis('off')

    axes[3].imshow(to_rgb(gt_hrhsi) * 2.5)
    axes[3].set_title("Ground Truth HR-HSI", fontsize=title_fontsize)
    axes[3].axis('off')

    # --- 核心修改：将颜色条范围固定为 [0, 0.1] ---
    im = axes[4].imshow(error_map, cmap='viridis', vmin=0, vmax=0.1)
    axes[4].set_title("Error Map (MAE)", fontsize=title_fontsize)
    axes[4].axis('off')
    fig.colorbar(im, ax=axes[4])

    plt.tight_layout(pad=1.0)
    file_name = f"epoch_{epoch + 1}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"已保存验证图像至: {save_path}")


def main(args):
    # --- 1. 创建独立的实验结果保存路径 ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(args.save_dir, f"exp_harvard_{timestamp}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"实验结果、日志和检查点将保存至: {save_path}")

    # --- 2. WandB 初始化 ---
    wandb.init(project="SRModel-Harvard-Aligned", config=args, mode=args.wandb_mode)

    # --- 3. 环境设置 ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 4. 数据加载 ---
    print("正在设置数据加载器...")
    train_set = HarvardDataset(root_dir=args.data_path, dataset_type='train',
                               downsacle_factor=args.downscale_factor, transform=random_transform)
    val_set = HarvardDataset(root_dir=args.data_path, dataset_type='val',
                             downsacle_factor=args.downscale_factor)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"训练集大小: {len(train_set)}, 验证集大小: {len(val_set)}")

    # --- 5. 模型初始化 ---
    print("正在初始化新的模型: GroupedDisentangleFusionNet...")
    hsi_group_channels = [int(x) for x in args.hsi_group_channels.split(',')]
    total_hsi_channels = sum(hsi_group_channels)
    print(f"HSI将按以下通道数分组: {hsi_group_channels} (总计 {total_hsi_channels} 通道)")

    model = GroupedDisentangleFusionNet(
        hsi_group_channels=hsi_group_channels,
        msi_channels=args.msi_c,
        embed_dim=args.embed_dim,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {num_params / 1e6:.2f} M")
    wandb.watch(model)

    # --- 6. 优化器, 调度器, 损失函数 ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    l1_loss_fn = nn.L1Loss()
    sam_loss_fn = SAMLoss().to(device)
    ssim_loss_fn = SSIMLoss(channel=total_hsi_channels).to(device)

    # --- 7. 加载检查点 ---
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"正在从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_psnr = checkpoint.get('best_psnr', 0.0)
            print(f"已从 epoch {start_epoch} 恢复, 此前最佳PSNR为 {best_psnr:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("仅加载了模型权重，将从 epoch 0 开始新的训练。")

    # --- 8. 训练与验证循环 ---
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        num_batches_to_train = min(len(train_loader), 200)
        progress_bar = tqdm(enumerate(train_loader),
                            total=num_batches_to_train,
                            desc=f"Epoch {epoch + 1}/{args.epochs} [训练中]")

        for batch_idx, (lr_hsis, hr_msis, references, lrmsi_gts) in progress_bar:
            if batch_idx >= num_batches_to_train:
                break

            lr_hsis, hr_msis, references, lrmsi_gts = lr_hsis.to(device), hr_msis.to(device), references.to(device), lrmsi_gts.to(device)
            optimizer.zero_grad()
            pred_hrhsi, pred_lrmsi, hsi_ll_r, hsi_ll_u, msi_ll_r, msi_ll_u = model(lr_hsis, hr_msis)

            main_loss = l1_loss_fn(pred_hrhsi, references)
            aux_loss = l1_loss_fn(pred_lrmsi, lrmsi_gts)
            hsi_cos_sim = F.cosine_similarity(rearrange(hsi_ll_r, 'b c h w -> b (h w) c'), rearrange(hsi_ll_u, 'b c h w -> b (h w) c'), dim=2).mean()
            msi_cos_sim = F.cosine_similarity(rearrange(msi_ll_r, 'b c h w -> b (h w) c'), rearrange(msi_ll_u, 'b c h w -> b (h w) c'), dim=2).mean()
            cos_loss = (hsi_cos_sim + msi_cos_sim) / 2.0
            pred_groups = torch.split(pred_hrhsi, hsi_group_channels, dim=1)
            ref_groups = torch.split(references, hsi_group_channels, dim=1)
            sam_losses = [sam_loss_fn(p, r) for p, r in zip(pred_groups, ref_groups)]
            sam_loss = torch.mean(torch.stack(sam_losses))
            ssim_loss = ssim_loss_fn(pred_hrhsi, references)

            total_loss = (args.lambda_l1 * main_loss +
                          args.lambda_aux * aux_loss +
                          args.lambda_cos * cos_loss +
                          args.lambda_sam * sam_loss +
                          args.lambda_ssim * ssim_loss)

            total_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_psnr, val_ssim, val_sam, val_ergas = [], [], [], []
        with torch.no_grad():
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader),
                                    desc=f"Epoch {epoch + 1}/{args.epochs} [验证中]")
            for val_batch_idx, (lr_hsis, hr_msis, references, lrmsi_gts) in val_progress_bar:
                lr_hsis, hr_msis, references = lr_hsis.to(device), hr_msis.to(device), references.to(device)
                pred_hrhsi, _, _, _, _, _ = model(lr_hsis, hr_msis)
                if val_batch_idx == 0:
                    visuals_save_dir = os.path.join(save_path, "validation_visuals")
                    save_harvard_validation_results(
                        epoch=epoch, lrhsi=lr_hsis, hrmsi=hr_msis,
                        pred_hrhsi=pred_hrhsi, gt_hrhsi=references, save_dir=visuals_save_dir
                    )
                pred, gt = pred_hrhsi.clamp(0.0, 1.0), references.clamp(0.0, 1.0)
                val_psnr.append(psnr(gt, pred))
                val_ssim.append(ssim(pred, gt))
                val_sam.append(sam(gt, pred))
                val_ergas.append(ergas(gt, pred))

        avg_psnr, avg_ssim, avg_sam, avg_ergas = np.mean(val_psnr), np.mean(val_ssim), np.mean(val_sam), np.mean(val_ergas)
        print(
            f"\n--- Epoch {epoch + 1} 总结 --- | "
            f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}"
        )
        wandb.log({
            "Epoch": epoch + 1, "PSNR": avg_psnr, "SSIM": avg_ssim, "SAM": avg_sam, "ERGAS": avg_ergas,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })
        checkpoint_path = os.path.join(save_path, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }, checkpoint_path)
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"已保存新的最佳模型，PSNR为: {best_psnr:.4f}，路径: {best_model_path}\n")
            wandb.run.summary["best_psnr"] = best_psnr

    print("训练完成。")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为Harvard数据集训练GroupedDisentangleFusionNet模型")
    parser.add_argument('--data_path', type=str, default="F:/HarvardDataset/npy/", help="Harvard数据集的根目录。")
    parser.add_argument('--save_dir', type=str, default="./experiments_harvard", help="保存实验结果的目录。")
    parser.add_argument('--resume', type=str, default=None, help="用于恢复训练的检查点路径。")
    parser.add_argument('--wandb_mode', type=str, default="disabled", choices=['online', 'offline', 'disabled'], help="WandB模式。")
    parser.add_argument('--epochs', type=int, default=100, help="训练总轮数。")
    parser.add_argument('--batch_size', type=int, default=2, help="批次大小。")
    parser.add_argument('--lr', type=float, default=3e-4, help="初始学习率。")
    parser.add_argument('--downscale_factor', type=int, default=16, help="图像下采样因子。")
    parser.add_argument('--hsi_group_channels', type=str, default="17,14", help="逗号分隔的HSI通道分组。")
    parser.add_argument('--msi_c', type=int, default=3, help="MSI图像的通道数。")
    parser.add_argument('--embed_dim', type=int, default=48, help="嵌入维度。")
    parser.add_argument('--lambda_l1', type=float, default=1.0, help="主L1损失的权重。")
    parser.add_argument('--lambda_aux', type=float, default=0, help="辅助L1损失的权重。")
    parser.add_argument('--lambda_cos', type=float, default=0, help="余弦相似度解缠损失的权重。")
    parser.add_argument('--lambda_sam', type=float, default=0, help="SAM光谱损失的权重。")
    parser.add_argument('--lambda_ssim', type=float, default=0, help="SSIM结构相似性损失的权重。")
    parser.add_argument('--num_workers', type=int, default=0, help="数据加载的工作进程数。")

    args = parser.parse_args()
    main(args)
