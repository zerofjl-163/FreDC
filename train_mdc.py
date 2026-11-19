# train_mdc.py

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
import csv
import json
import random
from datetime import datetime
import wandb
import matplotlib.pyplot as plt
# --- 从您的项目文件导入所需模块 ---
from net.network import GroupedDisentangleFusionNet
from dataloader.load_mdc import MDCDataset
from metrics.metric import psnr, ssim, sam, ergas
# 假设SAMLoss在model文件夹下
from net.samLoss import SAMLoss

def seed_everything(seed):
    """Seed Python, NumPy and PyTorch and select deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    """Give every DataLoader worker a reproducible, distinct RNG state."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(spec):
    if spec == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {spec} requested CUDA, but CUDA is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {spec} is invalid; only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return device


def save_validation_results(epoch, lrhsi, hrmsi, pred_hrhsi, gt_hrhsi, save_dir="validation_results"):
    """
    将验证结果可视化并保存为图像，包含输入、输出、真值以及误差图。
    此版本已适配MDC数据集（50个HSI波段, 3通道MSI）。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 从batch中选择第一个样本并移至CPU
    lrhsi = lrhsi[0].cpu().numpy()
    hrmsi = hrmsi[0].cpu().numpy()
    pred_hrhsi = pred_hrhsi[0].cpu().detach().numpy()
    gt_hrhsi = gt_hrhsi[0].cpu().numpy()

    # 限制数据范围到 [0, 1] 以免显示错误
    pred_hrhsi = np.clip(pred_hrhsi, 0, 1)

    # --- **核心修改 1: 简化MSI处理** ---
    # MDC数据集的HR-MSI是标准的3通道RGB，直接转置即可用于显示
    hrmsi_rgb = hrmsi.transpose(1, 2, 0)

    # --- **核心修改 2: 为MDC数据集选择合适的HSI波段进行RGB显示** ---
    # 对于MDC数据集 (50个波段), 我们选择 [25, 15, 5] 分别代表 R, G, B
    rgb_bands = [20,2,28]

    def to_rgb(img):
        # 检查通道数是否足够
        if img.shape[0] < max(rgb_bands) + 1:
            # 如果通道数不足，则使用最后三个通道作为备用方案
            print(
                f"警告: 图像通道数 ({img.shape[0]}) 不足以选择指定的RGB波段 {rgb_bands}。将使用最后三个通道进行可视化。")
            rgb_display_bands = [img.shape[0] - 1, img.shape[0] - 2, img.shape[0] - 3]
        else:
            rgb_display_bands = rgb_bands

        # 将 (C, H, W) 转换为 (H, W, 3)
        rgb_img = img[rgb_display_bands, :, :].transpose(1, 2, 0)

        # 归一化用于显示
        max_val = rgb_img.max()
        min_val = rgb_img.min()
        if max_val > min_val:
            rgb_img = (rgb_img - min_val) / (max_val - min_val)
        return np.clip(rgb_img, 0, 1)

    # --- 计算误差图 (逻辑保持不变) ---
    # 计算绝对误差，并在光谱维度上取平均
    error_map = np.mean(np.abs(pred_hrhsi - gt_hrhsi), axis=0)

    # --- 绘制5个子图 (结构保持不变) ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    title_fontsize = 12

    axes[0].imshow(to_rgb(lrhsi))
    axes[0].set_title("Input LR-HSI (upscaled)", fontsize=title_fontsize)
    axes[0].axis('off')

    axes[1].imshow(hrmsi_rgb)
    axes[1].set_title("Input HR-MSI (RGB)", fontsize=title_fontsize)
    axes[1].axis('off')

    axes[2].imshow(to_rgb(pred_hrhsi))
    axes[2].set_title("Predicted HR-HSI", fontsize=title_fontsize)
    axes[2].axis('off')

    axes[3].imshow(to_rgb(gt_hrhsi))
    axes[3].set_title("Ground Truth HR-HSI", fontsize=title_fontsize)
    axes[3].axis('off')

    # 绘制误差图
    im = axes[4].imshow(error_map, cmap='viridis')
    axes[4].set_title("Error Map (MAE)", fontsize=title_fontsize)
    axes[4].axis('off')
    # 为误差图添加颜色条
    fig.colorbar(im, ax=axes[4])

    plt.tight_layout(pad=1.0)

    # --- 文件名保存逻辑 (保持不变，兼容数字和字符串) ---
    if isinstance(epoch, int):
        file_name = f"epoch_{epoch + 1}.png"
    else:
        file_name = f"results_{epoch}.png"

    save_path = os.path.join(save_dir, file_name)

    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved validation image to {save_path}")


def main(args):
    seed_everything(args.seed)

    # --- 1. 创建独立的实验结果保存路径 ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(args.save_dir, f"exp_mdc_{timestamp}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"实验结果、日志和检查点将保存至: {save_path}")

    # 保存本次运行的完整启动配置。文件位于 --save_dir 下对应的实验目录中，
    # 与检查点放在一起，避免不同学习率/恢复路径的实验相互覆盖。
    config_path = os.path.join(save_path, "training_config.json")
    history_path = os.path.join(save_path, "training_history.csv")
    run_config = vars(args).copy()
    run_config.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_dir": os.path.abspath(save_path),
    })
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(run_config, config_file, ensure_ascii=False, indent=2)

    history_fields = [
        "epoch", "loss_stage", "learning_rate",
        "lambda_aux", "lambda_cos", "lambda_sam",
        "train_total_loss", "train_main_loss", "train_aux_loss",
        "train_cos_loss", "train_sam_loss",
        "val_psnr", "val_ssim", "val_sam", "val_ergas",
    ]
    with open(history_path, 'w', newline='', encoding='utf-8') as history_file:
        csv.DictWriter(history_file, fieldnames=history_fields).writeheader()
    print(f"训练参数已保存至: {config_path}")
    print(f"训练历史将保存至: {history_path}")

    # --- 2. WandB 初始化 ---
    wandb.init(project="MDC-Dataset-Training", config=args, mode=args.wandb_mode)

    # --- 3. 环境设置 ---
    device = resolve_device(args.device)
    print(f"正在使用设备: {device}")

    # --- 4. 数据加载 ---
    print("正在设置MDC数据加载器...")
    train_set = MDCDataset(root_dir=args.data_path, dataset_type='train',
                           downscale_factor=args.downscale_factor)#de=args.denoise
    val_set = MDCDataset(root_dir=args.data_path, dataset_type='val',
                         downscale_factor=args.downscale_factor)#, de=args.denoise

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )
    print(f"训练集大小: {len(train_set)}, 验证集大小: {len(val_set)}")

    # --- 5. 模型初始化 ---
    print("正在初始化模型: GroupedDisentangleFusionNet...")
    hsi_group_channels = [int(x) for x in args.hsi_group_channels.split(',')]
    print(f"HSI将按以下通道数分组: {hsi_group_channels}")

    model = GroupedDisentangleFusionNet(
        hsi_group_channels=hsi_group_channels,
        msi_channels=args.msi_c,
        embed_dim=args.embed_dim
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {num_params / 1e6:.2f} M")
    wandb.watch(model)

    # --- 6. 优化器, 调度器, 损失函数 ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)#1e-6
    l1_loss_fn = nn.L1Loss()
    sam_loss_fn = SAMLoss().to(device)

    # --- 7. 加载检查点 (断点续训) ---
    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"正在从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # start_epoch = checkpoint['epoch'] + 1
        # best_psnr = checkpoint.get('best_psnr', 0.0)
        model.load_state_dict(checkpoint)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = 0
        best_psnr = 0
        print(f"已从 epoch {start_epoch} 恢复, 此前最佳PSNR为 {best_psnr:.4f}")

    # --- 8. 训练与验证循环 ---
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        model.train()

        # 两阶段损失调度（epoch 按实验记录从 1 开始计数）：
        #   1～loss_switch_epoch：使用原版权重；
        #   loss_switch_epoch+1～结束：关闭辅助/解耦损失，增强 SAM 损失。
        if epoch < args.loss_switch_epoch:
            lambda_aux = args.lambda_aux
            lambda_cos = args.lambda_cos
            lambda_sam = args.lambda_sam
            loss_stage = 1
        else:
            lambda_aux = args.stage2_lambda_aux
            lambda_cos = args.stage2_lambda_cos
            lambda_sam = args.stage2_lambda_sam
            loss_stage = 2

        # 在第二阶段开始时将学习率恢复为本次实验的初始学习率，并为剩余
        # epoch 重新启动余弦退火。以100轮、切换点40为例，新调度器覆盖41～100轮。
        if epoch == args.loss_switch_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
                param_group['initial_lr'] = args.lr
            stage2_epochs = max(1, args.epochs - args.loss_switch_epoch)
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=stage2_epochs,
                eta_min=1e-6,
            )
            print(
                f"第二阶段学习率已恢复为初始值 {args.lr:.6g}，"
                f"余弦调度器已按剩余 {stage2_epochs} 轮重新启动。"
            )

        if epoch == start_epoch or epoch == args.loss_switch_epoch:
            stage_end = min(args.loss_switch_epoch, args.epochs) if loss_stage == 1 else args.epochs
            print(
                f"损失阶段 {loss_stage}：epoch {epoch + 1}-{stage_end}，"
                f"lambda_aux={lambda_aux:g}, lambda_cos={lambda_cos:g}, "
                f"lambda_sam={lambda_sam:g}"
            )
        epoch_lr = optimizer.param_groups[0]['lr']

        train_loss_sums = {
            "total": 0.0,
            "main": 0.0,
            "aux": 0.0,
            "cos": 0.0,
            "sam": 0.0,
        }
        trained_batches = 0

        # --- **核心修改：设置进度条和批次限制** ---
        # 如果指定了每个 epoch 的批次数，则使用该值，否则使用整个加载器的长度
        num_batches_to_train = args.batches_per_epoch if args.batches_per_epoch else len(train_loader)
        progress_bar = tqdm(
            enumerate(train_loader),
            total=num_batches_to_train,
            desc=f"Epoch {epoch + 1}/{args.epochs} [训练中]"
        )

        for i, (lr_hsis, hr_msis, references, lrmsi_gts) in progress_bar:
            # --- **核心修改：检查是否达到批次限制** ---
            if args.batches_per_epoch is not None and i >= args.batches_per_epoch:
                break

            lr_hsis, hr_msis, references, lrmsi_gts = lr_hsis.to(device), hr_msis.to(device), references.to(
                device), lrmsi_gts.to(device)

            optimizer.zero_grad()

            pred_hrhsi, pred_lrmsi, hsi_ll_r, hsi_ll_u, msi_ll_r, msi_ll_u = model(lr_hsis, hr_msis)
            # --- 计算各类损失 ---
            main_loss = l1_loss_fn(pred_hrhsi, references)#HRHSI重建损失
            aux_loss = l1_loss_fn(pred_lrmsi, lrmsi_gts)#LRMSI重建损失

            hsi_cos_sim = F.cosine_similarity(rearrange(hsi_ll_r, 'b c h w -> b (h w) c'),
                                              rearrange(hsi_ll_u, 'b c h w -> b (h w) c'), dim=2).mean()
            msi_cos_sim = F.cosine_similarity(rearrange(msi_ll_r, 'b c h w -> b (h w) c'),
                                              rearrange(msi_ll_u, 'b c h w -> b (h w) c'), dim=2).mean()
            cos_loss = (hsi_cos_sim + msi_cos_sim) / 2.0

            pred_groups = torch.split(pred_hrhsi, hsi_group_channels, dim=1)
            ref_groups = torch.split(references, hsi_group_channels, dim=1)
            sam_losses = [sam_loss_fn(p, r) for p, r in zip(pred_groups, ref_groups)]
            sam_loss = torch.mean(torch.stack(sam_losses))

            # --- 计算总损失 ---
            total_loss = main_loss + lambda_aux * aux_loss + lambda_cos * cos_loss + lambda_sam * sam_loss

            total_loss.backward()
            optimizer.step()
            train_loss_sums["total"] += float(total_loss.detach())
            train_loss_sums["main"] += float(main_loss.detach())
            train_loss_sums["aux"] += float(aux_loss.detach())
            train_loss_sums["cos"] += float(cos_loss.detach())
            train_loss_sums["sam"] += float(sam_loss.detach())
            trained_batches += 1
            progress_bar.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        if trained_batches == 0:
            raise RuntimeError("当前 epoch 没有执行任何训练 batch，无法记录训练损失。")
        avg_train_losses = {
            name: value / trained_batches for name, value in train_loss_sums.items()
        }
        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_psnr, val_ssim, val_sam, val_ergas = [], [], [], []
        with torch.no_grad():
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader),
                                    desc=f"Epoch {epoch + 1}/{args.epochs} [验证中]")
            for i, (lr_hsis, hr_msis, references, lrmsi_gts) in val_progress_bar:
                if args.batches_per_epoch is not None and i >= args.batches_per_epoch / 5:
                    break
                lr_hsis, hr_msis, references = lr_hsis.to(device), hr_msis.to(device), references.to(device)

                pred_hrhsi, _, _, _, _, _ = model(lr_hsis, hr_msis)

                pred, gt = pred_hrhsi.clamp(0.0, 1.0), references.clamp(0.0, 1.0)
                val_psnr.append(psnr(gt, pred))
                val_ssim.append(ssim(pred, gt))
                val_sam.append(sam(gt, pred))
                val_ergas.append(ergas(gt, pred))

                if i == 0:
                    vis_path = os.path.join(save_path, "visualizations")
                    save_validation_results(
                        epoch=epoch,
                        lrhsi=lr_hsis,
                        hrmsi=hr_msis,
                        pred_hrhsi=pred_hrhsi,
                        gt_hrhsi=references,
                        save_dir=vis_path
                    )

        avg_psnr, avg_ssim, avg_sam, avg_ergas = np.mean(val_psnr), np.mean(val_ssim), np.mean(val_sam), np.mean(
            val_ergas)
        print(
            f"\n--- Epoch {epoch + 1} 总结 --- | "
            f"Loss: {avg_train_losses['total']:.6f} | "
            f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}"
        )

        history_row = {
            "epoch": epoch + 1,
            "loss_stage": loss_stage,
            "learning_rate": epoch_lr,
            "lambda_aux": lambda_aux,
            "lambda_cos": lambda_cos,
            "lambda_sam": lambda_sam,
            "train_total_loss": avg_train_losses["total"],
            "train_main_loss": avg_train_losses["main"],
            "train_aux_loss": avg_train_losses["aux"],
            "train_cos_loss": avg_train_losses["cos"],
            "train_sam_loss": avg_train_losses["sam"],
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
            "val_sam": avg_sam,
            "val_ergas": avg_ergas,
        }
        # 每个 epoch 单独打开并追加，训练被中断时已完成的记录不会丢失。
        with open(history_path, 'a', newline='', encoding='utf-8') as history_file:
            writer = csv.DictWriter(history_file, fieldnames=history_fields)
            writer.writerow(history_row)

        # --- WandB 日志 ---
        wandb.log({
            "Epoch": epoch + 1,
            "PSNR": avg_psnr, "SSIM": avg_ssim, "SAM": avg_sam, "ERGAS": avg_ergas,
            "Learning Rate": epoch_lr,
            "Loss Stage": loss_stage,
            "lambda_aux": lambda_aux,
            "lambda_cos": lambda_cos,
            "lambda_sam": lambda_sam,
            "Train Total Loss": avg_train_losses["total"],
            "Train Main Loss": avg_train_losses["main"],
            "Train Aux Loss": avg_train_losses["aux"],
            "Train Cos Loss": avg_train_losses["cos"],
            "Train SAM Loss": avg_train_losses["sam"],
        })

        # ``best_model.pth`` is selected only by the mean PSNR over the
        # validation split.  The test split is never consulted here.
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            with open(os.path.join(save_path, "best_model_metrics.json"), 'w', encoding='utf-8') as best_file:
                json.dump({
                    'selection_metric': 'val_psnr',
                    'epoch': epoch + 1,
                    'val_psnr': float(avg_psnr),
                    'val_ssim': float(avg_ssim),
                    'val_sam': float(avg_sam),
                    'val_ergas': float(avg_ergas),
                }, best_file, ensure_ascii=False, indent=2)
            print(f"已保存新的最佳模型，PSNR为: {best_psnr:.4f}，路径: {best_model_path}\n")
            wandb.run.summary["best_psnr"] = best_psnr

        # Save this after updating ``best_psnr`` so last_checkpoint always
        # records the best validation score observed through the current epoch.
        checkpoint_path = os.path.join(save_path, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'epoch_learning_rate': epoch_lr,
            'loss_stage': loss_stage,
            'loss_weights': {
                'lambda_aux': lambda_aux,
                'lambda_cos': lambda_cos,
                'lambda_sam': lambda_sam,
            },
        }, checkpoint_path)

    print("训练完成。")
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为MDC数据集训练GroupedDisentangleFusionNet模型")

    # 路径和实验管理
    parser.add_argument('--data_path', type=str, default="/mnt/16T/data/FJL/MDC_New/", help="MDC数据集的根目录。")
    parser.add_argument('--save_dir', type=str, default="./experiments_mdc", help="保存实验结果的目录。")
    parser.add_argument('--resume', type=str, default="", help="用于恢复训练的检查点路径。")
    parser.add_argument('--wandb_mode', type=str, default="disabled", choices=['online', 'offline', 'disabled'],
                        help="WandB模式。")
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=500, help="训练总轮数。")
    parser.add_argument('--batch_size', type=int, default=16, help="批次大小。")
    parser.add_argument('--lr', type=float, default=5e-4, help="初始学习率。")
    parser.add_argument('--batches_per_epoch', type=int, default=None,
                        help="每个训练轮次处理的批次数。如果未指定，则使用整个数据集。")
    # 数据和模型参数
    parser.add_argument('--downscale_factor', type=int, default=4, help="图像下采样因子。")
    parser.add_argument('--hsi_group_channels', type=str, default="25,25", help="逗号分隔的HSI通道分组。")
    parser.add_argument('--msi_c', type=int, default=3, help="MSI图像的通道数。")
    # 模型架构参数
    parser.add_argument('--embed_dim', type=int, default=48, help="嵌入维度。")
    # 损失权重
    parser.add_argument('--lambda_aux', type=float, default=0.01, help="辅助L1损失的权重。")
    parser.add_argument('--lambda_cos', type=float, default=0.01, help="余弦相似度解缠损失的权重。")
    parser.add_argument('--lambda_sam', type=float, default=0.5, help="SAM光谱损失的权重。")
    parser.add_argument('--loss_switch_epoch', type=int, default=40,
                        help="使用原版权重的前期训练轮数，默认前40轮。")
    parser.add_argument('--stage2_lambda_aux', type=float, default=0.0,
                        help="切换后辅助L1损失的权重。")
    parser.add_argument('--stage2_lambda_cos', type=float, default=0.0,
                        help="切换后余弦相似度解缠损失的权重。")
    parser.add_argument('--stage2_lambda_sam', type=float, default=1.0,
                        help="切换后SAM光谱损失的权重。")

    # 系统参数
    parser.add_argument('--num_workers', type=int, default=4, help="数据加载的工作进程数。")
    parser.add_argument('--device', type=str, default='auto',
                        help="训练设备，例如 auto、cuda:0、cuda:1 或 cpu。")
    parser.add_argument('--seed', type=int, default=3407, help="Python、NumPy、PyTorch及DataLoader随机种子。")

    args = parser.parse_args()
    if not 0 <= args.loss_switch_epoch <= args.epochs:
        parser.error('--loss_switch_epoch 必须位于 0 和 --epochs 之间。')
    loss_weights = (
        args.lambda_aux, args.lambda_cos, args.lambda_sam,
        args.stage2_lambda_aux, args.stage2_lambda_cos, args.stage2_lambda_sam,
    )
    if any(weight < 0 for weight in loss_weights):
        parser.error('所有损失权重必须为非负数。')
    main(args)
