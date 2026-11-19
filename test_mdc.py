# test_mdc.py

import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv  # --- **核心修改: 导入csv模块** ---

# --- 从您的项目文件导入所需模块 ---
from net.network import GroupedDisentangleFusionNet
from dataloader.load_mdc import MDCDataset
from metrics.metric import psnr, ssim, sam, ergas


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

    # MDC数据集的HR-MSI是标准的3通道RGB，直接转置即可用于显示
    hrmsi_rgb = hrmsi.transpose(1, 2, 0)

    # 为MDC数据集选择合适的HSI波段进行RGB显示
    # 对于MDC数据集 (50个波段), 我们选择 [25, 15, 5] 分别代表 R, G, B
    rgb_bands = [20, 2, 28]#25,15,5

    def to_rgb(img):
        if img.shape[0] < max(rgb_bands) + 1:
            print(
                f"警告: 图像通道数 ({img.shape[0]}) 不足以选择指定的RGB波段 {rgb_bands}。将使用最后三个通道进行可视化。")
            rgb_display_bands = [img.shape[0] - 1, img.shape[0] - 2, img.shape[0] - 3]
        else:
            rgb_display_bands = rgb_bands

        rgb_img = img[rgb_display_bands, :, :].transpose(1, 2, 0)

        max_val = rgb_img.max()
        min_val = rgb_img.min()
        if max_val > min_val:
            rgb_img = (rgb_img - min_val) / (max_val - min_val)
        return np.clip(rgb_img, 0, 1)

    error_map = np.mean(np.abs(pred_hrhsi - gt_hrhsi), axis=0)
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
    im = axes[4].imshow(error_map, cmap='viridis')
    axes[4].set_title("Error Map (MAE)", fontsize=title_fontsize)
    axes[4].axis('off')
    fig.colorbar(im, ax=axes[4])
    plt.tight_layout(pad=1.0)

    if isinstance(epoch, int):
        file_name = f"epoch_{epoch + 1}.png"
    else:
        file_name = f"results_{epoch}.png"

    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved validation image to {save_path}")


def test(args):
    """
    使用指定的数据集和模型检查点来执行模型评估。
    此函数在MDC测试集的每一张完整图像上进行评估，并聚合最终结果。
    """
    # --- 1. 环境设置 ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 2. 检查点和结果保存路径 ---
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件未找到 '{args.checkpoint_path}'")
        return

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    save_path = os.path.join(checkpoint_dir, "test_results_full_images_mdc")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"测试的可视化结果与指标将保存至: {save_path}")

    # --- 3. 数据加载 ---
    print("正在设置MDC测试数据加载器 (加载完整图像)...")

    test_dataset = MDCDataset(root_dir=args.data_path, dataset_type='test',
                              downscale_factor=args.downscale_factor, de=args.denoise)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"测试数据集包含 {len(test_dataset)} 张完整图像。")

    # --- 4. 模型初始化 ---
    print("正在初始化模型...")
    try:
        hsi_group_channels = [int(x.strip()) for x in args.hsi_group_channels.split(',')]
        print(f"高光谱图像将按以下通道数分组: {hsi_group_channels}")
    except ValueError:
        print(f"错误: hsi_group_channels 参数格式无效。")
        return

    model = GroupedDisentangleFusionNet(
        hsi_group_channels=hsi_group_channels,
        msi_channels=args.msi_c,
        embed_dim=args.embed_dim
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params / 1e6:.2f} M")

    # --- 5. 加载训练好的模型权重 ---
    print(f"正在从以下路径加载模型权重: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        return

    # --- 6. 评估循环 ---
    print("\n开始在完整的MDC测试图像上进行评估...")
    model.eval()

    all_psnr, all_ssim, all_sam, all_ergas = [], [], [], []
    # --- **核心修改: 初始化一个列表用于存储每张图片的详细指标** ---
    detailed_metrics = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="[测试中]")
        for i, (lrhsi, hrmsi, reference, _) in progress_bar:

            lrhsi, hrmsi, reference = lrhsi.to(device), hrmsi.to(device), reference.to(device)

            try:
                pred_hrhsi, _, _, _, _, _ = model(lrhsi, hrmsi)
            except torch.cuda.OutOfMemoryError:
                print(f"\n错误: GPU 显存不足，无法处理图像 {i}。")
                continue

            pred = pred_hrhsi.clamp(0, 1)
            gt = reference.clamp(0, 1)

            # 计算指标
            current_psnr = psnr(gt, pred)
            current_ssim = ssim(pred, gt)
            current_sam = sam(gt, pred)
            current_ergas = ergas(gt, pred)

            # 添加到聚合列表
            all_psnr.append(current_psnr)
            all_ssim.append(current_ssim)
            all_sam.append(current_sam)
            all_ergas.append(current_ergas)

            # --- **核心修改: 记录详细指标到字典** ---
            # 从数据加载器中获取原始文件名
            filename = os.path.basename(test_dataset.dataset_paths[i]['hr_hsi'])
            detailed_metrics.append({
                'filename': filename,
                'psnr': current_psnr,
                'ssim': current_ssim,
                'sam': current_sam,
                'ergas': current_ergas
            })

            # 为当前图像保存可视化结果
            save_validation_results(
                epoch=f'test_image_{i}_{filename.replace(".npy", "")}',
                lrhsi=lrhsi,
                hrmsi=hrmsi,
                pred_hrhsi=pred,
                gt_hrhsi=gt,
                save_dir=save_path
            )
            progress_bar.set_postfix(PSNR=f"{current_psnr:.2f}", SAM=f"{current_sam:.2f}")

    # --- 7. 聚合、打印并保存最终指标 ---
    avg_psnr, std_psnr = np.mean(all_psnr), np.std(all_psnr)
    avg_ssim, std_ssim = np.mean(all_ssim), np.std(all_ssim)
    avg_sam, std_sam = np.mean(all_sam), np.std(all_sam)
    avg_ergas, std_ergas = np.mean(all_ergas), np.std(all_ergas)

    summary_content = (
        f"--- MDC 测试集评估总结 ({len(all_psnr)} 张图像) ---\n"
        f"模型检查点: {os.path.abspath(args.checkpoint_path)}\n"
        f"---------------------------------------------------\n"
        f" 指标      |   平均值   |   标准差\n"
        f"---------------------------------------------------\n"
        f" PSNR      |   {avg_psnr:<8.4f} |   {std_psnr:<8.4f}\n"
        f" SSIM      |   {avg_ssim:<8.4f} |   {std_ssim:<8.4f}\n"
        f" SAM       |   {avg_sam:<8.4f} |   {std_sam:<8.4f}\n"
        f" ERGAS     |   {avg_ergas:<8.4f} |   {std_ergas:<8.4f}\n"
        f"---------------------------------------------------\n"
    )
    print(f"\n{summary_content}")

    # 保存总结文件
    metrics_file_path = os.path.join(save_path, "metrics_summary.txt")
    try:
        with open(metrics_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"评估指标总结已成功保存至: {metrics_file_path}")
    except Exception as e:
        print(f"保存指标总结文件时出错: {e}")

    # --- **核心修改: 将详细指标写入CSV文件** ---
    csv_file_path = os.path.join(save_path, "metrics_per_image.csv")
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'psnr', 'ssim', 'sam', 'ergas']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_metrics)
        print(f"每张图像的详细指标已成功保存至: {csv_file_path}")
    except Exception as e:
        print(f"保存详细指标CSV文件时出错: {e}")

    print("测试完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="在MDC数据集上测试GroupedDisentangleFusionNet模型")

    parser.add_argument('--data_path', type=str, default="/mnt/16T/data/FJL/MDC_New/", help="MDC数据集的根目录。")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="指向训练好的模型检查点文件路径。")

    parser.add_argument('--downscale_factor', type=int, default=4, help="图像下采样因子。")
    parser.add_argument('--hsi_group_channels', type=str, default="25,25", help="逗号分隔的HSI通道分组。")
    parser.add_argument('--msi_c', type=int, default=3, help="MSI图像的通道数。")
    parser.add_argument('--denoise', action='store_true', default=True, dest='denoise',
                        help="加载数据时启用小波去噪 (默认启用)。")
    parser.add_argument('--no-denoise', action='store_false', dest='denoise', help="加载数据时禁用小波去噪。")
    parser.add_argument('--embed_dim', type=int, default=48, help="模型Transformer块的嵌入维度。")
    parser.add_argument('--encoder_layers', type=int, default=2, help="编码器层数。")
    parser.add_argument('--decoder_blocks', type=int, default=3, help="解码器Transformer块数。")
    parser.add_argument('--decoder_heads', type=int, default=2, help="解码器多头注意力头数。")
    parser.add_argument('--num_workers', type=int, default=0, help="数据加载的工作进程数。")

    args = parser.parse_args()
    test(args)
