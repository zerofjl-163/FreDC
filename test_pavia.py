# test_pavia.py

import torch
import numpy as np
import os
import argparse

from torch.nn.functional import interpolate
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# --- 从您的项目文件导入所需模块 ---
from net.network import GroupedDisentangleFusionNet
# --- 修改导入，使用新的测试数据加载器 ---
from dataloader.load_pavia_test import PaviaTestDataset
from metrics.metric import psnr, ssim, sam, ergas


def save_validation_results(epoch, lrhsi, hrmsi, pred_hrhsi, gt_hrhsi, save_dir):
    """
    保存验证/测试结果的可视化图像，包括对比图和独立的误差图。
    （此版本中的误差图不包含颜色条或标题）。

    Args:
        epoch (str or int): 当前的 epoch 或标识符，用于文件名。
        lrhsi (torch.Tensor): 低分辨率高光谱图像张量 (B, C, H, W)。
        hrmsi (torch.Tensor): 高分辨率多光谱图像张量 (B, C, H, W)。
        pred_hrhsi (torch.Tensor): 模型预测的高分辨率高光谱图像张量 (B, C, H, W)。
        gt_hrhsi (torch.Tensor): 地面真实的高分辨率高光谱图像张量 (B, C, H, W)。
        save_dir (str): 保存图像的目录。
    """
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 准备张量 ---
    # 去掉批次维度 (B=1)，我们保留张量格式以便于进行插值操作
    lrhsi_t = lrhsi.squeeze(0)
    pred_hrhsi_t = pred_hrhsi.squeeze(0)
    gt_hrhsi_t = gt_hrhsi.squeeze(0)

    # 使用双三次插值将低分辨率高光谱图像上采样至与地面真实图像相同尺寸，以便于可视化对比
    target_size = gt_hrhsi_t.shape[1:]  # (H, W)
    lrhsi_upscaled_t = interpolate(lrhsi_t.unsqueeze(0), size=target_size, mode='bicubic', align_corners=False).squeeze(
        0)

    # 将张量移动到 CPU 并转换为 NumPy 数组
    lrhsi_upscaled = lrhsi_upscaled_t.cpu().numpy()
    pred_hrhsi = pred_hrhsi_t.cpu().numpy()
    gt_hrhsi = gt_hrhsi_t.cpu().numpy()

    # 将维度从 (C, H, W) 转换为 (H, W, C) 以便 Matplotlib 绘图
    lrhsi_upscaled = np.transpose(lrhsi_upscaled, (1, 2, 0))
    pred_hrhsi = np.transpose(pred_hrhsi, (1, 2, 0))
    gt_hrhsi = np.transpose(gt_hrhsi, (1, 2, 0))

    # 将数值裁剪到 [0, 1] 范围以确保正确显示
    lrhsi_upscaled = np.clip(lrhsi_upscaled, 0, 1)
    pred_hrhsi = np.clip(pred_hrhsi, 0, 1)
    gt_hrhsi = np.clip(gt_hrhsi, 0, 1)

    # --- 为假彩色RGB可视化选择波段 ---
    # 对于Pavia数据集, 波段 [50, 27, 10] 是一个常用的假彩色组合
    rgb_bands = [50, 27, 10]

    gt_rgb = gt_hrhsi[:, :, rgb_bands]
    pred_rgb = pred_hrhsi[:, :, rgb_bands]
    lrhsi_rgb = lrhsi_upscaled[:, :, rgb_bands]

    # --- 计算误差图 ---
    # 计算在所有光谱波段上的平均绝对误差 (Mean Absolute Error, MAE)
    error_map = np.mean(np.abs(gt_hrhsi - pred_hrhsi), axis=2)

    # --- 保存图像 ---
    # 保存包含地面真实、预测结果和上采样输入的对比图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Result for: {epoch}', fontsize=16)

    axs[0].imshow(gt_rgb)
    axs[0].set_title('Ground Truth (False-Color)')
    axs[0].axis('off')

    axs[1].imshow(pred_rgb)
    axs[1].set_title('Prediction (False-Color)')
    axs[1].axis('off')

    axs[2].imshow(lrhsi_rgb)
    axs[2].set_title('Upscaled LR-HSI Input')
    axs[2].axis('off')

    comparison_filename = os.path.join(save_dir, f'comparison_{epoch}.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(comparison_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存对比图至: {comparison_filename}")

    # --- **核心修改部分：将误差图单独保存为PNG（无颜色条和标题）** ---
    errormap_filename = os.path.join(save_dir, f'errormap_{epoch}.png')

    # 使用 plt.imsave 直接将 NumPy 数组保存为图像。
    # 这个函数不会添加任何额外的元素，如坐标轴、标题或颜色条。
    plt.imsave(errormap_filename, error_map, cmap='jet', vmin=0, vmax=0.1)

    print(f"已保存误差图至: {errormap_filename}")


def test(args):
    """
    使用指定的数据集和模型检查点来执行模型评估。
    此函数在完整的测试图像上进行评估，而不是在图像块上。
    """
    # --- 1. 环境设置 ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 2. 检查点和结果保存路径 ---
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件未找到 '{args.checkpoint_path}'")
        return

    # 创建一个新的文件夹以区分整图测试结果
    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    save_path = os.path.join(checkpoint_dir, "test_results_full_image")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"测试的可视化结果将保存至: {save_path}")

    # --- 3. 数据加载 (修改后) ---
    print("正在设置测试数据加载器 (加载完整图像)...")

    # 使用新的测试数据集，它直接处理完整的测试图像
    test_dataset = PaviaTestDataset(data_path=args.data_path, downscale_factor=args.downscale_factor)

    # batch_size 必须为 1，因为我们一次处理一张完整的图像
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 必须为1
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"测试数据集包含 {len(test_dataset)} 张完整图像。")

    # --- 4. 模型初始化 (无需更改) ---f
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
        embed_dim=args.embed_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params / 1e6:.2f} M")

    # --- 5. 加载训练好的模型权重 (无需更改) ---
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

    # --- 6. 评估循环 (修改后) ---
    print("\n开始在完整的测试图像上进行评估...")
    model.eval()

    with torch.no_grad():
        # test_loader 只会循环一次
        for lrhsi, hrmsi, hrhsi_gt in tqdm(test_loader, desc="[Testing Full Image]"):

            # 将数据移动到指定设备 (数据已有批次维度 B=1)
            lrhsi, hrmsi, hrhsi_gt = lrhsi.to(device), hrmsi.to(device), hrhsi_gt.to(device)

            # 模型前向传播。如果GPU内存不足，这里可能会出错。
            try:
                pred_hrhsi, _, _, _, _, _ = model(lrhsi, hrmsi)
            except torch.cuda.OutOfMemoryError:
                print("\n错误: GPU 显存不足，无法处理完整的测试图像。")
                print("建议实现滑窗推理 (sliding-window inference) 或使用显存更大的GPU。")
                return

            # 将预测和真实值都裁剪到 [0, 1] 范围内
            pred = pred_hrhsi.clamp(0, 1)
            gt = hrhsi_gt.clamp(0, 1)

            # 直接计算最终指标
            final_psnr = psnr(gt, pred)
            final_ssim = ssim(pred, gt)
            final_sam = sam(gt, pred)
            final_ergas = ergas(gt, pred)

            # 保存可视化结果
            print(f"正在保存可视化结果...")
            save_validation_results(
                epoch='test_full_image',  # 使用新的标识符
                lrhsi=lrhsi,
                hrmsi=hrmsi,
                pred_hrhsi=pred,
                gt_hrhsi=gt,
                save_dir=save_path
            )

    # --- 7. 打印最终指标 (修改后) ---
    print("\n--- 完整测试图像评估总结 ---")
    print(f"  PSNR:  {final_psnr:.4f}")
    print(f"  SSIM:  {final_ssim:.4f}")
    print(f"  SAM:   {final_sam:.4f}")
    print(f"  ERGAS: {final_ergas:.4f}")
    print("--------------------------")
    print("测试完成。")

    # --- **核心修改部分：将指标保存到文本文件** ---
    metrics_file_path = os.path.join(save_path, "metrics_summary.txt")

    # 准备要写入文件的内容
    metrics_content = (
        f"--- 完整测试图像评估总结 ---\n"
        f"模型检查点: {os.path.abspath(args.checkpoint_path)}\n"
        f"--------------------------\n"
        f"  PSNR:  {final_psnr:.4f}\n"
        f"  SSIM:  {final_ssim:.4f}\n"
        f"  SAM:   {final_sam:.4f}\n"
        f"  ERGAS: {final_ergas:.4f}\n"
        f"--------------------------\n"
    )

    # 写入文件
    try:
        with open(metrics_file_path, 'w', encoding='utf-8') as f:
            f.write(metrics_content)
        print(f"评估指标已成功保存至: {metrics_file_path}")
    except Exception as e:
        print(f"保存指标文件时出错: {e}")
    # --- **修改结束** ---

    print("测试完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test HSI Disentanglement and Fusion Network on the full test image.")

    # --- 路径参数 ---
    parser.add_argument('--data_path', type=str, default="/mnt/16T/data/FJL/Pavia/Pavia.mat",
                        help="指向 Pavia 数据集 (.mat 文件) 的路径。")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="指向训练好的模型检查点文件的路径 (例如: './experiments/exp_.../best_model.pth')。")
    parser.add_argument('--downscale_factor', type=int, default=4, help="downscale factor during training.")
    # --- 模型参数 (必须与训练时使用的参数一致) ---
    parser.add_argument('--msi_c', type=int, default=4, help="多光谱图像 (MSI) 的通道数。")
    parser.add_argument('--hsi_group_channels', type=str, default="35,35,32",
                        help="用于每个子网的高光谱图像 (HSI) 通道分组，以逗号分隔。")
    parser.add_argument('--embed_dim', type=int, default=48,
                        help="模型中 Transformer 模块的嵌入维度。")

    # --- 系统参数 ---
    parser.add_argument('--num_workers', type=int, default=0, help="用于数据加载的工作进程数。")

    args = parser.parse_args()

    test(args)
