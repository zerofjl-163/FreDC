# visualize.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def save_validation_results(epoch, lrhsi, hrmsi, pred_hrhsi, gt_hrhsi, save_dir="validation_results"):
    """
    将验证结果可视化并保存为图像，包含输入、输出、真值以及误差图。
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

    # --- 处理4通道MSI，提取RGB ---
    # 原始顺序是 B, G, R, NIR (索引 0, 1, 2, 3)
    # 我们需要 R, G, B (索引 2, 1, 0)
    if hrmsi.shape[0] == 4:
        hrmsi_rgb = hrmsi[[2, 1, 0], :, :]
    else:
        hrmsi_rgb = hrmsi

    # HSI选择RGB波段进行显示
    rgb_bands = [60, 35, 10]

    def to_rgb(img):
        # 将 (C, H, W) 转换为 (H, W, 3)
        rgb_img = img[rgb_bands, :, :].transpose(1, 2, 0)
        # 归一化用于显示
        max_val = rgb_img.max()
        min_val = rgb_img.min()
        if max_val > min_val:
            rgb_img = (rgb_img - min_val) / (max_val - min_val)
        return rgb_img

    # --- 新增：计算误差图 ---
    # 计算绝对误差，并在光谱维度上取平均
    error_map = np.mean(np.abs(pred_hrhsi - gt_hrhsi), axis=0)

    # --- 绘制5个子图 ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    title_fontsize = 12

    axes[0].imshow(to_rgb(lrhsi))
    axes[0].set_title("Input LR-HSI (upscaled)", fontsize=title_fontsize)
    axes[0].axis('off')

    axes[1].imshow(hrmsi_rgb.transpose(1, 2, 0))
    axes[1].set_title("Input HR-MSI (RGB)", fontsize=title_fontsize)
    axes[1].axis('off')

    axes[2].imshow(to_rgb(pred_hrhsi) * 3)
    axes[2].set_title("Predicted HR-HSI", fontsize=title_fontsize)
    axes[2].axis('off')

    axes[3].imshow(to_rgb(gt_hrhsi) * 3)
    axes[3].set_title("Ground Truth HR-HSI", fontsize=title_fontsize)
    axes[3].axis('off')

    # 绘制误差图
    im = axes[4].imshow(error_map, cmap='jet', vmin=0, vmax=0.1)
    axes[4].set_title("Error Map (MAE)", fontsize=title_fontsize)
    axes[4].axis('off')
    # 为误差图添加颜色条
    fig.colorbar(im, ax=axes[4])

    plt.tight_layout(pad=1.0)

    # --- **核心修改部分** ---
    # 判断 epoch 是数字还是字符串，以生成正确的文件名
    if isinstance(epoch, int):
        # 如果是数字 (来自训练过程)，文件名包含 epoch+1
        file_name = f"epoch_{epoch + 1}.png"
    else:
        # 如果是字符串 (来自测试过程)，直接使用该字符串作为文件名
        file_name = f"results_{epoch}.png"

    save_path = os.path.join(save_dir, file_name)

    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved validation image to {save_path}")