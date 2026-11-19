# test_harvard.py

import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
# --- 从您的项目文件导入所需模块 ---
from net.network import GroupedDisentangleFusionNet
from dataloader.load_harvard import HarvardDataset
from metrics.metric import psnr, ssim, sam, ergas


# --- 内置的可视化函数，专为Harvard数据集定制 ---
def save_harvard_validation_results(epoch, lrhsi, hrmsi, pred_hrhsi, gt_hrhsi, save_dir):
    """
    将针对Harvard数据集的验证结果可视化并保存为图像。
    此版本会保存一个包含所有信息的组合图，以及一个独立的、无边框的误差图。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- 数据准备 ---
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

    # --- 计算误差图 ---
    error_map = np.mean(np.abs(pred_hrhsi - gt_hrhsi), axis=0)

    # --- 文件名定义 ---
    if isinstance(epoch, int):
        base_name = f"epoch_{epoch + 1}"
    else:
        base_name = f"{epoch}"

    comparison_file_name = f"{base_name}.png"
    errormap_file_name = f"errormap_{base_name}.png"  # 为独立的误差图定义文件名

    comparison_save_path = os.path.join(save_dir, comparison_file_name)
    errormap_save_path = os.path.join(save_dir, errormap_file_name)

    # --- **核心修改：单独保存纯净的误差图** ---
    # 使用 plt.imsave 可以直接将numpy数组保存为图像，不会添加任何坐标轴、标题或边框
    # vmin 和 vmax 用于将数值范围标准化到与组合图一致的颜色范围
    plt.imsave(errormap_save_path, error_map, cmap='jet', vmin=0, vmax=0.09)
    print(f"已单独保存纯净误差图至: {errormap_save_path}")
    # --- **修改结束** ---

    # --- 保存包含所有信息的组合对比图 (逻辑不变) ---
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

    im = axes[4].imshow(error_map, cmap='jet', vmin=0, vmax=0.1)
    axes[4].set_title("Error Map (MAE)", fontsize=title_fontsize)
    axes[4].axis('off')
    fig.colorbar(im, ax=axes[4])

    plt.tight_layout(pad=1.0)
    plt.savefig(comparison_save_path)
    plt.close(fig)
    print(f"已保存组合对比图至: {comparison_save_path}")


def test(args):
    """
    使用指定的数据集和模型检查点来执行模型评估。
    此函数在Harvard测试集的每一张完整图像上进行评估，并聚合最终结果。
    """
    # --- 1. 环境设置 ---
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 2. 检查点和结果保存路径 ---
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件未找到 '{args.checkpoint_path}'")
        return

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    save_path = os.path.join(checkpoint_dir, "test_results_full_images")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"测试的可视化结果与指标将保存至: {save_path}")

    # --- 3. 数据加载 ---
    print("正在设置测试数据加载器 (加载完整图像)...")
    test_dataset = HarvardDataset(root_dir=args.data_path, dataset_type='test',
                                  downsacle_factor=args.downscale_factor)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    dataset_size = len(test_dataset)
    print(f"测试数据集包含 {dataset_size} 张完整图像。")

    # --- 解析和验证 --select_indices 参数 ---
    selected_indices = None
    if args.select_indices:
        try:
            indices_str = args.select_indices.split(',')
            selected_indices = {int(i.strip()) for i in indices_str if i.strip()}

            if not selected_indices:
                print("警告: --select_indices 参数已提供但为空，将继续测试所有样本。")
                selected_indices = None
            else:
                max_idx = max(selected_indices)
                if max_idx >= dataset_size:
                    print(f"错误: 指定的索引 {max_idx} 超出数据集范围 (0-{dataset_size - 1})。")
                    return
                print(f"-> 将仅测试指定的 {len(selected_indices)} 个样本，索引为: {sorted(list(selected_indices))}")

        except ValueError:
            print(f"错误: --select_indices 参数格式无效。请提供以逗号分隔的整数，例如 '0,5,10'。")
            return
    else:
        print("-> 未指定特定样本，将测试所有样本。")

    metrics_per_image_path = os.path.join(save_path, "metrics_per_image.csv")
    try:
        with open(metrics_per_image_path, 'w', encoding='utf-8') as f:
            f.write("image_index,psnr,ssim,sam,ergas\n")
    except Exception as e:
        print(f"创建指标CSV文件时出错: {e}")
        return

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
        msi_channels=args.msi_c, embed_dim=args.embed_dim
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {num_params / 1e6:.2f} M")

    # --- 5. 加载训练好的模型权重 ---
    print(f"正在从以下路径加载模型权重: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("模型权重加载成功。")
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        return

    # --- 6. 评估循环 ---
    print("\n开始在完整的测试图像上进行评估...")
    model.eval()
    all_psnr, all_ssim, all_sam, all_ergas = [], [], [], []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="[测试中]")
        for i, (lrhsi, hrmsi, reference, _) in progress_bar:

            if selected_indices is not None and i not in selected_indices:
                continue

            lrhsi, hrmsi, reference = lrhsi.to(device), hrmsi.to(device), reference.to(device)

            try:
                pred_hrhsi, _, _, _, _, _ = model(lrhsi, hrmsi)
            except torch.cuda.OutOfMemoryError:
                print(f"\n错误: GPU 显存不足，无法处理图像 {i}。")
                print("建议实现滑窗推理 (sliding-window inference) 或使用显存更大的GPU。")
                continue

            pred, gt = pred_hrhsi.clamp(0, 1), reference.clamp(0, 1)

            current_psnr = psnr(gt, pred)
            current_ssim = ssim(pred, gt)
            current_sam = sam(gt, pred)
            current_ergas = ergas(gt, pred)

            all_psnr.append(current_psnr)
            all_ssim.append(current_ssim)
            all_sam.append(current_sam)
            all_ergas.append(current_ergas)

            try:
                with open(metrics_per_image_path, 'a', encoding='utf-8') as f:
                    f.write(f"{i},{current_psnr:.4f},{current_ssim:.4f},{current_sam:.4f},{current_ergas:.4f}\n")
            except Exception as e:
                print(f"\n写入指标到CSV时出错: {e}")

            save_harvard_validation_results(
                epoch=f'test_image_{i}', lrhsi=lrhsi, hrmsi=hrmsi,
                pred_hrhsi=pred, gt_hrhsi=gt, save_dir=save_path
            )

            progress_bar.set_postfix(
                PSNR=f"{current_psnr:.2f}",
                SSIM=f"{current_ssim:.3f}",
                SAM=f"{current_sam:.2f}",
                ERGAS=f"{current_ergas:.2f}"
            )
            del lrhsi, hrmsi, reference, pred_hrhsi, pred, gt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- 7. 聚合、打印并保存最终指标 ---
    if not all_psnr:
        print("\n没有样本被评估，无法生成总结报告。")
        print("请检查您通过 --select_indices 提供的索引是否正确。")
        return

    avg_psnr, std_psnr = np.mean(all_psnr), np.std(all_psnr)
    avg_ssim, std_ssim = np.mean(all_ssim), np.std(all_ssim)
    avg_sam, std_sam = np.mean(all_sam), np.std(all_sam)
    avg_ergas, std_ergas = np.mean(all_ergas), np.std(all_ergas)

    summary_content = (
        f"--- Harvard 测试集评估总结 ({len(all_psnr)} 张图像) ---\n"
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

    metrics_file_path = os.path.join(save_path, "metrics_summary.txt")
    try:
        with open(metrics_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"评估指标总结已成功保存至: {metrics_file_path}")
    except Exception as e:
        print(f"保存指标文件时出错: {e}")

    print("测试完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="在Harvard数据集上测试GroupedDisentangleFusionNet模型")

    parser.add_argument('--data_path', type=str, default="F:/HarvardDataset/npy/", help="Harvard数据集的根目录。")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="指向训练好的模型检查点文件路径。")

    parser.add_argument('--select_indices', type=str, default="",
                        help="指定要测试的样本索引，以逗号分隔 (例如 '0,5,10')。如果未提供，则测试所有样本。")

    parser.add_argument('--downscale_factor', type=int, default=16, help="图像下采样因子。")
    parser.add_argument('--hsi_group_channels', type=str, default="17,14", help="逗号分隔的HSI通道分组。")
    parser.add_argument('--msi_c', type=int, default=3, help="MSI图像的通道数。")
    parser.add_argument('--embed_dim', type=int, default=48, help="模型Transformer块的嵌入维度。")
    parser.add_argument('--encoder_layers', type=int, default=2, help="编码器层数。")
    parser.add_argument('--decoder_blocks', type=int, default=3, help="解码器Transformer块数。")
    parser.add_argument('--decoder_heads', type=int, default=2, help="解码器多头注意力头数。")

    parser.add_argument('--num_workers', type=int, default=0, help="数据加载的工作进程数。")

    args = parser.parse_args()
    test(args)
