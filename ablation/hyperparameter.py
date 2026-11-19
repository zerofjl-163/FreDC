import matplotlib.pyplot as plt

# ==========================================
# 全局样式设置
# ==========================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"

color_cons = "#a4aaca"
color_fsl = "#abccc6"

x_ticks = [0, 0.005, 0.01, 0.015, 0.02]
y_psnr_cons = [40.5347, 42.8520, 44.7850, 43.2130, 42.8540]
y_psnr_fsl = [41.2560, 42.6240, 44.7850, 43.9520, 43.6120]


def plot_final_sensitivity_fixed(x, y, line_color, filename):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)

    max_idx = y.index(max(y))
    x_max = x[max_idx]
    y_max = y[max_idx]

    # 1. 绘制核心曲线
    # clip_on=False 确保圆点即使在 x=0 处也会完整渲染
    ax.plot(x, y, marker='o', color=line_color, linewidth=2.5, markersize=8,
            markeredgecolor='black', markeredgewidth=0.5, label='PSNR',
            zorder=5, clip_on=False)

    # 2. 绘制参考虚线
    y_limit_min = min(y) - 0.5
    ax.hlines(y=y_max, xmin=min(x), xmax=x_max, colors=line_color,
              linestyles=':', linewidth=1.5, zorder=2)
    ax.vlines(x=x_max, ymin=y_limit_min, ymax=y_max, colors=line_color,
              linestyles=':', linewidth=1.5, zorder=2)

    # 3. 边框设置
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. 坐标轴范围
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(y_limit_min, max(y) + 0.5)

    # ==========================================
    # 【修改重点】字体大小调整区域
    # ==========================================

    # 方法 A: 使用 tick_params 同时设置 X轴和Y轴刻度字体 (推荐)
    ax.tick_params(axis='both', labelsize=16)

    # 方法 B: 如果你需要单独设置 X轴刻度 (如你原代码所示)
    # ax.set_xticks(x)
    # ax.set_xticklabels(x, fontsize=16)

    # 添加坐标轴标签 (可选)，并设置字体大小
    # 注意：这里为了演示效果添加了标签文字，你可以根据实际需求修改
    ax.set_xlabel('', fontsize=16)
    ax.set_ylabel('', fontsize=16)

    # 5. 图例
    legend = ax.legend(loc='lower right', frameon=True, fontsize=16,
                       facecolor='white', edgecolor='black', framealpha=1)
    legend.get_frame().set_linewidth(1.0)

    # 6. 背景网格
    ax.grid(True, which='both', linestyle='--', alpha=0.2, color='gray', zorder=1)

    plt.tight_layout()
    plt.show()


# 执行绘图
plot_final_sensitivity_fixed(x_ticks, y_psnr_cons, color_cons, "sensitivity_cons")
plot_final_sensitivity_fixed(x_ticks, y_psnr_fsl, color_fsl, "sensitivity_fsl")