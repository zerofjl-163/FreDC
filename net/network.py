import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
from net.utils import WaveletTransform, InverseWaveletTransform

# -----------------------------------------------------------------------------
# 0. 显存优化版 Cross Attention（归一化的线性空间注意力）(保持不变)
# -----------------------------------------------------------------------------
class LightweightCrossAttention(nn.Module):
    """
    归一化 (Normalized) 的线性空间注意力
    这对于稳定训练至关重要
    """

    def __init__(self, dim, num_heads=4, qkv_bias=False, down_ratio=2, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.down_ratio = down_ratio

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.out_proj = nn.Conv2d(dim, dim, 1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 增加一个 epsilon 防止除以0
        self.eps = 1e-6

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape

        # 1. 降采样 key/value
        if self.down_ratio > 1:
            x_kv = F.avg_pool2d(x_kv, kernel_size=self.down_ratio, stride=self.down_ratio)

        # 2. 投影
        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        # 3. Reshape
        Hk, Wk = k.shape[2], k.shape[3]
        Nq = H * W
        Nk = Hk * Wk

        q = rearrange(q, 'b (h c) x y -> b h (x y) c', h=self.num_heads)  # (B, H_heads, Nq, C_head)
        k = rearrange(k, 'b (h c) x y -> b h (x y) c', h=self.num_heads)  # (B, H_heads, Nk, C_head)
        v = rearrange(v, 'b (h c) x y -> b h (x y) c', h=self.num_heads)  # (B, H_heads, Nk, C_head)

        # 4. 线性注意力核函数
        q_prime = F.elu(q) + 1.0  # (B, H_heads, Nq, C_head)
        k_prime = F.elu(k) + 1.0  # (B, H_heads, Nk, C_head)

        # 5. 改变计算顺序

        # 5.1 计算 K'^T @ V
        kv = torch.matmul(k_prime.transpose(-2, -1), v) * self.scale  # (B, H_heads, C_head, C_head)
        kv = self.attn_drop(kv)

        # 5.2 计算 Q' @ (K'^T V) -> 未归一化的输出
        out_unnormalized = torch.matmul(q_prime, kv)  # (B, H_heads, Nq, C_head)

        # 6. --- (新增) 归一化步骤 ---
        k_prime_T = k_prime.transpose(-2, -1)
        k_sum = torch.sum(k_prime_T, dim=-1, keepdim=True)
        denominator = torch.matmul(q_prime, k_sum)

        out = out_unnormalized / (denominator + self.eps)
        # --- 归一化结束 ---

        # 7. Reshape back
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=H, y=W)

        # 8. Final projection
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# -----------------------------------------------------------------------------
# 1. Encoder 模块（显存友好）(保持不变)
# -----------------------------------------------------------------------------
class SimpleEncoder(nn.Module):
    def __init__(self, in_ch, embed_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_ch if i == 0 else embed_dim, embed_dim, 3, 1, 1),
                nn.GroupNorm(4, embed_dim),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class MLPAttentionV2(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        inter = max(16, embed_dim // 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, inter, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x): return self.mlp(x)


class DisentanglementEncoderV3(nn.Module):
    def __init__(self, in_ch, embed_dim, num_layers=2):
        super().__init__()
        self.encoder = SimpleEncoder(in_ch, embed_dim, num_layers)
        self.mlp = MLPAttentionV2(embed_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.GroupNorm(4, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f = self.encoder(x)
        M = self.mlp(f)
        fr = M * f
        fu = self.refine(f - fr)
        return fr, fu


# --- (新增) 轻量级解码器模块 --- (保持不变)
class SimpleDecoder(nn.Module):
    """ 替换 1x1 Conv，增强重建能力 """

    def __init__(self, in_ch, out_ch, num_layers=1):
        super().__init__()
        layers = []

        # 中间层
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.ReLU(inplace=True)
            ))

        # 输出层 (1x1 Conv 调整通道)
        layers.append(nn.Conv2d(in_ch, out_ch, 1))

        self.decoder = nn.Sequential(*layers)

        # (可选) 残差连接
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # 使用残差连接：输入x的投影 + 块的输出
        return self.proj(x) + self.decoder(x)


# -----------------------------------------------------------------------------
# 2. 子网（(修改：使用 LightweightCrossAttention)）
# -----------------------------------------------------------------------------
class DisentangleFusionSubnet(nn.Module):
    def __init__(self, hsi_ch, msi_channels, embed_dim=48, heads=1):  # heads 参数现在会传递给 LightweightCrossAttention
        super().__init__()
        self.swt = WaveletTransform('swt')
        self.iswt = InverseWaveletTransform('swt')

        self.enc_hsi_ll = DisentanglementEncoderV3(hsi_ch, embed_dim)
        self.enc_msi_ll = DisentanglementEncoderV3(msi_channels, embed_dim)
        self.enc_hsi_hf = DisentanglementEncoderV3(hsi_ch, embed_dim)
        self.enc_msi_hf = DisentanglementEncoderV3(msi_channels, embed_dim)

        # --- (修改) 使用 LightweightCrossAttention 替代 SimpleFusion ---
        self.cross_ll = LightweightCrossAttention(embed_dim, num_heads=heads)
        self.cross_hf = LightweightCrossAttention(embed_dim, num_heads=heads)

        self.adj_ll = SimpleDecoder(embed_dim, hsi_ch)
        self.adj_hf = SimpleDecoder(embed_dim, hsi_ch)

        # --- (修改) 辅助融合模块也使用 LightweightCrossAttention ---
        self.cross_aux_ll = LightweightCrossAttention(embed_dim, num_heads=heads)
        self.cross_aux_hf = LightweightCrossAttention(embed_dim, num_heads=heads)

        self.adj_ll_aux = SimpleDecoder(embed_dim, msi_channels)
        self.adj_hf_aux = SimpleDecoder(embed_dim, msi_channels)

    def forward(self, lrhsi, hrmsi):
        B = lrhsi.size(0)
        hsi_ll, hsi_lh, hsi_hl, hsi_hh = self.swt(lrhsi)
        msi_ll, msi_lh, msi_hl, msi_hh = self.swt(hrmsi)

        hsi_ll_r, hsi_ll_u = self.enc_hsi_ll(hsi_ll)
        msi_ll_r, msi_ll_u = self.enc_msi_ll(msi_ll)

        hsi_lh_r, hsi_lh_u = self.enc_hsi_hf(hsi_lh)
        hsi_hl_r, hsi_hl_u = self.enc_hsi_hf(hsi_hl)
        hsi_hh_r, hsi_hh_u = self.enc_hsi_hf(hsi_hh)

        msi_lh_r, msi_lh_u = self.enc_msi_hf(msi_lh)
        msi_hl_r, msi_hl_u = self.enc_msi_hf(msi_hl)
        msi_hh_r, msi_hh_u = self.enc_msi_hf(msi_hh)

        # 主路径融合 (H = H_r + Fusion(M_u, H_r))
        fused_ll = hsi_ll_r + self.cross_ll(msi_ll_u, hsi_ll_r)
        fused_lh = hsi_lh_r + self.cross_hf(msi_lh_u, hsi_lh_r)
        fused_hl = hsi_hl_r + self.cross_hf(msi_hl_u, hsi_hl_r)
        fused_hh = hsi_hh_r + self.cross_hf(msi_hh_u, hsi_hh_r)

        decoded_ll = self.adj_ll(fused_ll)
        decoded_lh = self.adj_hf(fused_lh)
        decoded_hl = self.adj_hf(fused_hl)
        decoded_hh = self.adj_hf(fused_hh)
        pred_hrhsi = self.iswt(decoded_ll, decoded_lh, decoded_hl, decoded_hh)

        # --- (修改) 辅助路径融合 ---
        fused_aux_ll = msi_ll_r + self.cross_aux_ll(hsi_ll_u, msi_ll_r)
        fused_aux_lh = msi_lh_r + self.cross_aux_hf(hsi_lh_u, msi_lh_r)
        fused_aux_hl = msi_hl_r + self.cross_aux_hf(hsi_hl_u, msi_hl_r)
        fused_aux_hh = msi_hh_r + self.cross_aux_hf(hsi_hh_u, msi_hh_r)

        aux_dec_ll = self.adj_ll_aux(fused_aux_ll)
        aux_dec_lh = self.adj_hf_aux(fused_aux_lh)
        aux_dec_hl = self.adj_hf_aux(fused_aux_hl)
        aux_dec_hh = self.adj_hf_aux(fused_aux_hh)

        pred_lrmsi = self.iswt(aux_dec_ll, aux_dec_lh, aux_dec_hl, aux_dec_hh)

        # --- (修改) 评估模式下返回 pred_hrhsi 和 pred_lrmsi ---
        if not self.training:
            return pred_hrhsi, pred_lrmsi

        # 训练模式返回所有
        return pred_hrhsi, pred_lrmsi, hsi_ll_r, hsi_ll_u, msi_ll_r, msi_ll_u


# -----------------------------------------------------------------------------
# 3. 主网络 (修改：调整评估模式的返回)
# -----------------------------------------------------------------------------
class GroupedDisentangleFusionNet(nn.Module):
    def __init__(self, hsi_group_channels: List[int], msi_channels: int, embed_dim=48):
        super().__init__()
        self.groups = hsi_group_channels
        self.subnets = nn.ModuleList([
            DisentangleFusionSubnet(g, msi_channels, embed_dim) for g in hsi_group_channels
        ])

    def forward(self, lrhsi, hrmsi):
        lr_groups = torch.split(lrhsi, self.groups, dim=1)

        if not self.training:
            # --- (修改) 评估模式下的返回 ---
            preds_hrhsi = []
            preds_lrmsi = []

            for i, sub in enumerate(self.subnets):
                # 评估模式下 sub 返回 (pred_hrhsi, pred_lrmsi)
                pred_h, pred_m = sub(lr_groups[i], hrmsi)
                preds_hrhsi.append(pred_h)
                preds_lrmsi.append(pred_m)

            # 组合 HSI 预测
            pred_hrhsi = torch.cat(preds_hrhsi, dim=1)
            # 平均 MSI 预测
            avg_pred_lrmsi = torch.stack(preds_lrmsi).mean(0)

            # 返回与训练时格式一致的 6 元组
            return pred_hrhsi, avg_pred_lrmsi, None, None, None, None

        # --- 训练模式 (保持不变) ---
        outs = [sub(lr_groups[i], hrmsi) for i, sub in enumerate(self.subnets)]
        pred_hrhsi = torch.cat([o[0] for o in outs], dim=1)
        avg_pred_lrmsi = torch.stack([o[1] for o in outs]).mean(0)
        avg_hsi_ll_r = torch.stack([o[2] for o in outs]).mean(0)
        avg_hsi_ll_u = torch.stack([o[3] for o in outs]).mean(0)
        avg_msi_ll_r = torch.stack([o[4] for o in outs]).mean(0)
        avg_msi_ll_u = torch.stack([o[5] for o in outs]).mean(0)
        return pred_hrhsi, avg_pred_lrmsi, avg_hsi_ll_r, avg_hsi_ll_u, avg_msi_ll_r, avg_msi_ll_u


# -----------------------------------------------------------------------------
# 4. 测试 (保持不变)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 确保在 CUDA 可用时使用
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, H, W = 2, 64, 64
    msi_c = 4
    hsi_group_channels = [35, 35, 32]

    # 假设 lrhsi 和 hrmsi 具有相同的空间分辨率 (例如 64x64)
    xh = torch.randn(B, sum(hsi_group_channels), H, W).to(device)
    xm = torch.randn(B, msi_c, H, W).to(device)

    model = GroupedDisentangleFusionNet(hsi_group_channels, msi_c, embed_dim=48).to(device)

    model.train()
    out = model(xh, xm)
    print("Train:", [o.shape if isinstance(o, torch.Tensor) else None for o in out])

    model.eval()
    with torch.no_grad():
        out_eval = model(xh, xm)
    print("Eval (HRHSI, LRMSI):", [o.shape if isinstance(o, torch.Tensor) else None for o in out_eval])
