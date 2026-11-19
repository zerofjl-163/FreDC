import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 新增的 SAM Loss 定义 ---
class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, preds, targets):
        # 确保预测和目标的形状是一致的
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape"

        # 增加一个小的 epsilon 来防止除以零
        epsilon = 1e-12

        dot_product = (preds * targets).sum(dim=1)
        preds_norm = preds.norm(dim=1)
        target_norm = targets.norm(dim=1)

        # 计算余弦相似度并限制在[-1, 1]范围内
        cos_sim = (dot_product + epsilon) / (preds_norm * target_norm + epsilon)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # 计算角度
        sam_score = torch.acos(cos_sim)
        return torch.mean(sam_score)

class SADLoss(nn.Module):
    def __init__(self, redu=True):
        super(SADLoss, self).__init__()
        self.redu = redu

    def forward(self, preds, targets):
        score = F.cosine_similarity(preds, targets, dim=1)
        score = torch.clamp(score, -1, 1).acos()
        if self.redu:
            loss = torch.mean(score)
        else:
            loss = torch.sum(score)
        return loss
