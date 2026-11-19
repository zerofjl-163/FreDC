import torch
import torch.nn.functional as F


# https://blog.csdn.net/qq_49729636/article/details/134502721
# https://zhuanlan.zhihu.com/p/622244511?utm_id=0


def psnr(reference, pred, data_range=None):
    base = 10.0
    b, c, h, w = pred.shape
    preds = pred.reshape(b, c, h * w)
    reference = reference.reshape(b, c, h * w)
    diff = preds - reference
    sum_squared_error = torch.sum(diff * diff, dim=2)

    data_range = torch.amax(reference, dim=2) - torch.amin(reference, dim=2)
    n_obs = torch.tensor(reference.size(), device=reference.device)[[2]].prod()
    n_obs = n_obs.expand_as(sum_squared_error)

    psnr_base_e = -torch.log(torch.sqrt(sum_squared_error / n_obs))
    psnr_vals = psnr_base_e * (20 / torch.log(torch.tensor(base)))

    return torch.mean(psnr_vals).item()



def sam(reference, pred):
    """
    Compute the Spectral Angle Mapper (SAM) for a batch of images.

    :param reference: Ground truth image tensor, shape [B, C, H, W]
    :param pred: Processed image tensor, shape [B, C, H, W]
    :return: Average SAM value over the batch
    """
    dot_product = (pred * reference).sum(dim=1)
    preds_norm = pred.norm(dim=1)
    target_norm = reference.norm(dim=1)
    sam_score = torch.clamp((dot_product + 1e-12) / (preds_norm * target_norm + 1e-12), -1, 1).acos()
    sam_degree = torch.rad2deg(sam_score)
    return torch.mean(sam_degree).item()


def gaussian_kernel_2d(channel, kernel_size, sigma, dtype, device):
    """
    Create a 2D Gaussian kernel.
    """
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= kernel_size // 2

    g = coords ** 2
    g = (-g.unsqueeze(1) - g.unsqueeze(0)) / (2 * sigma ** 2)
    g = g.exp()

    g /= g.sum()
    return g.expand(channel, 1, kernel_size, kernel_size)


def ssim(preds, target, kernel_size=11, sigma=1.5):
    """
    Compute SSIM based on modified method.
    """
    data_range = max(preds.max() - preds.min(), target.max() - target.min())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    device = preds.device
    dtype = preds.dtype
    channel = preds.size(1)

    kernel = gaussian_kernel_2d(channel, kernel_size, sigma, dtype, device)

    pad_h = (kernel_size - 1) // 2
    pad_w = (kernel_size - 1) // 2

    preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")

    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))  # (5 * B, C, H, W)
    outputs = F.conv2d(input_list, kernel, groups=channel)
    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = output_list[2] - mu_pred_sq
    sigma_target_sq = output_list[3] - mu_target_sq
    sigma_pred_target = output_list[4] - mu_pred_target

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
    ssim_idx = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w]

    return torch.mean(ssim_idx).item()


def ergas(reference, pred, h=4):
    """
    Compute the ERGAS metric for a batch of images.

    :param reference: Ground truth image tensor, shape [B, C, H, W]
    :param pred: Processed image tensor, shape [B, C, H, W]
    :param h: Scale factor, typically the upscaling factor in super-resolution
    :return: ERGAS value
    """
    # Calculate MSE for each band and each image in the batch
    mse = torch.mean((reference - pred) ** 2, dim=[2, 3])

    # Calculate mean of the target images for each band
    mean_target = torch.mean(reference, dim=[2, 3])

    # Calculate ERGAS for each image in the batch
    ergas_per_band = 100 / h * torch.sqrt(mse / (mean_target ** 2))
    ergas = torch.sqrt(torch.mean(ergas_per_band, dim=1))

    # Return the mean ERGAS value over the batch
    return torch.mean(ergas).item()


def rmse(reference, pred):
    """计算RMSE"""
    return torch.sqrt(torch.mean((pred - reference) ** 2)).item()


def qnr(pred, lr_hsi, hr_msi, alpha=1, beta=1, p=1, q=1):
    '''
    Quality with No Reference metric
    "Multispectral and Panchromatic Data Fusion Assessment Without Reference", Photogrammetric Engineering & Remote Sensing 2008.
    https://github.com/Andrewwango/perspective-equivariant-imaging/blob/main/perspective_ei/multispectral/qnr.py

    :param pred:
    :param lr_hsi:
    :param hr_msi:
    :param alpha: weight for spectral quality, defaults to 1
    :param beta: weight for structural quality, defaults to 1
    :param p: power exponent for spectral D, defaults to 1
    :param q: power exponent for spatial D, defaults to 1
    :return: D_lambda (down), D_s (down), QNR (up)
    '''

    def quality_index(x, y):
        # Calculate the means
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        y_mean = torch.mean(y, dim=[2, 3], keepdim=True)

        # Calculate the variances
        x_var = torch.var(x, dim=[2, 3], keepdim=True, unbiased=False)
        y_var = torch.var(y, dim=[2, 3], keepdim=True, unbiased=False)

        # Calculate the covariance
        xy_cov = torch.mean((x - x_mean) * (y - y_mean), dim=[2, 3], keepdim=True)

        # Calculate Q index
        numerator = 4 * xy_cov * x_mean * y_mean
        denominator = (x_var + y_var) * (x_mean ** 2 + y_mean ** 2)

        # Avoid division by zero
        epsilon = 1e-10
        q_value = numerator / (denominator + epsilon)

        # Remove the singleton dimensions
        q_value = torch.squeeze(q_value)

        return q_value

    def spectral_distortion_index(pred, lr_hsi, p):
        _, n_bands, _, _ = pred.shape
        d_lambda_sum = 0.0
        for b1 in range(n_bands):
            for b2 in range(n_bands):
                if b1 == b2:
                    continue
                q_pred = quality_index(pred[:, b1:b1 + 1, :, :], pred[:, b2:b2 + 1, :, :])
                q_lr_hsi = quality_index(lr_hsi[:, b1:b1 + 1, :, :], lr_hsi[:, b2:b2 + 1, :, :])
                d_lambda_sum += torch.abs(q_pred - q_lr_hsi) ** p
        d_lambda = (d_lambda_sum / (n_bands * (n_bands - 1))) ** (1 / p)
        return d_lambda

    def spatial_distortion_index(pred, hr_msi, q):
        _, n_bands, _, _ = pred.shape
        d_s_sum = 0.0
        for b in range(n_bands):
            q_pred = quality_index(pred[:, b:b + 1, :, :], hr_msi[:, b:b + 1, :, :])
            d_s_sum += torch.abs(q_pred) ** q
        d_s = (d_s_sum / n_bands) ** (1 / q)
        return d_s

    d_lambda = spectral_distortion_index(pred, lr_hsi, p)
    d_s = spatial_distortion_index(pred, hr_msi, q)
    qnr_value = (1 - d_lambda) ** alpha * (1 - d_s) ** beta
    return d_lambda, d_s, qnr_value
