"""
    # Copyright (C) 2023, Inria
    # GRAPHDECO research group, https://team.inria.fr/graphdeco
    # All rights reserved.
    #
    # This software is free for non-commercial, research and evaluation use
    # under the terms of the LICENSE.md file.
    #
    # For inquiries contact  george.drettakis@inria.fr
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except Exception:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2


class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def calc_gaussian_1d(window_size: int,
                     sigma: float,
                     device: torch.device | None = None):
    assert (window_size % 2 == 1)
    mu = window_size // 2
    k = -0.5 / (sigma ** 2)
    xs = torch.arange(window_size, dtype=torch.float32, device=device)
    gauss = torch.exp(k * ((xs - mu) ** 2))
    return gauss / gauss.sum()


def create_window(window_size: int,
                  num_channels: int,
                  sigma: float = 1.5,
                  device: torch.device | None = None):
    window_1d = calc_gaussian_1d(
        window_size=window_size,
        sigma=sigma,
        device=device)
    window_2d = torch.outer(window_1d, window_1d)
    window = window_2d.expand(num_channels, 1, -1, -1).contiguous()
    return window


def ssim(img1: torch.Tensor,
         img2: torch.Tensor,
         window_size: int = 11,
         size_average: bool = True):
    device = img1.device
    num_channels = img1.size(-3)
    window = create_window(
        window_size=window_size,
        num_channels=num_channels,
        device=device)

    # if img1.is_cuda:
    #     window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(
        img1=img1,
        img2=img2,
        window=window,
        window_size=window_size,
        channel=num_channels,
        size_average=size_average)


def _ssim(img1: torch.Tensor,
          img2: torch.Tensor,
          window: torch.Tensor,
          window_size: int,
          channel: int,
          size_average: bool = True):
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


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
