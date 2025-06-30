"""
    Not used.
    Use `from fused_ssim import fused_ssim`.
"""

import torch
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except Exception:
    pass


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


def fast_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
