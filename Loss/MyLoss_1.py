import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM


class OFVLoss(nn.Module):
    def __init__(self,
                 alpha_pixel=10.0,
                 alpha_grad=1.0,
                 alpha_ssim=0.1,
                 alpha_temp=10.0):
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_grad = alpha_grad
        self.alpha_ssim = alpha_ssim
        self.alpha_temp = alpha_temp
        self.l1_loss = nn.L1Loss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=11)
        self.grid_cache = None
        self.pixel_weight_param = nn.Parameter(torch.tensor(0.0))

    def gradient(self, x):
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).float().unsqueeze(0).unsqueeze(
            0) / 4.0
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).float().unsqueeze(0).unsqueeze(
            0) / 4.0
        grad_x = F.conv2d(x, kernel_x, padding=1)
        grad_y = F.conv2d(x, kernel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def flow_warp_fast(self, x, flow):
        B, C, H, W = x.size()
        if self.grid_cache is None or self.grid_cache.size(1) != H or self.grid_cache.size(
                2) != W or self.grid_cache.device != x.device:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(0, H, device=x.device, dtype=x.dtype),
                torch.arange(0, W, device=x.device, dtype=x.dtype),
                indexing='ij'
            )
            self.grid_cache = torch.stack((grid_x, grid_y), 2).unsqueeze(0)

        grid = self.grid_cache.expand(B, -1, -1, -1)
        vgrid = grid + flow.permute(0, 2, 3, 1)
        vgrid_norm = torch.empty_like(vgrid)
        vgrid_norm[..., 0] = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
        vgrid_norm[..., 1] = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
        return F.grid_sample(x, vgrid_norm, mode='bilinear', padding_mode='reflection', align_corners=True)

    def forward(self, fused, ir, vi, flow_vi):
        B, T, C, H, W = fused.shape
        fused_2d = fused.reshape(B * T, C, H, W)
        ir_2d = ir.reshape(B * T, C, H, W)
        vi_2d = vi.reshape(B * T, C, H, W)
        grad_ir = self.gradient(ir_2d)
        grad_vi = self.gradient(vi_2d)
        grad_diff = grad_ir - grad_vi
        batch_std = torch.std(grad_diff.view(B * T, -1), dim=1, keepdim=True).view(B * T, 1, 1, 1)
        adaptive_scale = 1.0 / (batch_std + 1e-6)
        saliency_map = torch.sigmoid(grad_diff * adaptive_scale)

        target_pixel_adaptive = saliency_map * ir_2d + (1 - saliency_map) * vi_2d
        target_pixel_max = torch.max(ir_2d, vi_2d)

        loss_pixel_adaptive = self.l1_loss(fused_2d, target_pixel_adaptive)
        loss_pixel_max_val = self.l1_loss(fused_2d, target_pixel_max)
        w_max = torch.sigmoid(self.pixel_weight_param)
        loss_int = (1.0 - w_max) * loss_pixel_adaptive + w_max * loss_pixel_max_val

        target_grad = torch.max(grad_ir, grad_vi)
        loss_grad = self.l1_loss(self.gradient(fused_2d), target_grad)

        loss_ssim = 1 - self.ssim(fused_2d, target_pixel_adaptive)

        loss_temp = 0.0
        if T > 1:
            for t in range(1, T):
                f_curr = fused[:, t]
                f_prev = fused[:, t - 1]
                flow = flow_vi[:, t - 1]
                f_prev_warped = self.flow_warp_fast(f_prev, flow)
                loss_temp += self.l1_loss(f_curr, f_prev_warped)
            loss_temp /= (T - 1)

        total_loss = (self.alpha_pixel * loss_int +
                      self.alpha_grad * loss_grad +
                      self.alpha_ssim * loss_ssim +
                      self.alpha_temp * loss_temp)

        return total_loss, loss_int, loss_grad, loss_ssim, loss_temp
