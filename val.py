# -*-coding:utf-8 -*-

"""
# File       : val.py
# Time       ：2025/4/14 10:49
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""
import torch
from tqdm import tqdm
from collections import OrderedDict
from utils.show_plt import *

def val(args, model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0
    use_amp = args.float16

    with torch.no_grad():
        eval_tqdm = tqdm(val_loader, total=len(val_loader), desc="eval", leave=False, delay=1)
        for i, (VS_vi, VS_vi_y, VS_cr, VS_cb, VS_ir, flow_ir, flow_vi, _) in enumerate(eval_tqdm):
            VS_ir, VS_vi_y = VS_ir.to(device), VS_vi_y.to(device)
            flow_ir = flow_ir.to(device)
            flow_vi = flow_vi.to(device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    fused_output, confidence_mask_seq, h_seq, c_seq, latent_ir_seq, latent_vi_seq, fused_seq, _ = model(VS_ir, VS_vi_y, flow_ir, flow_vi)
                    total_loss, loss_pixel, loss_grad, loss_ssim, loss_temp = criterion(fused_output[:,1:], VS_ir[:,1:], VS_vi_y[:,1:], flow_vi)
            else:
                fused_output, confidence_mask_seq, h_seq, c_seq, latent_ir_seq, latent_vi_seq, fused_seq, _ = model(VS_ir,
                                                                                                                 VS_vi_y,
                                                                                                                 flow_ir,
                                                                                                                 flow_vi)
                total_loss, loss_pixel, loss_grad, loss_ssim, loss_temp = criterion(fused_output[:, 1:], VS_ir[:, 1:],
                                                                                    VS_vi_y[:, 1:], flow_vi)

            running_loss += total_loss.item()
            eval_tqdm.set_postfix(OrderedDict([
                ('total_loss', f"{total_loss.item():.6f}"),
                ('loss_pixel', f"{loss_pixel.item():.6f}"),
                ('loss_grad', f"{loss_grad.item():.6f}"),
                ('loss_ssim', f"{loss_ssim.item():.6f}"),
                ('loss_temp', f"{loss_temp.item():.6f}"),
            ]))

        showplt_all(args, VS_ir, VS_vi_y, fused_output,
                    confidence_mask_seq,
                    h_seq, c_seq,
                    latent_ir_seq, latent_vi_seq,
                    fused_seq,
                    epoch, frames_to_show=3)
        return running_loss / len(val_loader)

