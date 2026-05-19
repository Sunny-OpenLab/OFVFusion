import torch
from tqdm import tqdm
from collections import OrderedDict


def train(args, model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler=None):
    model.train()
    running_loss = 0
    use_amp = args.float16 

    train_tqdm = tqdm(train_loader, total=len(train_loader), desc="Train", leave=False, delay=1)

    for i, (VS_vi, VS_vi_y, VS_cr, VS_cb, VS_ir, flow_ir, flow_vi, _) in enumerate(train_tqdm):
        VS_ir = VS_ir.to(device)
        VS_vi_y = VS_vi_y.to(device)
        flow_ir = flow_ir.to(device)
        flow_vi = flow_vi.to(device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                fused_output, *_ = model(VS_ir, VS_vi_y, flow_ir, flow_vi)
                total_loss, loss_pixel, loss_grad, loss_ssim, loss_temp = criterion(
                    fused_output[:,1:], VS_ir[:,1:], VS_vi_y[:,1:], flow_vi
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            fused_output, *_ = model(VS_ir, VS_vi_y, flow_ir, flow_vi)
            total_loss, loss_pixel, loss_grad, loss_ssim, loss_temp = criterion(
                fused_output[:,1:], VS_ir[:,1:], VS_vi_y[:,1:], flow_vi
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += total_loss.item()

        train_tqdm.set_postfix({
            'total_loss': f"{total_loss.item():.6f}",
            'loss_pixel': f"{loss_pixel.item():.6f}",
            'loss_grad': f"{loss_grad.item():.6f}",
            'loss_ssim': f"{loss_ssim.item():.6f}",
            'loss_temp': f"{loss_temp.item():.6f}",
        })

    return running_loss / len(train_loader)
