# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2025/9/5 10:36
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""


import datetime
import os
import yaml
import shutil
import csv

import torch
import torch.optim as optim
import requests

from args import parse_args

from pprint import pprint
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from utils.get_data import get_data_VS
from utils.early_stopping import EarlyStopping
from eva import evaluation_main
from utils.CAdamW import CAdamW

from Models.OFVFusion import OFVFusion
from Loss.MyLoss_1 import OFVLoss

from train import train
from val import val
from test import test, test_VS

from inference_video import inference_main




import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Initializing zero-element tensors is a no-op*")


files_to_save = ["./args.py", "./main.py",
                 "./train.py", "./val.py", "./test.py",
                 "./Models/OFVFusion.py", "./Loss/MyLoss_1.py",
                 "./utils/__init__.py", "./utils/get_data.py",
                 "./utils/CBAM.py", "./utils/show_plt.py",]

def main(device, args):
    os.makedirs(args.epoch_save_path, exist_ok=True)  # 用于保存每个epoch内输出的图片

    model = OFVFusion().to(device)

    # 光流 T-1 帧，通道为2
    flow_ir = torch.randn(1, 5, 2, 224, 224).to(device)
    flow_vi = torch.randn(1, 5, 2, 224, 224).to(device)

    from thop import profile
    VS_ir = torch.randn(1, 6, 1, 224, 224).to(device)
    VS_vi = torch.randn(1, 6, 1, 224, 224).to(device)
    flops, params = profile(model, inputs=(VS_ir, VS_vi, flow_ir, flow_vi), )
    gflops = flops / 1e9  # 转换为 GFLOPs
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f} G")
    print(f"Params: {params / 1e3:.2f} K")

    criterion = OFVLoss().to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.learning_rate, weight_decay=1e-2)
    # optimizer = CAdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)

    # 学习率下降策略
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.lr_patience,
        factor=args.lr_factor,
        # verbose=True,  # 打印学习率变化
        min_lr=1e-8,  # 学习率下限
        threshold=1e-3
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=args.T_0,  # 第一个周期 10 个 epoch
    #     T_mult=args.T_mult,  # 后续周期长度翻倍
    #     eta_min=args.eta_min)

    with open(os.path.join(args.hyperparameters_path, "hyperparameters.yaml"), "a") as f:
        f.write("\n")
        f.write("gflops : " + str(gflops))
        f.write('\n')
        f.write("params(k) : " + str(params / 1e3))
        f.write('\n')

    start_epoch = args.start_epoch
    best_loss = float("inf")

    if (not args.NEW) & bool(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    train_dataset = get_data_VS(args, mode="train")
    val_dataset = get_data_VS(args, mode="val")
    test_dataset = get_data_VS(args, mode="test")

    train_indices = random.sample(range(len(train_dataset)), len(train_dataset) // args.indices)
    val_indices = random.sample(range(len(val_dataset)), len(val_dataset) // args.indices)

    train_dataset_subset = Subset(train_dataset, train_indices)
    val_dataset_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    print('Datasets done, train:{}, val:{}, test:{}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    print('DataLoader done, train:{}, val:{}, test:{}'.format(len(train_loader), len(val_loader), len(test_loader)))

    # early_stopping
    early_stopping = EarlyStopping(args.log_dir, args.early_stopping_patience, verbose=True, delta=1e-3)

    train_losses, val_losses = [], []
    scaler = torch.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(args, model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler)
        val_loss = val(args, model, val_loader, criterion, device, epoch)

        if hasattr(criterion, "get_weights"):
            weights = criterion.get_weights()
            print(f"Epoch {epoch} Weights:", {k: f"{v:.4f}" for k, v in weights.items()})

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # # 更新学习率
        scheduler.step(val_loss)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            # 'optimizer_state_dict': optimizer.state_dict()
        }

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, os.path.join(args.log_dir, "best_checkpoint.pth"))

        # 打印信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} "
              f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, LR: {current_lr:.8f}")

        early_stopping(val_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

    with open(os.path.join(args.log_dir, "train_loss.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in train_losses])
    with open(os.path.join(args.log_dir, "val_loss.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[value] for value in val_losses])

    segline = "=" * 40
    if not args.resume:
        train_run_timestamp = datetime.datetime.now()
        print(f"{segline}train_run_timestamp:\t\t{train_run_timestamp - timestamp}{segline}")

    if args.test:
        print('testing on test set')
        test(args, model, device)

    if args.test_VS:
        print('testing on test_VS set')
        test_VS(args, model, test_loader, device)

    if args.metrics:
        print('evaluation_main running')
        eva = evaluation_main(args)
        eva.evaluation_main(args)

        # eva.evaluation_Video(args)

        if not args.resume:
            evaluation_end_timestamp = datetime.datetime.now()
            print(f"{segline}evaluation_run_timestamp:\t{evaluation_end_timestamp - evaluation_end_timestamp}{segline}")

    # -------------------- 释放显存 --------------------
    # print("释放 GPU 显存...")
    # del model, optimizer, criterion
    # del train_loader, val_loader, test_loader
    # del train_dataset, val_dataset, test_dataset
    # torch.cuda.empty_cache()
    # # 生成推理视频
    if args.save_comparison_video_num:
        inference_main(args)


if __name__ == '__main__':
    timestamp = datetime.datetime.now()

    timestamp_date = timestamp.strftime("%Y%m%d")
    timestamp_time = timestamp.strftime("%H%M%S")

    # timestamp_date = '20251125'
    # timestamp_time = '102051'

    args = parse_args(timestamp_date, timestamp_time)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    init_seeds(torch.cuda.is_available(), args.seed)
    pprint(vars(args), sort_dicts=False)

    # 保存超参数信息到日志文件
    with open(os.path.join(args.hyperparameters_path, "hyperparameters.yaml"), "w", encoding='utf-8') as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)

    # 保存文件到日志目录
    os.makedirs(os.path.join(args.hyperparameters_path, "py_files"), exist_ok=True)
    for file in files_to_save:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(args.hyperparameters_path, "py_files"))
            print(f"已将 {file} 保存到 {args.hyperparameters_path}")
        else:
            print(f"未找到文件 {file}，无法保存")

    try:
        main(device, args)

    finally:
        endtimestamp = datetime.datetime.now()
        print(f"starttime: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"endtime: {endtimestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"elapsed time: {endtimestamp - timestamp}")
        with open(os.path.join(args.log_dir, "hyperparameters.yaml"), "a") as f:
            f.write("\n")
            f.write(f"starttime: {timestamp}\n")
            f.write(f"endtime: {endtimestamp}\n")
            f.write(f"elapsed time: {endtimestamp - timestamp}\n")

        if args.system_name == "Linux":
            if args.shutdown:
                headers = {
                    "Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjI5MzA4NSwidXVpZCI6ImQ4YmJjMzg5LTk3NzctNDA1My04Z"
                                     "GYxLTQ4NDdiOGQyNTE2MCIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI"
                                     "6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.cYf-EWJd8Ac360IBe9VCcg7xSq4"
                                     "NSxe6kjxHGalnMpKHdw5nO7crueKNQmlbJqUmCEHQeSOL9qbouHqltDktlA"}
                resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                                     json={
                                         "title": "title",
                                         "name": "训练完成",
                                         # "content": "eg. 训练完成"
                                     }, headers=headers)
                print(resp.content.decode())
                os.system("/usr/bin/shutdown")


