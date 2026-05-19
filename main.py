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

files_to_save = ["./args.py", "./main.py",
                 "./train.py", "./val.py", "./test.py",
                 "./Models/OFVFusion.py", "./Loss/MyLoss_1.py",
                 "./utils/__init__.py", "./utils/get_data.py",
                 "./utils/CBAM.py", "./utils/show_plt.py",]

def main(device, args):
    os.makedirs(args.epoch_save_path, exist_ok=True) 

    model = OFVFusion().to(device)
    flow_ir = torch.randn(1, 5, 2, 224, 224).to(device)
    flow_vi = torch.randn(1, 5, 2, 224, 224).to(device)

    from thop import profile
    VS_ir = torch.randn(1, 6, 1, 224, 224).to(device)
    VS_vi = torch.randn(1, 6, 1, 224, 224).to(device)
    flops, params = profile(model, inputs=(VS_ir, VS_vi, flow_ir, flow_vi), )
    gflops = flops / 1e9  
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f} G")
    print(f"Params: {params / 1e3:.2f} K")

    criterion = OFVLoss().to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.learning_rate, weight_decay=1e-2)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.lr_patience,
        factor=args.lr_factor,
        min_lr=1e-8,  
        threshold=1e-3
    )

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
        scheduler.step(val_loss)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
        }

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, os.path.join(args.log_dir, "best_checkpoint.pth"))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} "
              f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, LR: {current_lr:.8f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break 

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

        if not args.resume:
            evaluation_end_timestamp = datetime.datetime.now()
            print(f"{segline}evaluation_run_timestamp:\t{evaluation_end_timestamp - evaluation_end_timestamp}{segline}")
 
    if args.save_comparison_video_num:
        inference_main(args)


if __name__ == '__main__':
    timestamp = datetime.datetime.now()
    timestamp_date = timestamp.strftime("%Y%m%d")
    timestamp_time = timestamp.strftime("%H%M%S")
    args = parse_args(timestamp_date, timestamp_time)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    init_seeds(torch.cuda.is_available(), args.seed)
    pprint(vars(args), sort_dicts=False)
    with open(os.path.join(args.hyperparameters_path, "hyperparameters.yaml"), "w", encoding='utf-8') as f:
        yaml.dump(args, f, default_flow_style=False, allow_unicode=True)
    os.makedirs(os.path.join(args.hyperparameters_path, "py_files"), exist_ok=True)
    for file in files_to_save:
        shutil.copy(file, os.path.join(args.hyperparameters_path, "py_files"))
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
                os.system("/usr/bin/shutdown")


