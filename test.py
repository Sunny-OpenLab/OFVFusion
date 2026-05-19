# -*-coding:utf-8 -*-

"""
# File       : test.py
# Time       ：2025/4/14 10:49
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""
import os
import time
import re
import datetime
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from pprint import pprint

from Models.OFVFusion import OFVFusion

from args import parse_args
from utils.get_data import get_test_data_3D
from eva import evaluation_main
from utils import *
import utils
from args import DATASETNAME_VS

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Initializing zero-element tensors is a no-op*")
warnings.filterwarnings("ignore", category=UserWarning, message="An output with one or more elements was resized*")

def test(args, model, device):
    use_amp = args.float16
    for dataset_name in os.listdir(args.test_dataset_path_VS):
        test_folder = os.path.join(args.test_dataset_path_VS, dataset_name)
        test_dataset = get_test_data_3D(test_folder)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

        model.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_network.pth')), strict=False)
        model.eval()
        # model.to(device)

        test_tqdm = tqdm(test_loader, total=len(test_loader), leave=False)
        elapsed_time_list = []

        with torch.no_grad():
            for i, (VS_vi, VS_vi_y, VS_cr, VS_cb, VS_ir, flow_ir, flow_vi, name) in enumerate(test_tqdm):
                VS_vi_y = VS_vi_y.to(device)
                VS_cr = VS_cr.to(device)
                VS_cb = VS_cb.to(device)
                VS_ir = VS_ir.to(device)
                flow_ir = flow_ir.to(device)
                flow_vi = flow_vi.to(device)
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        start_time = time.time()
                        # fused_output, _, _, _, _, _ = model(VS_ir, VS_vi_y, flow_ir.half(), flow_vi.half())
                        fused_output, *_ = model(VS_ir, VS_vi_y, flow_ir, flow_vi)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        elapsed_time_list.append(elapsed_time)
                else:
                    start_time = time.time()
                    # fused_output, _, _, _, _, _ = model(VS_ir, VS_vi_y, flow_ir.half(), flow_vi.half())
                    fused_output, *_ = model(VS_ir, VS_vi_y, flow_ir, flow_vi)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    elapsed_time_list.append(elapsed_time)

                # rgb_fused_image = utils.Y_Cr_Cb2RGB(fused_output[0, -1, :, :, :], VS_cr[0, -1, :, :, :], VS_cb[0, -1, :, :, :])
                rgb_fused_image = utils.Y_Cr_Cb2RGB(fused_output[0, -1, 0, :, :], VS_cr[0, -1, 0, :, :],
                                                    VS_cb[0, -1, 0, :, :])
                rgb_fused_image = transforms.ToPILImage(mode='RGB')(rgb_fused_image)

                # if not os.path.exists(os.path.join(args.test_save_path, "video")):
                #     os.makedirs(os.path.join(args.test_save_path, "video"))
                video_save_path = os.path.join(args.test_save_path, os.path.splitext(dataset_name)[0])
                os.makedirs(video_save_path, exist_ok=True)
                rgb_fused_image.save(f'{video_save_path}/{name[0]}')


#######################################################################################
#################################test_VS#############################################
#################################test_VS#############################################
def test_VS(args, model, test_loader, device):
    """这个测试的是视频x序列"""
    use_amp = args.float16

    model.load_state_dict(
        torch.load(os.path.join(args.log_dir, 'best_network.pth')),
        strict=False
    )
    model.eval()
    model.to(device)

    elapsed_time_list = []

    with torch.no_grad():
        test_tqdm = tqdm(test_loader, total=len(test_loader),
                         desc="test", leave=False, delay=1)

        for i, (VS_vi, VS_vi_y, VS_cr, VS_cb,
                VS_ir, flow_ir, flow_vi, full_path) in enumerate(test_tqdm):

            # ===============================
            # 1. 数据放到 GPU
            # ===============================
            VS_vi_y = VS_vi_y.to(device)
            VS_cr = VS_cr.to(device)
            VS_cb = VS_cb.to(device)
            VS_ir = VS_ir.to(device)
            flow_ir = flow_ir.to(device)
            flow_vi = flow_vi.to(device)

            # ===============================
            # 2. 前向推理
            # ===============================
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    start_time = time.time()
                    fused_output, *_ = model(
                        VS_ir, VS_vi_y,
                        flow_ir.half(), flow_vi.half()
                    )
                    elapsed_time_list.append(time.time() - start_time)
            else:
                start_time = time.time()
                fused_output, *_ = model(VS_ir, VS_vi_y, flow_ir, flow_vi)
                elapsed_time_list.append(time.time() - start_time)

            # =====================================================
            # 3. ⭐ 统一整理 full_path → seq_paths
            # =====================================================
            # full_path 可能是：
            # 1) [ ['a6.jpg', 'a7.jpg', ...] ]
            # 2) [ ('a6.jpg',), ('a7.jpg',), ... ]
            # 3) ['a6.jpg'] （异常情况）

            if isinstance(full_path[0], (list, tuple)):
                # 情况 1 或 2
                if len(full_path) == 1 and isinstance(full_path[0][0], str):
                    # [ ['a6.jpg', 'a7.jpg', ...] ]
                    seq_paths = list(full_path[0])
                else:
                    # [ ('a6.jpg',), ('a7.jpg',), ... ]
                    seq_paths = [p[0] for p in full_path]
            else:
                # ['a6.jpg'] 这种退化情况
                seq_paths = list(full_path)

            T = fused_output.shape[1]

            # 防御式检查
            if len(seq_paths) != T:
                raise RuntimeError(
                    f"时间维不一致: fused_output={T}, seq_paths={len(seq_paths)}\n"
                    f"seq_paths={seq_paths}"
                )

            # ===============================
            # 4. 解析序列名（用第 0 帧）
            # ===============================
            first_path = seq_paths[0]
            dir_name = os.path.basename(os.path.dirname(first_path))

            if dir_name and dir_name not in ['ir', 'vi', 'test', 'train', 'val']:
                seq_name = dir_name
            else:
                parts = first_path.replace('\\', '/').split('/')
                for p in reversed(parts):
                    if '_' in p and p not in ['ir', 'vi', 'test', 'train', 'val']:
                        seq_name = p
                        break
                else:
                    seq_name = "unknown_sequence"

            # ===============================
            # 5. 创建保存目录
            # ===============================
            video_save_path = os.path.join(
                args.test_save_path, DATASETNAME_VS, seq_name
            )
            os.makedirs(video_save_path, exist_ok=True)

            # ===============================
            # 6. ⭐ 逐帧同名保存
            # ===============================
            for t in range(T):
                fused_frame = fused_output[0, t, 0].float()
                cr_frame = VS_cr[0, t, 0].float()
                cb_frame = VS_cb[0, t, 0].float()

                rgb_fused_image = utils.Y_Cr_Cb2RGB(
                    fused_frame, cr_frame, cb_frame
                )
                rgb_fused_image = transforms.ToPILImage(
                    mode='RGB'
                )(rgb_fused_image)

                frame_filename = os.path.basename(seq_paths[t])
                rgb_fused_image.save(
                    os.path.join(video_save_path, frame_filename)
                )
#################################test_VS#############################################
#######################################################################################

if __name__ == '__main__':
    # 输入正确的log时间
    timestamp_date = '20251216'
    timestamp_time = '083421'

    args = parse_args(timestamp_date, timestamp_time)

    pprint(vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OFVFusion().to(device)

    print("running test……")
    test(args, model, device)

    # #
    # from utils.get_data import get_data_VS
    # test_dataset = get_data_VS(args, mode="test")
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers,
    #                          pin_memory=True)
    # print("running test_VS……")
    # test_VS(args, model, test_loader, device)
    #
    # print("running eva……")
    # eva = evaluation_main(args)
    #
    # print("running eva——video……")
    # eva.evaluation_Video(args)


