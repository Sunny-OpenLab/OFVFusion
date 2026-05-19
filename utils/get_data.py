# -*-coding:utf-8 -*-

"""
# File       : get_data.py
# Time       ：2025/4/14 14:34
# Author     ：Qiang42
# version    ：python 3.8
# Description：
"""
import os
import cv2
import tarfile
import torch
import numpy as np

from torchvision import transforms
from torch.utils import data
from PIL import Image

import utils

from args import DATASETNAME_VS

FRAME = 6

to_tensor = transforms.Compose([transforms.ToTensor()])

def normalize_0_1(img):
    # img: Tensor [C,H,W]
    min_val = img.min()
    max_val = img.max()
    return (img - min_val) / (max_val - min_val + 1e-8)

class get_data_VS(data.Dataset):
    def __init__(self, args, mode="train", transform=to_tensor):
        super().__init__()

        self.tar_path = args.dataset_path_VS
        self.transform = transform
        self.members = []

        dataset_name = DATASETNAME_VS

        with tarfile.open(self.tar_path, 'r') as tar_ref:
            if mode=="train":
                print('Loading train datasets...')
                data_dir = dataset_name + '/train'
            elif mode=="val":
                print('Loading val datasets...')
                data_dir = dataset_name + '/val'
            elif mode=="test":
                print('Loading test_video datasets...')
                data_dir = dataset_name + '/test'
            for member in tar_ref.getmembers():
                if member.isfile() and member.name.startswith(data_dir):
                    self.members.append(member)

        self.inf_members = [m for m in self.members if 'ir/' in m.name]
        self.vis_members = [m for m in self.members if 'vi/' in m.name]

        # self.inf_members.sort(key=lambda m: os.path.basename(m.name))
        # self.vis_members.sort(key=lambda m: os.path.basename(m.name))
        self.inf_members.sort(key=lambda m: m.name)
        self.vis_members.sort(key=lambda m: m.name)

    # =========  Farneback 光流计算 =========
    def _compute_farneback_flow(self, frames_np):
        """
        frames_np: numpy array [T, 1, H, W], range [0, 255], dtype=uint8
        return: torch tensor [T-1, 2, H, W]
        """
        T, _, H, W = frames_np.shape
        flow_list = []

        # Farneback 参数配置 (针对 333x173 分辨率优化)
        # winsize: 窗口越大越平滑，抗噪越强，但丢失细节。红外建议大一点。
        fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=21,  # 增大窗口以获得更平滑的运动场 (原15)
            iterations=3,
            poly_n=7,  # 7 比 5 更平滑
            poly_sigma=1.5,
            flags=0
        )

        for t in range(1, T):
            prev = frames_np[t - 1, 0]
            next = frames_np[t, 0]

            # 1. 高斯模糊：红外图噪声大，可以稍微加大核
            prev_blur = cv2.GaussianBlur(prev, (5, 5), 0)
            next_blur = cv2.GaussianBlur(next, (5, 5), 0)

            # 2. 计算光流
            flow = cv2.calcOpticalFlowFarneback(prev_blur, next_blur, None, **fb_params)

            # 3. 移除硬阈值截断 (flow[mask]=0)，改用简单的限幅或保留原始值
            # 神经网络通常能处理微小的背景噪声，硬截断反而会造成时序不连续。
            # 如果必须去噪，建议在 loss 计算时处理，或者仅对红外做轻微截断。

            # (可选) 简单的异常值去除：如果光流太大（比如瞬移），可能是计算错误
            flow[np.abs(flow) > 20] = 0

            flow_list.append(flow)

        flow_np = np.stack(flow_list, axis=0)  # [T-1, H, W, 2]
        flow_np = np.transpose(flow_np, (0, 3, 1, 2))  # [T-1, 2, H, W]

        # 添加极小的 epsilon 防止除0 (虽然这里没除法，但保持习惯)
        return torch.from_numpy(flow_np).float()

    def __getitem__(self, index):

        with tarfile.open(self.tar_path, 'r') as tar_ref:
            # 每个样本有FRAME帧，因此计算在成员列表中的起始索引
            start_idx = index * FRAME
            inf_images = []
            vis_images = []
            vis_names = []
            # 遍历FRAME帧图像
            for i in range(FRAME):
                inf_member = self.inf_members[start_idx + i]
                vis_member = self.vis_members[start_idx + i]

                inf_img = Image.open(tar_ref.extractfile(inf_member)).convert('L')
                vis_img = Image.open(tar_ref.extractfile(vis_member)).convert('RGB')

                inf_img = self.transform(inf_img)
                vis_img = self.transform(vis_img)

                inf_img = normalize_0_1(inf_img)
                vis_img = normalize_0_1(vis_img)

                inf_images.append(inf_img)
                vis_images.append(vis_img)

                vis_names.append(vis_member.name)  # ⭐ 核心

            # 将FRAME帧沿着时间轴(T)拼接
            # 注意：infrared图像为单通道，[1, FRAME, H, W]；visible图像为3通道，[3, FRAME, H, W]
            inf_images = torch.stack(inf_images, dim=1)
            vis_images = torch.stack(vis_images, dim=1)
            # 如果需要对可见光图像进一步处理，比如转为YCrCb通道，可调用工具函数
            vis_y_image, vis_cr_image, vis_cb_image = utils.RGB2Y_Cr_Cb(vis_images)

            # 重新排列维度，从 [C, T, H, W] -> [T, C, H, W]
            inf_images = inf_images.permute(1, 0, 2, 3)  # [T, C, H, W]
            vis_images = vis_images.permute(1, 0, 2, 3)  # [T, C, H, W]
            vis_y_image = vis_y_image.permute(1, 0, 2, 3)
            vis_cr_image = vis_cr_image.permute(1, 0, 2, 3)
            vis_cb_image = vis_cb_image.permute(1, 0, 2, 3)

            # return vis_images, vis_y_image, vis_cr_image, vis_cb_image, inf_images, os.path.basename(vis_member.name)
            # ============================================================
            # ✅ 计算光流（使用 IR 和 VIS-Y，两者都为 1 通道）
            # ============================================================

            ir_np = (inf_images.numpy() * 255).astype(np.uint8)  # [T,1,H,W]
            vis_y_np = (vis_y_image.numpy() * 255).astype(np.uint8)

            flow_ir = self._compute_farneback_flow(ir_np)  # [T-1, 2, H, W] # _compute_farneback_flow | _compute_lk_flow
            flow_vi = self._compute_farneback_flow(vis_y_np)                # _compute_farneback_flow | _compute_lk_flow

            return (
                vis_images, vis_y_image, vis_cr_image, vis_cb_image,
                inf_images,
                flow_ir, flow_vi,
                vis_names  # ⭐ 不再是单个 name
            )

    def __len__(self):
        return len(self.inf_members) // FRAME


def replicate_frames(image, T):
    """
    复制图像 T 次以扩展时间维度。
    """
    image_tensor = torch.unsqueeze(image, 0)  # 添加时间维度
    return image_tensor.repeat(T, 1, 1, 1)  # 复制 T 次

class get_test_data_3D(data.Dataset):
    """
    测试集 Dataset，支持伪视频（每张图像复制 T 次）
    保持原尺寸，不做 resize/pad
    """
    def __init__(self, test_folder, transform=to_tensor, T=FRAME):
        self.test_folder = test_folder
        self.transform = transform
        self.T = T
        self.members = []

        with tarfile.open(self.test_folder, 'r') as tar_ref:
            print('Loading test datasets...')
            for member in tar_ref.getmembers():
                if member.isfile():
                    self.members.append(member)

        # 分红外和可见光
        self.inf_members = [m for m in self.members if 'ir/' in m.name]
        self.vis_members = [m for m in self.members if 'vi/' in m.name]

        # 排序保证对齐
        self.inf_members.sort(key=lambda m: os.path.basename(m.name))
        self.vis_members.sort(key=lambda m: os.path.basename(m.name))

    def __getitem__(self, index):
        with tarfile.open(self.test_folder, 'r') as tar_ref:
            inf_member = self.inf_members[index]
            vis_member = self.vis_members[index]

            # 加载图像
            inf_image = Image.open(tar_ref.extractfile(inf_member)).convert('L')
            vis_image = Image.open(tar_ref.extractfile(vis_member)).convert('RGB')

            if self.transform:
                inf_image = self.transform(inf_image)
                vis_image = self.transform(vis_image)

            inf_image = normalize_0_1(inf_image)
            vis_image = normalize_0_1(vis_image)

            # 复制 T 帧
            inf_seq = replicate_frames(inf_image, self.T)  # list of [1,H,W]
            vis_seq = replicate_frames(vis_image, self.T)  # list of [3,H,W]

            # 计算光流（伪视频直接生成零张量）
            H, W = inf_image.shape[1], inf_image.shape[2]
            flow_ir = torch.zeros(self.T - 1, 2, H, W)
            flow_vi = torch.zeros(self.T - 1, 2, H, W)

            # 转换为 YCrCb
            vis_y_image, vis_cr_image, vis_cb_image = utils.RGB2Y_Cr_Cb(vis_image)
            vis_y_seq = replicate_frames(vis_y_image, self.T)
            vis_cr_seq = replicate_frames(vis_cr_image, self.T)
            vis_cb_seq = replicate_frames(vis_cb_image, self.T)

            return vis_seq, vis_y_seq, vis_cr_seq, vis_cb_seq, inf_seq, flow_ir, flow_vi, os.path.basename(vis_member.name)

    def __len__(self):
        return len(self.inf_members)
