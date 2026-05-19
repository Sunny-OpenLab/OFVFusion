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
        self.inf_members.sort(key=lambda m: m.name)
        self.vis_members.sort(key=lambda m: m.name)

    def _compute_farneback_flow(self, frames_np):
        """
        frames_np: numpy array [T, 1, H, W], range [0, 255], dtype=uint8
        return: torch tensor [T-1, 2, H, W]
        """
        T, _, H, W = frames_np.shape
        flow_list = []
        fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=21,  
            iterations=3,
            poly_n=7, 
            poly_sigma=1.5,
            flags=0
        )

        for t in range(1, T):
            prev = frames_np[t - 1, 0]
            next = frames_np[t, 0]
            prev_blur = cv2.GaussianBlur(prev, (5, 5), 0)
            next_blur = cv2.GaussianBlur(next, (5, 5), 0)
            flow = cv2.calcOpticalFlowFarneback(prev_blur, next_blur, None, **fb_params)
            flow[np.abs(flow) > 20] = 0
            flow_list.append(flow)

        flow_np = np.stack(flow_list, axis=0) 
        flow_np = np.transpose(flow_np, (0, 3, 1, 2)) 

        return torch.from_numpy(flow_np).float()

    def __getitem__(self, index):

        with tarfile.open(self.tar_path, 'r') as tar_ref:
            start_idx = index * FRAME
            inf_images = []
            vis_images = []
            vis_names = []

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

                vis_names.append(vis_member.name) 

            inf_images = torch.stack(inf_images, dim=1)
            vis_images = torch.stack(vis_images, dim=1)
            vis_y_image, vis_cr_image, vis_cb_image = utils.RGB2Y_Cr_Cb(vis_images)

            inf_images = inf_images.permute(1, 0, 2, 3)
            vis_images = vis_images.permute(1, 0, 2, 3)  
            vis_y_image = vis_y_image.permute(1, 0, 2, 3)
            vis_cr_image = vis_cr_image.permute(1, 0, 2, 3)
            vis_cb_image = vis_cb_image.permute(1, 0, 2, 3)

            ir_np = (inf_images.numpy() * 255).astype(np.uint8) 
            vis_y_np = (vis_y_image.numpy() * 255).astype(np.uint8)

            flow_ir = self._compute_farneback_flow(ir_np) 
            flow_vi = self._compute_farneback_flow(vis_y_np)                

            return (
                vis_images, vis_y_image, vis_cr_image, vis_cb_image,
                inf_images,
                flow_ir, flow_vi,
                vis_names  
            )

    def __len__(self):
        return len(self.inf_members) // FRAME


def replicate_frames(image, T):
    image_tensor = torch.unsqueeze(image, 0)
    return image_tensor.repeat(T, 1, 1, 1)

class get_test_data_3D(data.Dataset):
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

        self.inf_members = [m for m in self.members if 'ir/' in m.name]
        self.vis_members = [m for m in self.members if 'vi/' in m.name]

        self.inf_members.sort(key=lambda m: os.path.basename(m.name))
        self.vis_members.sort(key=lambda m: os.path.basename(m.name))

    def __getitem__(self, index):
        with tarfile.open(self.test_folder, 'r') as tar_ref:
            inf_member = self.inf_members[index]
            vis_member = self.vis_members[index]

            inf_image = Image.open(tar_ref.extractfile(inf_member)).convert('L')
            vis_image = Image.open(tar_ref.extractfile(vis_member)).convert('RGB')

            if self.transform:
                inf_image = self.transform(inf_image)
                vis_image = self.transform(vis_image)

            inf_image = normalize_0_1(inf_image)
            vis_image = normalize_0_1(vis_image)

            inf_seq = replicate_frames(inf_image, self.T) 
            vis_seq = replicate_frames(vis_image, self.T) 

            H, W = inf_image.shape[1], inf_image.shape[2]
            flow_ir = torch.zeros(self.T - 1, 2, H, W)
            flow_vi = torch.zeros(self.T - 1, 2, H, W)

            vis_y_image, vis_cr_image, vis_cb_image = utils.RGB2Y_Cr_Cb(vis_image)
            vis_y_seq = replicate_frames(vis_y_image, self.T)
            vis_cr_seq = replicate_frames(vis_cr_image, self.T)
            vis_cb_seq = replicate_frames(vis_cb_image, self.T)

            return vis_seq, vis_y_seq, vis_cr_seq, vis_cb_seq, inf_seq, flow_ir, flow_vi, os.path.basename(vis_member.name)

    def __len__(self):
        return len(self.inf_members)
