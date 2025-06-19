"""fedvae: A Flower app for Federated CycleGAN."""

from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image
import multiprocessing
import random
from torch.utils.data import Dataset
from itertools import zip_longest



# set_seed(42)  # 在 partition 或 visualize 前设置

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Create directories for saving results
RESULTS_DIR = Path("results")
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
PLOTS_DIR = RESULTS_DIR / "plots"

for directory in [RESULTS_DIR, VISUALIZATION_DIR, CHECKPOINT_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

from collections import defaultdict

# def label_getter(sample):
#     # 简单方式：比较 horse 和 zebra 图像的张量范数（模拟标签）
#     return 0 if torch.norm(sample["horse"]) > torch.norm(sample["zebra"]) else 1


# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SingleDomainDataset(Dataset):
    def __init__(self, image_input, transform=None):
        self.transform = transform

        if isinstance(image_input, (str, Path)) and os.path.isdir(image_input):
            self.image_paths = sorted([
                Path(image_input) / f for f in os.listdir(image_input)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        elif isinstance(image_input, list):
            self.image_paths = list(map(Path, image_input))
        else:
            raise ValueError(f"Invalid image_input: {image_input}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.G_AB = Generator()  # Generator for A->B
        self.G_BA = Generator()  # Generator for B->A
        self.D_X = Discriminator()  # Discriminator for domain A
        self.D_Y = Discriminator()  # Discriminator for domain B

    def forward(self, x):
        return self.G_AB(x)


fds = None  # Cache FederatedDataset


from torch.utils.data import DataLoader
from torchvision import transforms

from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os

def load_data(partition_id, num_partitions, alpha, data_root="./CT2MRI", batch_size=1):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # ⭐ 转为1通道
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 获取图像路径
    horse_dir = Path(data_root) / "trainA"
    zebra_dir = Path(data_root) / "trainB"
    horse_files = sorted([horse_dir / f for f in os.listdir(horse_dir) if f.endswith((".jpg", ".png"))])
    zebra_files = sorted([zebra_dir / f for f in os.listdir(zebra_dir) if f.endswith((".jpg", ".png"))])

    # 非IID划分
    from .data_utils import noniid_partition_horse_zebra
    client_data_dict = noniid_partition_horse_zebra(horse_files, zebra_files, num_partitions, alpha)

    horse_paths = client_data_dict[partition_id]["horse"]
    zebra_paths = client_data_dict[partition_id]["zebra"]

    # ✅ 使用 PairedDataset
    paired_train_ds = PairedDataset(horse_paths, zebra_paths, transform)
    paired_train_loader = DataLoader(paired_train_ds, batch_size=batch_size, shuffle=True)

    # 测试集仍然分开（用于 test/eval）
    test_horse = SingleDomainDataset(Path(data_root) / "testA", transform=transform)
    test_zebra = SingleDomainDataset(Path(data_root) / "testB", transform=transform)
    test_horse_loader = DataLoader(test_horse, batch_size=batch_size, shuffle=False)
    test_zebra_loader = DataLoader(test_zebra, batch_size=batch_size, shuffle=False)

    return paired_train_loader, (test_horse_loader, test_zebra_loader)

def get_client_loaders(client_data_dict, transform, batch_size):
    loaders = {}
    for client_id, domains in client_data_dict.items():
        horse_ds = SingleDomainDataset(domains["horse"], transform=transform)
        zebra_ds = SingleDomainDataset(domains["zebra"], transform=transform)

        horse_loader = DataLoader(horse_ds, batch_size=batch_size, shuffle=False)
        zebra_loader = DataLoader(zebra_ds, batch_size=batch_size, shuffle=False)

        loaders[client_id] = (horse_loader, zebra_loader)
    return loaders

# PairedDataset用于Horse2Zebra数据集，trainA和trainB非匹配数据
class PairedDataset(Dataset):
    def __init__(self, horse_paths, zebra_paths, transform):
        self.horse_paths = horse_paths
        self.zebra_paths = zebra_paths
        self.transform = transform
        self.length = min(len(horse_paths), len(zebra_paths))  # 对齐长度

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        horse_img = Image.open(self.horse_paths[idx]).convert("L")
        zebra_img = Image.open(self.zebra_paths[idx]).convert("L")
        if self.transform:
            horse_img = self.transform(horse_img)
            zebra_img = self.transform(zebra_img)
        return horse_img, zebra_img

# # PairedDataset用于IMR2Ya数据集，trainA和trainB匹配数据
# class PairedDataset(Dataset):
#     def __init__(self, horse_paths, zebra_paths, transform=None):
#         self.transform = transform
#
#         # 将文件名（不含路径）映射到完整路径
#         horse_dict = {Path(p).name: p for p in horse_paths}
#         zebra_dict = {Path(p).name: p for p in zebra_paths}
#
#         # 获取两类数据共有的文件名（即确保成对）
#         self.shared_keys = sorted(list(set(horse_dict.keys()) & set(zebra_dict.keys())))
#
#         if not self.shared_keys:
#             raise ValueError("No matching filenames found between trainA and trainB.")
#
#         self.horse_dict = horse_dict
#         self.zebra_dict = zebra_dict
#
#     def __len__(self):
#         return len(self.shared_keys)
#
#     def __getitem__(self, idx):
#         fname = self.shared_keys[idx]
#
#         horse_path = self.horse_dict[fname]
#         zebra_path = self.zebra_dict[fname]
#
#         horse_img = Image.open(horse_path).convert("L")
#         zebra_img = Image.open(zebra_path).convert("L")
#
#         if self.transform:
#             horse_img = self.transform(horse_img)
#             zebra_img = self.transform(zebra_img)
#
#         return horse_img, zebra_img

def save_images(images, epoch, client_id, prefix):
    """Save a grid of images."""
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Client {client_id} - {prefix} Images - Epoch {epoch}")
    
    # Create a grid of images
    plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu(), (1, 2, 0)))
    
    # Save the figure
    save_path = VISUALIZATION_DIR / f"client_{client_id}_{prefix}_epoch_{epoch}.png"
    plt.savefig(save_path)
    plt.close()


def plot_client_round_loss(history, client_id, save_path=None):
    """
    Plot training loss per federated round for this client.

    Args:
        history (dict): Dictionary with loss names as keys, values are lists
        client_id (int): Client ID
        save_path (Path): File path to save the figure
    """
    plt.figure(figsize=(12, 8))
    for loss_name, loss_values in history.items():
        plt.plot(loss_values, label=loss_name, linewidth=2)

    plt.title(f"Loss per Round - Client {client_id}", fontsize=14)
    plt.xlabel("Federated Round", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_combined_training_curves(all_histories, save_path=None):
    """
    Plot training curves for all clients on the same graph.
    
    Args:
        all_histories (dict): Dictionary of histories for each client
        save_path (Path, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot each loss type for all clients
    for loss_name in all_histories[0].keys():
        for client_id, history in all_histories.items():
            plt.plot(history[loss_name], 
                    label=f'Client {client_id} - {loss_name}',
                    alpha=0.7,
                    linewidth=2)
    
    plt.title('Combined Training Curves Across All Clients', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def train(net, paired_loader, epochs, learning_rate, device, client_id):
    net.train()
    net.to(device)

    optim_G = torch.optim.Adam(
        list(net.G_AB.parameters()) + list(net.G_BA.parameters()),
        lr=learning_rate, betas=(0.5, 0.999)
    )
    optim_D = torch.optim.Adam(
        list(net.D_X.parameters()) + list(net.D_Y.parameters()),
        lr=learning_rate, betas=(0.5, 0.999)
    )

    history = {
        "loss_G": [],
        "loss_D": [],
        "loss_cycle": [],
        "loss_identity": [],
        "batches_per_epoch": [],
    }

    for epoch in range(epochs):
        total_losses = {k: 0.0 for k in history if k != "batches_per_epoch"}
        steps = 0

        for batch_idx, (real_horse, real_zebra) in enumerate(paired_loader):
            real_horse = real_horse.to(device)
            real_zebra = real_zebra.to(device)

            loss_G, loss_D, loss_cycle, loss_id = train_step(
                net, real_horse, real_zebra, optim_G, optim_D, device
            )

            total_losses["loss_G"] += loss_G
            total_losses["loss_D"] += loss_D
            total_losses["loss_cycle"] += loss_cycle
            total_losses["loss_identity"] += loss_id
            steps += 1

            logger.info(
                f"[Client {client_id}] Epoch {epoch+1} | Batch {batch_idx+1}/{len(paired_loader)} | "
                f"G: {loss_G:.4f}, D: {loss_D:.4f}, Cycle: {loss_cycle:.4f}, ID: {loss_id:.4f}"
            )

        # 平均损失
        for k in total_losses:
            history[k].append(total_losses[k] / steps if steps > 0 else 0.0)
        history["batches_per_epoch"].append(steps)

        logger.info(
            f"[Client {client_id}] Epoch {epoch+1} Summary: "
            f"{ {k: round(v[-1], 4) for k, v in history.items() if v} }"
        )

    return history

import torch
from torch import Tensor
from typing import Tuple


import torch.nn.functional as F

def train_step(net, real_horse, real_zebra, optim_G, optim_D, device,
               lambda_cycle=10.0, lambda_identity=6.0):
    net.to(device)
    real_horse = real_horse.to(device)
    real_zebra = real_zebra.to(device)

    G_AB = net.G_AB  # Horse → Zebra
    G_BA = net.G_BA  # Zebra → Horse
    D_X = net.D_X    # Discriminate Horse
    D_Y = net.D_Y    # Discriminate Zebra

    # === 1. Train Generators ===
    optim_G.zero_grad()

    # Forward pass through generators
    fake_zebra = G_AB(real_horse)
    fake_horse = G_BA(real_zebra)

    recov_horse = G_BA(fake_zebra)
    recov_zebra = G_AB(fake_horse)

    # Identity loss (should preserve identity if input already in target domain)
    same_zebra = G_AB(real_zebra)
    same_horse = G_BA(real_horse)
    loss_identity = F.l1_loss(same_zebra, real_zebra) + F.l1_loss(same_horse, real_horse)

    # GAN loss
    pred_fake_zebra = D_Y(fake_zebra)
    pred_fake_horse = D_X(fake_horse)
    loss_GAN_AB = F.mse_loss(pred_fake_zebra, torch.ones_like(pred_fake_zebra))
    loss_GAN_BA = F.mse_loss(pred_fake_horse, torch.ones_like(pred_fake_horse))

    # Cycle loss
    loss_cycle = F.l1_loss(recov_horse, real_horse) + F.l1_loss(recov_zebra, real_zebra)

    # Total generator loss
    loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * loss_cycle + lambda_identity * loss_identity
    loss_G.backward()
    optim_G.step()

    # === 2. Train Discriminators ===
    optim_D.zero_grad()

    # Discriminator Y (real vs fake zebra)
    pred_real_zebra = D_Y(real_zebra)
    pred_fake_zebra = D_Y(fake_zebra.detach())
    loss_D_Y = 0.5 * (
        F.mse_loss(pred_real_zebra, torch.ones_like(pred_real_zebra)) +
        F.mse_loss(pred_fake_zebra, torch.zeros_like(pred_fake_zebra))
    )

    # Discriminator X (real vs fake horse)
    pred_real_horse = D_X(real_horse)
    pred_fake_horse = D_X(fake_horse.detach())
    loss_D_X = 0.5 * (
        F.mse_loss(pred_real_horse, torch.ones_like(pred_real_horse)) +
        F.mse_loss(pred_fake_horse, torch.zeros_like(pred_fake_horse))
    )

    # Total discriminator loss
    loss_D = loss_D_X + loss_D_Y
    loss_D.backward()
    optim_D.step()

    return (
        loss_G.item(),
        loss_D.item(),
        loss_cycle.item(),
        loss_identity.item()
    )

def test(net, horse_loader, zebra_loader, device, client_id):
    net.eval()
    total_loss_G = 0.0
    total_loss_cycle = 0.0
    total_loss_identity = 0.0
    steps = 0

    horse_iter = iter(horse_loader)
    zebra_iter = iter(zebra_loader)
    min_len = min(len(horse_loader), len(zebra_loader))

    with torch.no_grad():
        for _ in range(min_len):
            try:
                horse_data = next(horse_iter)
                zebra_data = next(zebra_iter)
            except StopIteration:
                break

            if horse_data is None or zebra_data is None:
                continue

            real_horse = horse_data.to(device)
            real_zebra = zebra_data.to(device)

            # === Generator forward ===
            fake_zebra = net.G_AB(real_horse)
            fake_horse = net.G_BA(real_zebra)

            recov_horse = net.G_BA(fake_zebra)
            recov_zebra = net.G_AB(fake_horse)

            same_horse = net.G_BA(real_horse)
            same_zebra = net.G_AB(real_zebra)

            # === L1 losses for evaluation only ===
            loss_G = F.l1_loss(fake_zebra, real_zebra) + F.l1_loss(fake_horse, real_horse)
            loss_cycle = F.l1_loss(recov_horse, real_horse) + F.l1_loss(recov_zebra, real_zebra)
            loss_identity = F.l1_loss(same_horse, real_horse) + F.l1_loss(same_zebra, real_zebra)

            total_loss_G += loss_G.item()
            total_loss_cycle += loss_cycle.item()
            total_loss_identity += loss_identity.item()
            steps += 1

    if steps == 0:
        logger.warning(f"[Client {client_id}] No valid test samples available.")
        return {
            "loss_G": float('inf'),
            "loss_cycle": float('inf'),
            "loss_identity": float('inf'),
        }

    return {
        "loss_G": total_loss_G / steps,
        "loss_cycle": total_loss_cycle / steps,
        "loss_identity": total_loss_identity / steps,
    }



def get_weights(net):
    """Get weights from all networks."""
    weights = []
    for model in [net.G_AB, net.G_BA, net.D_X, net.D_Y]:
        weights.extend([val.cpu().numpy() for _, val in model.state_dict().items()])
    return weights


def set_weights(net, parameters):
    from collections import OrderedDict
    import torch

    models = [net.G_AB, net.G_BA, net.D_X, net.D_Y]
    state_sizes = [len(m.state_dict()) for m in models]

    # 每层的参数名和 shape 是已知的，可以预先构造索引
    idx = 0
    for model in models:
        state_dict = model.state_dict()
        param_dict = OrderedDict()
        for k in state_dict.keys():
            shape = state_dict[k].shape
            param = torch.tensor(parameters[idx]).reshape(shape)
            param_dict[k] = param
            idx += 1
        model.load_state_dict(param_dict, strict=True)
