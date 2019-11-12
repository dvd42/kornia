"""
Data augmentation on the GPU
============================

In this data you learn how to use `kornia` modules in order to perform the data augmentation on the GPU in batch mode.
"""

################################
# 1. Create a dummy data loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

################################
# 2. Define the data augmentation operations
# Thanks to the `kornia` design all the operators can be placed inside inside a `nn.Sequential`.

import kornia

transform = nn.Sequential(
    kornia.color.Normalize(torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])),
    kornia.augmentation.ColorJitter(brightness=0.5, contrast=1.2, hue=0.5, saturation=0.3),
    kornia.augmentation.RandomHorizontalFlip(0.5),
)

################################
# 3. Run the dataset and perform the data augmentation

# NOTE: change device to 'cuda'
device = torch.device('cpu')
print(f"Running with device: {device}")

# create the dataloader
# dataset = CIFAR10(root="data", download=True, transform=ToTensor())
dataset = CIFAR10(root="data", download=True, transform=kornia.image_to_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# get samples and perform the data augmentation
# import ipdb; ipdb.set_trace()  # BREAKPOINT
for img, target in dataloader:

    images = img.to(device)
    labels = target.to(device)

    # perform the transforms
    images = transform(images)

    print(f"Iteration: {i_batch} Image shape: {images.shape}")
