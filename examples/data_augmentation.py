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
from torchvision.datasets import VOCDetection
from torchvision.transforms import Resize
from PIL import Image

# Necessary to parse VOC Targets
import xml.etree.ElementTree as ET


################################
# 1. Create VOC dataset
class VOC(VOCDetection):

    def __init__(self, root, download=False, transform=None):

        super(VOC, self).__init__(root, download=download, transform=transform)


    def __len__(self):
        return super(VOC, self).__len__()

    def __getitem__(self, index):

        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())
        boxes = []
        target = target['annotation']['object']
        target = target if isinstance(target, list) else [target]
        for t in target:
            boxes.append([int(c) for c in t['bndbox'].values()])


        boxes = torch.tensor(boxes)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img


################################
# 2. Define the data augmentation operations
# Thanks to the `kornia` design all the operators can be placed inside inside a `nn.Sequential`.

import kornia

transform = nn.Sequential(
    # kornia.augmentation.ColorJitter(brightness=0.5, contrast=1.2, hue=0.5, saturation=0.3),
    kornia.augmentation.RandomHorizontalFlip(0.5),
)

################################
# 3. Run the dataset and perform the data augmentation

# NOTE: change device to 'cuda'
device = torch.device('cpu')
print(f"Running with device: {device}")

# create the dataloader
dataset = VOC(root="data", download=True, transform=lambda x: kornia.image_to_tensor(Resize((512, 512))(x)))
import ipdb; ipdb.set_trace()  # BREAKPOINT
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# get samples and perform the data augmentation
for img in dataloader:

    images = img.to(device)
    # labels = target.to(device)

    # perform the transforms
    images = transform(images)
    from PIL import Image
    import ipdb; ipdb.set_trace()  # BREAKPOINT

    print(f"Iteration: {i_batch} Image shape: {images.shape}")
