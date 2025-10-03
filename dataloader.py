## Data Loader

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

mean = [0.4914, 0.4822, 0.4465]
std  = [0.2470, 0.2435, 0.2616]

train_transform = A.Compose([
    A.HorizontalFlip(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, 
                    fill_value=tuple([int(m*255) for m in mean]), mask_fill_value=None),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

class CIFAR10Alb(datasets.CIFAR10):
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

def get_loaders(batch_size=256):
    train_dataset = CIFAR10Alb(root="./data", train=True, download=True, transform=train_transform)
    test_dataset  = CIFAR10Alb(root="./data", train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
