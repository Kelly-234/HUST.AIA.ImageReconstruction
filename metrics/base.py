import os
import torch
from torchvision.utils import save_image
from torch.nn import functional as F


class BaseMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __str__(self):
        pass

    def forward(self, recon_x, x):
        pass
