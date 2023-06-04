import torch
from skimage.metrics import normalized_root_mse
import numpy as np

class RMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "RMSE"

    def forward(self, recon_x, x):
        img1 = (recon_x.detach().numpy() * 255).astype(np.uint8)
        img2 = (x.reshape(x.shape[0], -1).detach().numpy() * 255).astype(np.uint8)
        return normalized_root_mse(img1, img2, normalization='mean')