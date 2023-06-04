import torch
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

class SSIM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SSIM"

    def forward(self, recon_x, x):
        img1 = (recon_x.detach().numpy() * 255).astype(np.uint8)
        img2 = (x.reshape(x.shape[0], -1).detach().numpy() * 255).astype(np.uint8)
        return compare_ssim(img1, img2, multichannel=True)