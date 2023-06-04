import torch
from skimage.metrics import peak_signal_noise_ratio
import numpy as np

class PSNR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "PSNR"
    
    def forward(self, recon_x, x):
        img1 = (recon_x.detach().numpy() * 255).astype(np.uint8)
        img2 = (x.reshape(x.shape[0], -1).detach().numpy() * 255).astype(np.uint8)
        return peak_signal_noise_ratio(img1, img2, data_range=None)