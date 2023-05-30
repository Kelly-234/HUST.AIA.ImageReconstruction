import torch

class SSIM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "SSIM"
    
    def forward(self, recon_x, x):
        pass