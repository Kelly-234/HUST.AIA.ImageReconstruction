import torch

class PSNR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "PSNR"
    
    def forward(self, recon_x, x):
        pass