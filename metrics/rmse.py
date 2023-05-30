import torch

class RMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "RMSE"
    
    def forward(self, recon_x, x):
        pass