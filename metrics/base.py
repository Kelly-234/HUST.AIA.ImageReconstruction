import torch

class BaseMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        pass
    
    def forward(self, recon_x, x):
        pass