import torch

class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "MAE"
    
    def forward(self, recon_x, x):
        return torch.nn.functional.l1_loss(recon_x, x.view(x.shape[0],-1))