import os
import torch
from torchvision.utils import save_image
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        pass
    
    def forward(self, x):
        pass
    
    @staticmethod
    def get_loss(recon_x, x):
        return F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    def save_img(self, image, name, epoch, nrow=8):
        os.makedirs(os.path.join('results', str(self)), exist_ok=True)

        if epoch % 5 == 0:
            save_path = os.path.join('results', str(self), name + '_epoch' + str(epoch) + '.png')
            save_image(image, save_path, nrow=nrow)