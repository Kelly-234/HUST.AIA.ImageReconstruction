from .base import BaseModel

class UNet(BaseModel):
    def __init__(self):
        super().__init__()
        pass
            
    def __str__(self):
        return "UNet"
    
    def forward(self, x):
        pass