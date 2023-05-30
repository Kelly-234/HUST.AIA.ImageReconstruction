from .base import BaseModel

class SegNet(BaseModel):
    def __init__(self):
        super().__init__()
        pass
            
    def __str__(self):
        return "SegNet"
    
    def forward(self, x):
        pass