from .base import BaseModel

class HourGlass(BaseModel):
    def __init__(self):
        super().__init__()
        pass
            
    def __str__(self):
        return "HourGlass"
    
    def forward(self, x):
        pass