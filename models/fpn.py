from .base import BaseModel

class FPN(BaseModel):
    def __init__(self):
        super().__init__()
        pass
            
    def __str__(self):
        return "FPN"

    def forward(self, x):
        pass