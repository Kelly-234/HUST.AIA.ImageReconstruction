import torchvision.transforms.functional as TF
import cv2
import numpy as np

class GetCopy(object):
    def __init__(self):
        pass
        
    def __call__(self, x):
        x = np.array(x)
        return np.stack((x, x), axis=2)
    

class GetEdge(object):
    def __init__(self, l_thre=50, r_thre=100):
        self.l_thre = l_thre
        self.r_thre = r_thre
        
    def __call__(self, x):
        x = np.array(x)
        return np.stack((x, cv2.Canny(x, self.l_thre, self.r_thre)), axis=2)
    
    
class GetBlur(object):
    def __init__(self, kernal=None):
        if kernal is None:
            kernal = [5,5]
        self.kernal = kernal
        
    def __call__(self, x):
        blur_x = TF.gaussian_blur(x, self.kernal)
        return np.stack((blur_x, x), axis=2)