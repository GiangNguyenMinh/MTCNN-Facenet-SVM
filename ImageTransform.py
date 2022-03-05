import torchvision.transforms as transform
from facenet_pytorch import fixed_image_standardization
import numpy as np

class ImageTransform():
    """ transform numpy or PIL image to tensor and normalize
    Return:
        Tensor Image with normalzation
    """
    def __init__(self):
        self.data_transform = transform.Compose([
            transform.ToTensor(),
            fixed_image_standardization
        ])

    def __call__(self, img):
        img = np.float32(img)
        return self.data_transform(img)
