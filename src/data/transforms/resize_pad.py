from torchvision import transforms
from PIL import Image

class ResizeOrPad:
    def __init__(self, desired_size):
        self.desired_size = desired_size

    def __call__(self, img):
        # If the image is larger than the desired size, resize it
        if img.shape[1] > self.desired_size[0] or img.shape[2] > self.desired_size[1]:
            img = transforms.Resize(self.desired_size, Image.BILINEAR)(img)

        # If the image is smaller than the desired size, pad it
        padding = (0, 0, self.desired_size[1] - img.shape[2], self.desired_size[0] - img.shape[1])
        img = transforms.Pad(padding, fill=0)(img)
        return img