"""
Augmentations Implemented as Callable Classes.
"""
import cv2
import numpy as np
import torch
from typing import Any, Tuple
import torchvision

class NormalizeBPS(object):
    def __call__(self, img_array) -> np.array(np.float32):
        """
        Normalize the array values between 0 - 1
        """
        raise NotImplementedError

class ResizeBPS(object):
    def __init__(self, resize_height: int, resize_width:int):
        pass

    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        Resize the image to the specified width and height
        """
        raise NotImplementedError

class ZoomBPS(object):
    def __init__(self, zoom: float=1) -> None:
        pass

    def __call__(self, image) -> np.ndarray:
        """
        Resize the image using zoom as scaling factor with area interpolation
        """
        raise NotImplementedError

class VFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image vertically
        """
        raise NotImplementedError


class HFlipBPS(object):
    def __call__(self, image) -> np.ndarray:
        """
        Flip the image horizontally
        """
        raise NotImplementedError


class RotateBPS(object):
    def __init__(self, rotate: int) -> None:
        pass

    def __call__(self, image) -> Any:
        """
        Rotate image by 90, 180, or 270 degrees.
        """
        raise NotImplementedError


class RandomCropBPS(object):
    def __init__(self, output_height: int, output_width: int):
        pass

    def __call__(self, image):
    """
    Crop randomly the image in a sample.
    """
        raise NotImplementedError

class ToTensor(object):
    def __call__(self, image: np.ndarray) -> torch.Tensor:
    """
    Convert an ndarray to Tensor.
    """
       raise NotImplementedError

