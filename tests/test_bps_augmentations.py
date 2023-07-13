import unittest
import cv2
import numpy as np
import torch
import torchvision
from typing import Any, Tuple
import pyprojroot
import sys

sys.path.append(str(pyprojroot.here()))

from custom_dataset_dataloader_transforms.dataset.augmentation import NormalizeBPS, ResizeBPS, ZoomBPS, VFlipBPS, HFlipBPS, RotateBPS, RandomCropBPS, ToTensor

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.img = np.random.rand(256, 256).astype(np.uint16)
        self.normalize = NormalizeBPS()
        self.resize = ResizeBPS(resize_height=128, resize_width=128)
        self.zoom = ZoomBPS(zoom=2)
        self.vflip = VFlipBPS()
        self.hflip = HFlipBPS()
        self.rotate = RotateBPS(rotate=90)
        self.randomcrop = RandomCropBPS(output_height=128, output_width=128)
        self.to_tensor = ToTensor()

    def tearDown(self):
        pass

    def test_normalize(self):
        img_normalized = self.normalize(self.img)
        self.assertTrue(isinstance(img_normalized, np.ndarray), 'Image is not a numpy array.')
        self.assertLessEqual(np.max(img_normalized), 1.0, 'Image is not normalized.')

    def test_resize(self):
        img_resized = self.resize(self.img)
        self.assertTrue(isinstance(img_resized, np.ndarray), 'Image is not a numpy array.')
        self.assertEqual(img_resized.shape, (128, 128), 'Image is not the correct shape based on resize parameters.')

    def test_zoom(self):
        img_zoomed = self.zoom(self.img)
        self.assertTrue(isinstance(img_zoomed, np.ndarray), 'Image is not a numpy array.')
        self.assertEqual(img_zoomed.shape, (512, 512), 'Image is not the correct shape based on zoom parameters.')

    def test_vflip(self):
        img_flipped = self.vflip(self.img)
        self.assertTrue(isinstance(img_flipped, np.ndarray))
        self.assertTrue(np.allclose(img_flipped, np.flipud(self.img)), 'Image is not flipped vertically.')

    def test_hflip(self):
        img_flipped = self.hflip(self.img)
        self.assertTrue(isinstance(img_flipped, np.ndarray))
        self.assertTrue(np.allclose(img_flipped, np.fliplr(self.img)), 'Image is not flipped horizontally.')

    def test_rotate(self):
        img_rotated = self.rotate(self.img)
        self.assertTrue(isinstance(img_rotated, np.ndarray))
        self.assertEqual(img_rotated.shape, (256, 256))
        self.assertTrue(np.allclose(img_rotated, np.rot90(self.img)), 'Image is not rotated 90 degrees.')

    def test_randomcrop(self):
        img_cropped = self.randomcrop(self.img)
        self.assertTrue(isinstance(img_cropped, np.ndarray))
        self.assertEqual(img_cropped.shape, (128, 128), 'Image is not the correct shape based on crop parameters.')

    def test_to_tensor(self):
        img_tensor = self.to_tensor(self.img.astype(np.float32))
        self.assertTrue(isinstance(img_tensor, torch.Tensor))
        self.assertEqual(img_tensor.shape, (1, 256, 256), 'Image is not the correct shape based on tensor parameters.')

if __name__ == '__main__':
    unittest.main()
