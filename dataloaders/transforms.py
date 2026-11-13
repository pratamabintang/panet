"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tr_F


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
            scribble = scribble.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)
        inst = tr_F.resize(inst, self.size, interpolation=Image.NEAREST)
        scribble = tr_F.resize(scribble, self.size, interpolation=getattr(Image, 'ANTIALIAS', getattr(Image, 'LANCZOS', Image.BILINEAR)))

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample

class DilateScribble(object):
    """
    Dilate the scribble mask

    Args:
        size: window width
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        scribble = sample['scribble']
        dilated_scribble = Image.fromarray(
            ndimage.minimum_filter(np.array(scribble), size=self.size))
        dilated_scribble.putpalette(scribble.getpalette())

        sample['scribble'] = dilated_scribble
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()
        inst = torch.Tensor(np.array(inst)).long()
        scribble = torch.Tensor(np.array(scribble)).long()

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


class RandomFlip(object):
    """
    Randomly flip images/masks horizontally and/or vertically.

    Geometric transform: image, label, inst and scribble are flipped together
    so they stay aligned. Each axis is flipped independently with p=0.5.
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            inst = inst.transpose(Image.FLIP_LEFT_RIGHT)
            scribble = scribble.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_TOP_BOTTOM)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            inst = inst.transpose(Image.FLIP_TOP_BOTTOM)
            scribble = scribble.transpose(Image.FLIP_TOP_BOTTOM)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


class RandomRotation(object):
    """
    Randomly rotate images/masks by an angle sampled in [-degrees, degrees].

    Geometric transform: image uses bilinear interpolation while
    label/inst/scribble use nearest interpolation, all by the same angle.
    Pixels falling outside the rotated frame are filled with ``fill``.

    Args:
        degrees: max absolute rotation angle (180 = any orientation)
        fill:    fill value for pixels outside the rotated image
    """
    def __init__(self, degrees=180, fill=0):
        self.degrees = degrees
        self.fill = fill

    def __call__(self, sample):
        angle = random.uniform(-self.degrees, self.degrees)
        if angle == 0:
            return sample

        img, label = sample['image'], sample['label']
        inst, scribble = sample['inst'], sample['scribble']

        img = tr_F.rotate(img, angle, interpolation=Image.BILINEAR, fill=self.fill)
        if isinstance(label, dict):
            label = {catId: tr_F.rotate(x, angle, interpolation=Image.NEAREST, fill=self.fill)
                     for catId, x in label.items()}
        else:
            label = tr_F.rotate(label, angle, interpolation=Image.NEAREST, fill=self.fill)
        inst = tr_F.rotate(inst, angle, interpolation=Image.NEAREST, fill=self.fill)
        scribble = tr_F.rotate(scribble, angle, interpolation=Image.NEAREST, fill=self.fill)

        sample['image'] = img
        sample['label'] = label
        sample['inst'] = inst
        sample['scribble'] = scribble
        return sample


class ColorJitter(object):
    """
    Colour perturbation applied ONLY to the multispectral bands.

    Args:
        brightness, contrast, saturation, hue:
            jitter strengths passed to ``torchvision.transforms.ColorJitter``
        keys:
            sample keys to perturb. Defaults to ``('image',)`` so only the
            optical/multispectral stack is altered; topographic layers such as
            DEM or slope (stored under other keys) are never touched.
    """
    def __init__(self, brightness=0.2, contrast=0.3, saturation=0.2, hue=0.0,
                 keys=('image',)):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast,
                                    saturation=saturation, hue=hue)
        self.keys = tuple(keys)

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = self.jitter(sample[key])
        return sample
