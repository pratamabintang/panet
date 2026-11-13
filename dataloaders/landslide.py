"""
Load the Landslide4Sense dataset (few-shot, RGB only)

Each patch is stored as an HDF5 file:
    images/<split>/image_<n>.h5   -> dataset 'img'  (H x W x 14)
    annotations/<split>/mask_<n>.h5 -> dataset 'mask' (H x W)

Only the RGB bands are kept here, in true-colour order R=B4, G=B3, B=B2
(0-indexed band positions 3, 2, 1 of the 14-band stack). The raw values are
float reflectance (roughly 0.25-6.4), so they are divided by ``SCALE`` and
clipped to [0, 1] before the usual ``ToTensorNormalize`` (ImageNet stats).

Labels are binary: 0 = non-landslide, 1 = landslide.
"""

import os
import random

import h5py
import numpy as np
from PIL import Image
import torch

from .common import BaseDataset
from .common import PairedDataset
from .customized import attrib_basic, fewShot


class Landslide(BaseDataset):
    """
    Base class for the Landslide4Sense dataset (RGB + binary label)

    Args:
        base_dir:
            Landslide4Sense dataset directory (holds ``images``/``annotations``)
        split:
            which split to use, choose from ('train', 'validation', 'test')
        transforms:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """

    # 0-indexed positions of B4/B2 among the 14 bands; reordered below as R,G,B
    RGB_BANDS = (3, 2, 1)  # B4 (red), B3 (green), B2 (blue)
    SCALE = 4.0            # map raw reflectance to roughly [0, 1]

    def __init__(self, base_dir, split, transforms=None, to_tensor=None):
        super().__init__(base_dir)
        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'images', split)
        self._label_dir = os.path.join(self._base_dir, 'annotations', split)
        self.transforms = transforms
        self.to_tensor = to_tensor

        names = [f for f in os.listdir(self._image_dir) if f.endswith('.h5')]
        self.ids = sorted((os.path.splitext(f)[0] for f in names),
                          key=lambda x: int(x.split('_')[-1]))

    def __len__(self):
        return len(self.ids)

    def _image_path(self, id_):
        return os.path.join(self._image_dir, f'{id_}.h5')

    def _label_path(self, id_):
        return os.path.join(self._label_dir, f'{id_.replace("image", "mask")}.h5')

    @classmethod
    def _load_image(cls, path):
        with h5py.File(path, 'r') as hf:
            image = np.asarray(hf['img'][:], np.float32)
        # channels last -> select R,G,B and rescale to [0, 255] uint8
        rgb = image[..., cls.RGB_BANDS]
        rgb = np.clip(rgb / cls.SCALE, 0.0, 1.0)
        return Image.fromarray((rgb * 255.0).round().astype(np.uint8))

    @staticmethod
    def _load_label(path):
        with h5py.File(path, 'r') as hf:
            mask = np.asarray(hf['mask'][:])
        mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.uint8)  # 1 = landslide, 0 = background
        return Image.fromarray(mask)

    def has_foreground(self, id_):
        """Whether the patch contains at least one landslide pixel."""
        path = self._label_path(id_)
        if not os.path.exists(path):
            return False
        with h5py.File(path, 'r') as hf:
            mask = np.asarray(hf['mask'][:])
        return bool((mask > 0).any())

    def get_ids_with_class(self, class_id=1):
        """
        Ids of patches containing the given foreground class (only class 1,
        landslide, is meaningful for this binary task). Results are cached in
        ``class<class_id>_<split>.txt`` next to the dataset so that later runs
        do not have to rescan every mask.
        """
        cache = os.path.join(self._base_dir, f'class{class_id}_{self.split}.txt')
        if os.path.exists(cache):
            with open(cache, 'r') as f:
                cached = [line.strip() for line in f if line.strip()]
            known = set(self.ids)
            return [id_ for id_ in cached if id_ in known]

        ids = [id_ for id_ in self.ids if self.has_foreground(id_)]
        with open(cache, 'w') as f:
            f.write('\n'.join(ids))
        return ids

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        image = self._load_image(self._image_path(id_))

        label_path = self._label_path(id_)
        if os.path.exists(label_path):
            label = self._load_label(label_path)
        else:  # split without labels (e.g. original validation/test release)
            label = Image.fromarray(np.zeros((image.height, image.width), np.uint8))

        # No instance/scribble annotation for this dataset: keep zero masks so
        # the shared transforms / model interface stay unchanged.
        blank = Image.fromarray(np.zeros((image.height, image.width), np.uint8))
        sample = {'image': image,
                  'label': label,
                  'inst': blank,
                  'scribble': blank.copy()}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(np.array(sample['image']).transpose(2, 0, 1))
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t

        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample


def landslide_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots,
                      max_iters, n_queries=1):
    """
    Args:
        base_dir:
            Landslide4Sense dataset directory
        split:
            which split to use, choose from ('train', 'validation', 'test')
        transforms:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            foreground class labels of the data (only class 1, landslide)
        n_ways:
            n-way few-shot learning (1 for this binary task)
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of episodes
        n_queries:
            number of query images
    """
    landslide = Landslide(base_dir=base_dir, split=split,
                          transforms=transforms, to_tensor=to_tensor)
    landslide.add_attrib('basic', attrib_basic, {})

    # Only patches that actually contain the (single) foreground class are used
    # as support/query episodes, following the PANet convention.
    sub_ids = [landslide.get_ids_with_class(cls_id) for cls_id in labels]
    for cls_id, ids in zip(labels, sub_ids):
        if len(ids) == 0:
            raise RuntimeError(
                f'No patch with landslide (class {cls_id}) found in split '
                f'"{split}" of {base_dir}. Few-shot episodes cannot be built.')
    subsets = landslide.subsets(
        sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query})])
    return paired_data
