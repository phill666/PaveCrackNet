"""TUT dataset adapter for the RHACrackNet baseline."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

class TUTDataset(Dataset):
    _SPLITS = {"train": ("train_img", "train_lab"), "val": ("val_img", "val_lab"), "test": ("test_img", "test_lab")}

    def __init__(self, data_root, split, input_size=512, augment=None):
        if split not in self._SPLITS:
            raise ValueError(f"Unknown TUT split: {split}")
        self.data_root, self.split, self.input_size = Path(data_root), split, int(input_size)
        self.augment = split == "train" if augment is None else bool(augment)
        image_name, mask_name = self._SPLITS[split]
        self.image_dir, self.mask_dir = self.data_root / image_name, self.data_root / mask_name
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Missing TUT directories: {self.image_dir}, {self.mask_dir}")
        images = {p.stem: p for p in self.image_dir.glob("*") if p.is_file()}
        masks = {p.stem: p for p in self.mask_dir.glob("*") if p.is_file()}
        missing_masks, missing_images = sorted(set(images)-set(masks)), sorted(set(masks)-set(images))
        if missing_masks or missing_images:
            raise RuntimeError(f"Unpaired TUT files; missing masks={missing_masks[:5]}, missing images={missing_images[:5]}")
        self.samples = [(images[stem], masks[stem]) for stem in sorted(images)]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _official_train_augmentation(image, mask):
        if random.random() > 0.5:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        if random.random() > 0.5:
            image, mask = ImageOps.flip(image), ImageOps.flip(mask)
        if random.random() > 0.5:
            image, mask = ImageOps.flip(ImageOps.mirror(image)), ImageOps.flip(ImageOps.mirror(mask))
        if random.random() > 0.5:
            image = ImageEnhance.Brightness(image).enhance(1.5)
        if random.random() > 0.5:
            image = ImageEnhance.Brightness(image).enhance(0.5)
        if random.random() > 0.5:
            image = ImageEnhance.Contrast(image).enhance(1.5)
        return image, mask

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")
        size = (self.input_size, self.input_size)
        image = image.resize(size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
        if self.augment:
            image, mask = self._official_train_augmentation(image, mask)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask) > 0).astype(np.float32)
        return torch.from_numpy(image_array.transpose(2, 0, 1).copy()), torch.from_numpy(mask_array[None].copy())

    def compute_pos_weight(self, factor=3.0):
        positive, total = 0, 0
        size = (self.input_size, self.input_size)
        for _, mask_path in self.samples:
            with Image.open(mask_path) as source:
                binary = np.asarray(source.convert("L").resize(size, Image.Resampling.NEAREST)) > 0
            positive += int(binary.sum())
            total += int(binary.size)
        if positive == 0:
            raise RuntimeError("TUT training split contains no positive pixels")
        return total / positive * float(factor)
