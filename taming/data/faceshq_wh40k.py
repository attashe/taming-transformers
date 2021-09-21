import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        # print('Example: ', example)
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex, 0


class Wh40kTrain(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        # root = "data/wh40k_train"
        root = "data/deepdark_train"
        relpaths = filter(lambda x: x.endswith('jpg') or x.endswith('png') or x.endswith('jpeg'), 
                          os.listdir(root))
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=True)
        self.keys = keys


class Wh40kValidation(FacesBase):
    def __init__(self, size, keys=None):
        super().__init__()
        root = "data/wh40k_val"
        relpaths = filter(lambda x: x.endswith('jpg') or x.endswith('png') or x.endswith('jpeg'), 
                          os.listdir(root))
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=True)
        self.keys = keys


class Wh40kHQTrain(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = Wh40kTrain(size=size, keys=keys)
        self.data = d1
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.RandomCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex


class Wh40kHQValidation(Dataset):
    # CelebAHQ [0] + FFHQ [1]
    def __init__(self, size, keys=None, crop_size=None, coord=False):
        d1 = Wh40kValidation(size=size, keys=keys)
        self.data = d1
        self.coord = coord
        if crop_size is not None:
            self.cropper = albumentations.CenterCrop(height=crop_size,width=crop_size)
            if self.coord:
                self.cropper = albumentations.Compose([self.cropper],
                                                      additional_targets={"coord": "image"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex, y = self.data[i]
        # print('EX: ', ex)
        # print('Y: ', y)
        if hasattr(self, "cropper"):
            if not self.coord:
                out = self.cropper(image=ex["image"])
                ex["image"] = out["image"]
            else:
                h,w,_ = ex["image"].shape
                coord = np.arange(h*w).reshape(h,w,1)/(h*w)
                out = self.cropper(image=ex["image"], coord=coord)
                ex["image"] = out["image"]
                ex["coord"] = out["coord"]
        ex["class"] = y
        return ex
