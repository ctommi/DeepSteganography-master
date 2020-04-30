# coding=utf-8
from torchvision import datasets
import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class MyDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms_
        self.files = sorted(glob.glob(os.path.join(root, "%s" % mode) + "/*.*"))

    def __getitem__(self, index):
        image = Image.open(self.files[index])

        image = to_rgb(image)
        item = self.transform(image)
        return item

    def __len__(self):
        return len(self.files)
