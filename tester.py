from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
# from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
from model import StegNet
import math
from torch.utils.data import DataLoader
from datasets import *
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = '../DeepSteganography-master/model/'
IMAGE_PATH = '../DeepSteganography-master/test/'

img1 = Image.open(IMAGE_PATH + '1.jpg')
img2 = Image.open(IMAGE_PATH + '2.jpg')

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])


model = StegNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH + 'model.pkl'))

draw1 = ImageDraw.Draw(img1)
draw2 = ImageDraw.Draw(img2)

cover = transform(img1)
hidden = transform(img2)

torchvision.utils.save_image(hidden, filename=IMAGE_PATH + '_hidden.jpg')
torchvision.utils.save_image(cover, filename=IMAGE_PATH + '_output.jpg')

if torch.cuda.is_available():
    cover = cover.view(1, 3, 224, 224).cuda()
    hidden = hidden.view(1, 3, 224, 224).cuda()
else:
    cover = cover.view(1, 3, 224, 224)
    hidden = hidden.view(1, 3, 224, 224)

with torch.no_grad():
    model.eval()

    test_hidden, test_output = model(hidden, cover)

    torchvision.utils.save_image(test_hidden, filename=IMAGE_PATH + '_out' + '_hidden.jpg')
    torchvision.utils.save_image(test_output, filename=IMAGE_PATH + '_out' + '_output.jpg')

