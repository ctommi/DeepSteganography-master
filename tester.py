import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class mod(nn.Module):
    def __init__(self):
        super().__init__()

        self.r1 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.r2 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.r3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.r4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.r5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.r6 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.r7 = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, x, x2):
        r1 = self.r1(x)
        r2 = self.r2(x)
        r3 = self.r3(x)
        x = torch.cat((r1, r2, r3), 1)
        r4 = self.r4(x)
        r5 = self.r5(x)
        r6 = self.r6(x)
        x = torch.cat((r4, r5, r6), 1)
        x = self.r7(x)
        x2 = x
        return x, x2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mod().to(device)

dummy_input1 = torch.randn(16, 3, 30, 30)
dummy_input2 = torch.randn(16, 3, 30, 30)
dummy_input1 = dummy_input1.to(device)
dummy_input2 = dummy_input2.to(device)


with SummaryWriter(comment='RDNet') as w:
    w.add_graph(model,(dummy_input1,dummy_input2))
