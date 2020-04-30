import matplotlib.pyplot as plt
import numpy as np
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

learning_rate = 0.0005
epochs = 150
beta = 1
batch_size = 4

TRAIN_PATH = '../DeepSteganography-master/data/'
VAL_PATH = '../DeepSteganography-master/data/'
IMAGE_PATH = '../DeepSteganography-master/images/'
MODEL_PATH = '../DeepSteganography-master/model/'
filename = 'trainloss.txt'

def denormalize(image, std, mean):
    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image


def steg_loss(S_prime, C_prime, S, C, beta):
    loss_cover = F.mse_loss(C_prime, C)
    loss_secret = F.mse_loss(S_prime, S)
    loss = loss_cover + beta * loss_secret
    return loss, loss_cover, loss_secret

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# train_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(
#         TRAIN_PATH,
#         transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
#     batch_size=10, pin_memory=True, num_workers=1,
#     shuffle=True, drop_last=True)
#
# test_loader = torch.utils.DataLoader(
#     datasets.ImageFolder(
#         TEST_PATH,
#         transforms.Compose([
#             transforms.Scale(256),
#             transforms.RandomCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])),
#     batch_size=5, pin_memory=True, num_workers=1,
#     shuffle=True, drop_last=True)

# train_loader = transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomCrop(28),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
# ])
#
# test_loader = transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomCrop(28),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_loader)  # 训练数据集
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1,
#                                           drop_last=True)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取
#
# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_loader)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1,
#                                          drop_last=True)
# Data loading code
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])


# Training data loader
trainloader = DataLoader(
    MyDataset(TRAIN_PATH, transforms_=transform, ),
    batch_size=4,
    shuffle=True,
    pin_memory=True,
    num_workers=1,
)
# Test data loader
testloader = DataLoader(
    MyDataset(TRAIN_PATH , transforms_=transform, mode="val"),
    batch_size=4,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
)

model = StegNet().to(device)
dummy_input1 = torch.randn(2, 3, 224, 224)
dummy_input2 = torch.randn(2, 3, 224, 224)
dummy_input1 = dummy_input1.to(device)
dummy_input2 = dummy_input2.to(device)


if __name__ == "__main__":

    print(summary(model, [(3, 224, 224), (3, 224, 224)], device="cuda", batch_size=batch_size))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5,0.999))
    losses = []

    # 构建 SummaryWriter
    writer = SummaryWriter(comment='steg_program', filename_suffix="steg")
    writer.add_graph(model,(dummy_input1,dummy_input2))

    # -----------------------
    #  train
    # -----------------------

    for epoch in range(epochs):
        model.train()
        train_loss = []
        # print('epoch: {}'.format(epoch))

        for i, data in enumerate(trainloader):
            length = len(trainloader)
            inputs = data
            # print(inputs.size())
            # torchvision.utils.save_image(inputs[0],filename='image.jpg')

            inputs = inputs.to(device)
            covers = inputs[:len(inputs) // 2]
            secrets = inputs[len(inputs) // 2:]
            # print(covers.size())
            # print(secrets.size())
            covers = Variable(covers).to(device)
            secrets = Variable(secrets).to(device)

            optimizer.zero_grad()
            hidden, output = model(secrets, covers)
            loss, loss_cover, loss_secret = steg_loss(output, hidden, secrets, covers, beta)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            losses.append(loss.item())

            torch.save(model.state_dict(), MODEL_PATH + 'model.pkl')

            avg_train_loss = np.mean(train_loss)
            # print(avg_train_loss)
            # print(i)
            # print(loss.item())
            # print(loss_cover.item())
            # print(loss_secret.item())
            # print('Train Loss: {:.5f}, cover loss: {:.5f}, secret error: {:.5f}'.format(loss.item(),loss_cover.item(),loss_secret.item()))
            print(
                "[Epoch %d/%d] [Batch %d/%d] [Train Loss: %5f] [cover loss: %5f] [secret error: %5f]"
                % (epoch, epochs, i, len(trainloader), loss.item(),loss_cover.item(),loss_secret.item())
            )
        print('Epoch [{0}/{1}], Average loss: {2:.4f}'.format(epoch + 1, epochs, avg_train_loss))

        # 记录数据，保存于event file
        writer.add_scalars("Train_Loss", {"Train": avg_train_loss}, epoch)

        # 每个epoch，记录梯度，权值
        for name, param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)

        with open(filename, 'a') as f:
            f.write(str(avg_train_loss))
            f.write(str('\t'))

        # -----------------------
        #  val
        # -----------------------
        model.eval()

        test = []
        print("Waiting Test!")
        with torch.no_grad():
            for i, data in enumerate(testloader):
                test_images = data
                test_covers = test_images[:len(test_images) // 2]
                test_secrets = test_images[len(test_images) // 2:]

                # torchvision.utils.save_image(test_covers, filename=IMAGE_PATH + str(epoch) + '_covers.jpg')
                # torchvision.utils.save_image(test_secrets, filename=IMAGE_PATH + str(epoch) + '_secrets.jpg')

                test_covers = Variable(test_covers).to(device)
                test_secrets = Variable(test_secrets).to(device)

                test_hidden, test_output = model(test_secrets, test_covers)
                test_loss, test_loss_cover, test_loss_secret = steg_loss(test_output, test_hidden, test_secrets, test_covers, beta)
                test.append(test_loss.item())

                torchvision.utils.save_image(test_hidden, filename=IMAGE_PATH + str(epoch) + '_out' + '_covers.jpg')
                torchvision.utils.save_image(test_output, filename=IMAGE_PATH + str(epoch) + '_out' + '_secrets.jpg')
                test_hidden = torchvision.utils.make_grid(test_hidden, nrow=4, normalize=True, scale_each=True)
                test_output = torchvision.utils.make_grid(test_output, nrow=4, normalize=True, scale_each=True)
                writer.add_image("test_hidden img" + str(epoch), test_hidden, 0)
                writer.add_image("test_output img" + str(epoch), test_output, 0)

                # test_covers = test_covers.cpu()
                # test_secrets = test_secrets.cpu()
                # test_hidden = test_hidden.cpu()
                # test_output = test_output.cpu()
                # test_covers = test_covers.numpy()
                # test_secrets = test_secrets.numpy()
                # test_hidden = test_hidden.numpy()
                # test_output = test_output.numpy()

                # psnr_hidden = (calculate_psnr(test_secrets[0][0],test_hidden[0][0]) + calculate_psnr(test_secrets[0][1],test_hidden[0][1]) + calculate_psnr(test_secrets[0][2],test_hidden[0][2])) / 3
                # psnr_output = (calculate_psnr(test_covers[0][0],test_output[0][0]) + calculate_psnr(test_covers[0][1],test_output[0][1]) + calculate_psnr(test_covers[0][2],test_output[0][2])) / 3
                #
                # print("[psnr_hidden:  %d]" % psnr_hidden)
                # print("[psnr_output:  %d]" % psnr_output)

            avg_test_loss = np.mean(test)
            print('Epoch [{0}/{1}], Average loss: {2:.4f}'.format(epoch + 1, epochs, avg_test_loss))

            # 记录数据，保存于event file
            writer.add_scalars("Val_Loss", {"Val": avg_test_loss}, epoch)

    writer.close()

    # model.load_state_dict(torch.load(MODEL_PATH + 'model.pkl'))




