# creado por Diego Valencia
import os
import sys
import random
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial import distance

import torchvision
import torchvision.transforms as transforms

from utils import ImageCaptionDataset, train_for_classification, train_for_retrieval
import matplotlib.pyplot as plt


def plot_results(loss, score1, score1_title='Accuracy', score2=None, score2_title=None):
    f1 = plt.figure(1)
    ax1 = f1.add_subplot(111)
    ax1.set_title("Loss")
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.plot(loss, c='r')
    ax1.legend(['train-loss'])
    f1.show()

    f2 = plt.figure(2)
    ax2 = f2.add_subplot(111)
    ax2.set_title(score1_title)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel(score1_title.lower())
    ax2.plot(score1[0], c='b')
    ax2.plot(score1[1], c='g')
    ax2.legend([f'train-{score1_title.lower()}',
                f'val-{score1_title.lower()}'])
    f2.show()

    if score2:
        f3 = plt.figure(3)
        ax3 = f3.add_subplot(111)
        ax3.set_title(score2_title)
        ax3.set_xlabel('epochs')
        ax3.set_ylabel(score2_title.lower())
        ax3.plot(score2[0], c='b')
        ax3.plot(score2[1], c='g')
        ax3.legend([f'train-{score2_title.lower()}',
                    f'val-{score2_title.lower()}'])
        f3.show()


class Bloque(nn.Module):
    expansion = 1

    def __init__(self, inplane, plane, stride=1, groups=1, width=64, dilation=1, primera=False, ultimo=False):
        super(Bloque, self).__init__()
        # se le pasa el stride como parametro a la primera convolucion para el caso
        # donde solo se le aplica a la primera conv y en caso contrario defaults a 1
        self.primera = primera
        self.ultimo = ultimo
        # print(self.primera,self.ultimo,"--------")
        self.conv1 = nn.Conv2d(in_channels=inplane, out_channels=plane,
                               kernel_size=3, stride=stride, padding=1)
        # if primera:
        #  self.conv1 = nn.Conv2d(in_channels=inplane,out_channels=plane,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=plane, out_channels=plane, kernel_size=3, padding=1)
        # if primera:
        #  self.conv2 = nn.Conv2d(in_channels=plane,out_channels=plane,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(plane)
        self.stride = stride
        self.maxpoolAUX = nn.MaxPool2d((3, 3), stride=1, padding=0)

    def forward(self, x):
        inicio = x
#    if self.primera:
#      print("una vez",self.primera)
#      inicio =self.maxpoolAUX(inicio)
        self.primera = False
        salida = self.conv1(x)
        # print("conv1,",salida.size())
        salida = self.bn1(salida)
        # print("bn1,",salida.size())
        salida = self.relu(salida)
        # print("relu,",salida.size())

        salida = self.conv2(salida)
        # print("conv2,",salida.size())
        salida = self.bn2(salida)
        # print("bn2,",salida.size())

        # se juntan para obtener la funcionalidad de resnet
        # print("salida:",salida.size(),"inicio:",inicio.size()[3],"\n")

        salida += inicio
        # if self.ultimo:
        #  print("antes mxpoolaux",salida.size())
        #  salida = self.maxpoolAUX(salida)
        #  salida = self.maxpoolAUX(salida)
        #  print("estoy aplicando maxpoolaux",salida.size())
        salida = self.relu(salida)
        return salida


class ResNet(nn.Module):
    def __init__(self, n_classes, bloque, capas, groups=1, width_g=64):
        super(ResNet, self).__init__()
        self.inplane = 64
        self.dilation = 1

        self.groups = 1
        self.width = width_g

        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.capa1 = self.crear_capa_residuo(bloque, 64, capas[0])
        self.cambiodim1 = nn.Conv2d(64, 128, 1, 1)
        self.capa2 = self.crear_capa_residuo(bloque, 128, capas[1])
        self.cambiodim2 = nn.Conv2d(128, 256, 1, 1)
        self.capa3 = self.crear_capa_residuo(bloque, 256, capas[2])
        self.cambiodim3 = nn.Conv2d(256, 512, 1, 1)
        self.capa4 = self.crear_capa_residuo(bloque, 512, capas[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)
        # print(n_classes)

    def crear_capa_residuo(self, bloque, plane, bloques, stride=1, primera=False, ultima=False):
        capas = []

        # print(primera)
        self.inplane = plane
        # if primera:
        #  capas.append(bloque(self.inplane,plane,stride,self.groups,self.width,primera=primera))
        # else:
        #  capas.append(bloque(self.inplane,plane,stride,self.groups,self.width))
        capas.append(bloque(self.inplane, plane,
                            stride, self.groups, self.width))

        # if primera:
        #  for i in range(1,bloques):
        #    if i == bloques-1:
        #      capas.append(bloque(self.inplane,plane,width=self.width,primera=primera,ultimo = ultima))
        #    else:
        #      capas.append(bloque(self.inplane,plane,width=self.width,primera=primera))
        # else:
        #  for i in range(1,bloques):
        #    capas.append(bloque(self.inplane,plane,width=self.width))
        for i in range(1, bloques):
            capas.append(bloque(self.inplane, plane, width=self.width))

        return nn.Sequential(*capas)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.capa1(x)
        x = self.cambiodim1(x)

        x = self.capa2(x)
        x = self.cambiodim2(x)

        x = self.capa3(x)
        x = self.cambiodim3(x)

        x = self.capa4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        # return x
        return {'logits': x}


# ResNet con Cifar-10
torch.cuda.empty_cache()
BATCH_SIZE = 128
LR = 0.01
EPOCHS = 9
REPORTS_EVERY = 1

# modelo
net2 = ResNet(10, Bloque, [3, 4, 6, 3])
# optimizador
optimizer2 = optim.SGD(net2.parameters(), momentum=0.9, lr=LR)
criterion2 = nn.CrossEntropyLoss()  # función de pérdida
# schduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer2, step_size=4, gamma=0.1)

train_loader2 = DataLoader(trainset, batch_size=BATCH_SIZE,
                           shuffle=True, num_workers=2)
test_loader2 = DataLoader(testset, batch_size=64,
                          shuffle=False, num_workers=2)

train_loss2, acc2 = train_for_classification(net2, train_loader2,
                                             test_loader2, optimizer2,
                                             criterion2, lr_scheduler=scheduler,
                                             epochs=EPOCHS, reports_every=REPORTS_EVERY)

plot_results(train_loss2, acc2)
