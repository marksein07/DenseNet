#!/usr/bin/env python3

import argparse
import numpy as np
#from environment import Environment

#seed = 11037
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data

import torchvision
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import shutil

from utils.utils import *
from model.DenseNet import DenseNet

print(torch.__version__)
# torch.manual_seed(1)



def parse():
    parser = argparse.ArgumentParser(description="DensetNet - 2020 by marksein07")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def dataloader(BATCH_SIZE):
    transformation = [
    #transforms.RandomGrayscale(p=1.0),
    transforms.RandomCrop(32),
    #transforms.ColorJitter(0.1,0.1,0.1,0.1),
    transforms.RandomHorizontalFlip(p=1.0),
    #transforms.RandomVerticalFlip(p=1.0), 
    ]
    random_transformation = [transforms.RandomChoice(transformation)]
    augmentation_transform = transforms.Compose(
        [transforms.RandomApply(random_transformation, p=1),
         transforms.ToTensor(), ])
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    normal_transform = transforms.Compose(
        [transforms.ToTensor(), ])
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=normal_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    augmentation_trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=augmentation_transform)
    augmentation_trainloader = torch.utils.data.DataLoader(augmentation_trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                           download=True, transform=normal_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader

def run():
    BATCH_SIZE = 64
    DOWNLOAD_MNIST = True
    preceed=False
    log='log'
    device = torch.device('cuda:0')
    
    LR = 1e-1

    trainloader, testloader = dataloader(BATCH_SIZE)
    
    cnn = DenseNet( growth_rate=(12,12,12), block_config=(40,40,40),
                   num_init_features=16, bn_size=4, drop_rate=0.2, num_classes=10, 
                   memory_efficient=True, bias=False)
    # !!!!!!!! Change in here !!!!!!!!! #
    cnn.cuda()      # Moves all model parameters and buffers to the GPU.
    #cnn = torch.nn.DataParallel(cnn,device_ids=[0,1]).to(device)
    if preceed :
        cnn.load_state_dict(torch.load("DenseNetL=100,k=12"))

    SGD     = torch.optim.SGD
    Adagrad = torch.optim.Adagrad
    Adam    = torch.optim.Adam

    opt = SGD

    optimizer = opt(cnn.parameters(), lr=LR, weight_decay=1e-4, momentum=0.9)
    #optimizer = opt(cnn.parameters(), lr=LR, weight_decay=1e-3)
    #semantic_optimizer = torch.optim.Adagrad(cnn.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    epoch = 0
    if os.path.isdir(log) :
        shutil.rmtree(log)
    writer = SummaryWriter(log)
    
    train(cnn, optimizer, criterion, trainloader, testloader, device, channel_normalization, writer, opt, LR)

    
if __name__ == '__main__':
    #args = parse()
    #run(args)
    run()