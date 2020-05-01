#!/usr/bin/env python3

import argparse
import numpy as np
#from environment import Environment

#seed = 7704
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data

import torchvision
from torchvision import transforms

import numpy as np

from tensorboardX import SummaryWriter

from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import shutil

from utils.utils import *
from model.DenseNet import DenseNet

print(torch.__version__)
# torch.manual_seed(1)



def parse():
    parser = argparse.ArgumentParser(description="DensetNet - 2020 by marksein07")
    #parser.add_argument('--optimizer')
    parser.add_argument('--batch_size', type=int,default=64, help='traing and testing batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer L2 penalty')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--cuda', type=str, default='0', help='GPU Index for training')
    parser.add_argument('--log', type=str, default='../result/log', help='tensorboard log directory')
    parser.add_argument('--preceed', type=bool, default=False, help='whether load a pretrain model')
    parser.add_argument('--training_epoch', type=int, default=300, help='total training epoch')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def dataloader(BATCH_SIZE, download=True, shuffle=True, augmentation=False):
    normal_transformations = transforms.Compose( [transforms.ToTensor(), ])

    trainset = torchvision.datasets.CIFAR10(root='../result/cifar10', train=True, download=download, transform=normal_transformations)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../result/cifar10', train=False, download=download, transform=normal_transformations)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
    if augmentation :
        transformation = [
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(p=1.0),
        ]

        augmentation_transformation = transforms.Compose( [
            transforms.RandomChoice(transformation),
            transforms.ToTensor(),
        ] )

        augmentation_trainset = torchvision.datasets.CIFAR10(root='../result/cifar10', train=True, download=download, transform=augmentation_transformation)
        augmentation_trainloader = torch.utils.data.DataLoader(augmentation_trainset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)
        return trainloader, testloader, augmentation_trainloader


    return trainloader, testloader
def plot(*data_points_list, filename="plot", log_scale=True) :
    if log_scale :
        plt.yscale('log')
    for data_points in data_points_list :
        plt.plot(data_points)
    plt.savefig(filename)
    plt.close()
def training_setting(model, optimizer, lr, device, log, criterion, ):
    pass

def run(args):
    BATCH_SIZE = args.batch_size
    preceed = args.preceed
    log = args.log
    device = torch.device('cuda:'+args.cuda)
    LR = args.learning_rate

    trainloader, testloader = dataloader(BATCH_SIZE)
    
    model = DenseNet( growth_rate=12, DenseBlock_layer_num=(40,40,40), 
                     bottle_neck_size=4, dropout_rate=0.2, compression_rate=0.5, num_init_features=16,
                     num_input_features=3, num_classes=10, bias=False, memory_efficient=False)
    # !!!!!!!! Change in here !!!!!!!!! #
    if len(args.cuda) == 1 :
        model.to(device)      # Moves all model parameters and buffers to the GPU.
    elif len(args.cuda) > 1 :
        model = torch.nn.DataParallel(model,device_ids=[0,1]).to(device)
        
    if preceed :
        model.load_state_dict(torch.load("DenseNetL=100,k=12"))

    SGD     = torch.optim.SGD
    Adagrad = torch.optim.Adagrad
    Adam    = torch.optim.Adam

    opt = SGD

    optimizer = opt( model.parameters(),
                    lr           = args.learning_rate,
                    weight_decay = args.weight_decay,
                    momentum     = args.momentum )
    #optimizer = opt(model.parameters(), lr=LR, weight_decay=1e-3)
    #semantic_optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    epoch = 0
    if os.path.isdir(log) :
        shutil.rmtree(log)
    writer = SummaryWriter(log)
    
    training_error_rate, testing_error_rate, \
    training_loss, testing_loss = train(model, optimizer, criterion, trainloader, testloader, 
                                        device, channel_normalization, writer, opt, LR, 
                                        max_epoch = args.training_epoch, )
    plot(training_loss, testing_loss, filename='../result/loss')
    plot(training_error_rate, testing_error_rate, filename='../result/error_rate', log_scale=False)
    
if __name__ == '__main__':
    args = parse()
    run(args)
