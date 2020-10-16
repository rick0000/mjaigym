import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from pytorch_memlab import profile

""" network model """

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        shortcut = x + h
        y = self.relu2(shortcut)
        return y


class Head34Net(nn.Module):
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int):
        super(Head34Net, self).__init__()
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        blocks = []
        for _i in range(blocks_num):
            blocks.append(ResBlock(mid_channels))

        self.res_blocks = nn.Sequential(*blocks)
        self.postproc = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=1, kernel_size=(1,1), padding=(0,0), bias=False),
        )

    def forward(self, x):
        x = self.preproc(x)
        x = self.res_blocks(x)
        x = self.postproc(x)
        x = x.view(x.size(0), -1)
        return x


class Head2Net(nn.Module):
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int):
        super(Head2Net, self).__init__()
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        blocks = []
        for _i in range(blocks_num):
            blocks.append(ResBlock(mid_channels))

        self.res_blocks = nn.Sequential(*blocks)
        self.postproc = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=32, kernel_size=(3,1), padding=(1,0), bias=False),
        )
        self.fc1 = nn.Linear(34*32, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.preproc(x)
        x = self.res_blocks(x)
        x = self.postproc(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        
        return x

class ActorCriticNet(nn.Module):
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int):
        super(ActorCriticNet, self).__init__()
        
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3,1), padding=(1,0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        blocks = []
        for _i in range(blocks_num):
            blocks.append(ResBlock(mid_channels))

        self.res_blocks = nn.Sequential(*blocks)
        self.postproc = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=1, kernel_size=(1,1), padding=(0,0), bias=False),
        )

        # for v head
        self.v_post_proc = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=32, kernel_size=(3,1), padding= (1,0), bias=False),
            nn.BatchNorm2d(32),
            )
        self.v_fc1 = nn.Linear(34*32, 1024)
        self.v_fc2 = nn.Linear(1024, 256)
        self.v_out = nn.Linear(256, 1)

    def forward(self, x):
        h = self.preproc(x)
        h = self.res_blocks(h)
        x = self.postproc(h)
        x = x.view(x.size(0), -1)

        v = self.v_post_proc(h)
        v = v.view(v.size(0), -1)
        v = self.v_fc1(v)
        v = self.v_fc2(v)
        v = self.v_out(v)
        

        return x, v

    def forward_with_v_mid(self, x):
        h = self.preproc(x)
        h = self.res_blocks(h)
        x = self.postproc(h)
        x = x.view(x.size(0), -1)

        v = self.v_post_proc(h)
        v_mid = v.view(v.size(0), -1)
        v = self.v_fc1(v_mid)
        v = self.v_fc2(v)
        v = self.v_out(v)
        return x, v, v_mid,