#!/usr/bin/env python

##  playing_with_skip_connections.py

"""
This script illustrates how to actually use the inner class SkipConnections of
the DLStudio module.

As shown in the calls below, a CNN is constructed by calling on the constructor for
the BMEnet class.

You can easily create a CNN with arbitrary depth just by using the "depth"
constructor option for the BMEnet class.  BMEnet creates a network by using
multiple blocks of SkipBlock.
"""

import random
import numpy
import torch
import os, sys


import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
import re
import math
import random
import copy
import matplotlib.pyplot as plt

import gzip
import pickle

seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *
#"/home/kak/ImageDatasets/CIFAR-10/"
dls = DLStudio(
                  dataroot = "H:\\DLStudio-1.0.7\\Examples",
                  image_size = [32,32],
                  path_saved_model = "C:\\temp\\saved_model_resnext",
                  momentum = 0.9,
                  learning_rate = 1e-4,
                  epochs = 1,
                  batch_size = 4,
                  classes = ('plane','car','bird','cat','deer',
                             'dog','frog','horse','ship','truck'),
                  debug_train = 0,
                  debug_test = 0,
                  use_gpu = True,
              )


''' 
Before implementing a new skip connection, I have gone through multiple building blocks
BasicBlock in Resnet18, Bottleneckblock in Resnet50, Denseblock in Densenet, building block 
of ResNext and deep network with stochastic depth.

Whats new Implemented?
I implemented the skipconnection with ResNext Building block with cardinality 32. Each 32 
paths in ResNext has a Conv1×1–Conv3×3–Conv1×1 layers. Here the cardinality controls more
complex operations, it also increases the number of learnable parameters from 5.5M (in 
Resnet18 previous BMEnet with skipconnection) to 65.8M. Hence training loss is much better 
than the previous BMEnet. Here, the middle layer of the skip connection is a group convolution
where # of groups = cardinality of ResNext building block. One can easily change this cardinality.
After that I have performed the batchnormalization and ReLU operation on the output.

Although ResNext able to solve the vanishing gradient problem but it suffers from slow training process. 
To solve that problem and in order to add another dimension I have implemented deep
network with stochastic depth, where whether to use the stacked Conv1×1–Conv3×3 (grouped)–Conv1×1 layers
depends on a random number generated (either 0 or 1 following a bernoulli distribution) while training time. 


I have modified BMEnet also. After the conv2d-maxpool-ResNextSkipBlock(depth 4)-FC1-FC2. As 
Resnext skipblock has 256 input channel and 256 output channel. I have changed the # of filters
in conv2d layer from 64 to 256. Also changed the number of input neurons of FC1 to 256*256.

Conclusion?
I have tried 4 skipconnections on cifar10.
1)previous skipconnection (Basic Block) -- (74%, 0.216)
2)skipconnection with stochastic depth -- (73%, 0.254)
3)ResNext skipconnection -- (76%, 0.071)
4)ResNext SkipConnection with stochastic depth-- (72%, 0.227)

Here I have given the classification accuracy on CIFAR-10 and the loss after 10 epochs.
As one can clearly observe the ResNext skipconnection provides much better loss and 
classification accuracy. Hence I here only used that. I have also added the other two implementations
(skipconnection with stochastic depth & resnext skip connection with stochastic depth) here also
'''

#=============================
#SkipConnections for ResNext with Cardinality 32
#=============================

#ResNeXt skipconnection
class SkipConnections(nn.Module):
    def __init__(self):
        super(SkipConnections, self).__init__()
        #self.dl_studio = dl_studio
    
    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipConnections.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            #self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.conv1 = nn.Conv2d(256, 128, 1, stride=1)
            self.conv2 = nn.Conv2d(128,128,3,stride=1, padding=1,groups=32)
            self.conv3 = nn.Conv2d(128,256,1,stride=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x                                     
            #out = self.convo(x)
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.bn(out)                              
            out = torch.nn.functional.relu(out)
           
            #if self.in_ch == self.out_ch:
            #    out = self.convo(out)                              
            #    out = self.bn(out)                              
            #    out = torch.nn.functional.relu(out)
  
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity                              
                else:
                    out[:,:self.in_ch,:,:] += identity
                    out[:,self.in_ch:,:,:] += identity
            return out

    class BMEnet(nn.Module):
        def __init__(self, skip_connections=True, depth=32):
            super(SkipConnections.BMEnet, self).__init__()
            self.pool_count = 3
            self.depth = depth // 2
            #self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.conv = nn.Conv2d(3, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.skip64 = SkipConnections.SkipBlock(256, 256, skip_connections=skip_connections)
           
          
            self.fc1 =  nn.Linear(65536, 1000)
            self.fc2 =  nn.Linear(1000, 10)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv(x)))
            #print("\n after conv+relu+pool \n")
            #print(x.size())
            for _ in range(self.depth // 4):
                x = self.skip64(x)  
            #print("\n after skip64 \n")
            #print(x.size())
            x = x.view(-1, 65536 )
            #x = x.view(-1, 65536 )
            #print(x.size())
            x = torch.nn.functional.relu(self.fc1(x))
            #print("\n after fc1+relu \n")
            #print(x.size())
            x = self.fc2(x)
            #print("\n after fc2 \n")
            #print(x.size())
            return x            




"""
#=============================
#SkipConnections for deep network with stochastic depth
#=============================
class SkipConnections(nn.Module):
    def __init__(self):
        print("\n I am here \n")
        super(SkipConnections, self).__init__()
        #self.dl_studio = dl_studio
    
    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            print("\n Inside KaK \n")
            super(SkipConnections.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x    
            n, p = 1, .5
            s = np.random.binomial(n, p, 1)
            #print("\n current random no = %d"%s[0])
            if(s[0]==1):
                out = self.convo(x)                              
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
                if self.in_ch == self.out_ch:
                    out = self.convo(out)                              
                    out = self.bn(out)                              
                    out = torch.nn.functional.relu(out)
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
            else:
                out = x
            return out

    class BMEnet(nn.Module):
        def __init__(self, skip_connections=True, depth=32):
            super(SkipConnections.BMEnet, self).__init__()
            self.pool_count = 3
            self.depth = depth // 2
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.skip64 = SkipConnections.SkipBlock(64, 64, skip_connections=skip_connections)
            self.fc1 =  nn.Linear(16384, 1000)
            self.fc2 =  nn.Linear(1000, 10)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv(x)))
            #print("\n after conv+relu+pool \n")
            #print(x.size())
            for _ in range(self.depth // 4):
                x = self.skip64(x)
            x = x.view(-1, 16384 )
            #print(x.size())
            x = torch.nn.functional.relu(self.fc1(x))
            #print("\n after fc1+relu \n")
            #print(x.size())
            x = self.fc2(x)
            #print(x.size())
            return x 
"""

"""
#=============================
#SkipConnections for ResNext with stochastic depth
#=============================
class SkipConnections(nn.Module):
    def __init__(self):
        print("\n I am here \n")
        super(SkipConnections, self).__init__()
        #self.dl_studio = dl_studio
    
    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipConnections.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            #self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.conv1 = nn.Conv2d(256, 128, 1, stride=1)
            self.conv2 = nn.Conv2d(128,128,3,stride=1, padding=1,groups=32)
            self.conv3 = nn.Conv2d(128,256,1,stride=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            n, p = 1, 0.5
            s = np.random.binomial(n, p, 1)
            #out = self.convo(x)
            if(s[0]==1):
                out = self.conv1(x)
                out = self.conv2(out)
                out = self.conv3(out)
                out = self.bn(out)                              
                out = torch.nn.functional.relu(out)
               
                #if self.in_ch == self.out_ch:
                #    out = self.convo(out)                              
                #    out = self.bn(out)                              
                #    out = torch.nn.functional.relu(out)
      
                if self.downsample:
                    out = self.downsampler(out)
                    identity = self.downsampler(identity)
                if self.skip_connections:
                    if self.in_ch == self.out_ch:
                        out += identity                              
                    else:
                        out[:,:self.in_ch,:,:] += identity
                        out[:,self.in_ch:,:,:] += identity
            else:
                out = x
            return out

    class BMEnet(nn.Module):
        def __init__(self, skip_connections=True, depth=32):
            super(SkipConnections.BMEnet, self).__init__()
            self.pool_count = 3
            self.depth = depth // 2
            #self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.conv = nn.Conv2d(3, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.skip64 = SkipConnections.SkipBlock(256, 256, skip_connections=skip_connections)
           
          
            self.fc1 =  nn.Linear(65536, 1000)
            self.fc2 =  nn.Linear(1000, 10)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv(x)))
            #print("\n after conv+relu+pool \n")
            #print(x.size())
            for _ in range(self.depth // 4):
                x = self.skip64(x)  
            #print("\n after skip64 \n")
            #print(x.size())
            x = x.view(-1, 65536 )
            #x = x.view(-1, 65536 )
            #print(x.size())
            x = torch.nn.functional.relu(self.fc1(x))
            #print("\n after fc1+relu \n")
            #print(x.size())
            x = self.fc2(x)
            #print("\n after fc2 \n")
            #print(x.size())
            return x   
"""

            
             
def main():
    exp_skip = DLStudio.SkipConnections( dl_studio = dls )

    #exp_skip.load_cifar_10_dataset_with_augmentation()
    exp_skip.load_cifar_10_dataset()

    #model = exp_skip.BMEnet(skip_connections=False, depth=32)
    spcon = SkipConnections() #using the ResNext SkipConnection
    model = spcon.BMEnet(skip_connections=True, depth=32)

    dls.show_network_summary(model)

    exp_skip.run_code_for_training(model)

    exp_skip.run_code_for_testing(model)

if __name__== "__main__":
  main()

