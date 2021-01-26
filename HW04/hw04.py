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
import os, sys, glob


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
from torchsummary import summary

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

#traindir = "H:\\tiny-imagenet-200\\tiny-imagenet-200\\train\\"
traindir = "H:\\DLStudio-1.0.7\\Examples\\imagenet_images\\train\\"
#traindir = "H:\\DLStudio-1.0.7\\Examples\\train_all\\"
#testdir = "H:\\tiny-imagenet-200\\tiny-imagenet-200\\test\\"
testdir = "H:\\DLStudio-1.0.7\\Examples\\imagenet_images\\test\\"
epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4 
momentum = 0.9
rect_size = 64
image_size = [rect_size, rect_size]
batch_size = 4
path_saved_model = "C:\\temp\\saved_model_imagenet_resnext_5"
classes = glob.glob(traindir + "*")
"""
class_labels = []
for i in classes:
    class_labels.append(i.split("\\")[-1])
"""
class_labels = ('bicycle', 'boat', 'cat', 'dog', 'truck')
print("\n # of classes = ", len(class_labels))
log_point = 250

''' 
Before implementing a new skip connection, I have gone through multiple building blocks
BasicBlock in Resnet18, Bottleneckblock in Resnet50, Denseblock in Densenet, building block 
of ResNext and deep network with stochastic depth.

Whats new Implemented?
I implemented the skipconnection with ResNext Building block with cardinality 32. Each 32 
paths in ResNext has a Conv1×1–Conv3×3–Conv1×1 layers. Here the cardinality controls more
complex operations, it also increases the number of learnable parameters from 5.5M (in 
Resnet18 previous BMEnet with skipconnection) to 65.8M. For Imagenet, the learnable parameters
are 4 times than of CIFAR-10. For ResNext its 262.4M. Hence training loss is much better 
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
(skipconnection with stochastic depth & resnext skip connection with stochastic depth) here also.
I have also finally used ResNext for Imagenet dataset for 5 classes and the dimension of input images
modified to be 64*64 in order to eliminate the CUDA memory allocation error problem.

Reference:
From DLStudio, I have used the training and testing and partially modified it so that I can pass
the traindataloader and testdataloader for the ImageNet.
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
            #for cifar10 (4,256,16,16) so (4,65536)
            #for imagenet (4, 256,32,32)
           
          
            self.fc1 =  nn.Linear(int((image_size[0]*image_size[1]*256)/4), 1000)
            self.fc2 =  nn.Linear(1000, len(class_labels))

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv(x)))
            #print("\n after conv+relu+pool :", x.size())
            for _ in range(self.depth // 4):
                x = self.skip64(x)  
            #print("\n after skip64 :", x.size())
            x = x.view(-1, int((image_size[0]*image_size[1]*256)/4) )
            #x = x.view(-1, 65536 )
            #print("\n after changing view :", x.size())
            x = torch.nn.functional.relu(self.fc1(x))
            #print("\n after fc1+relu :", x.size())
            x = self.fc2(x)
            #print("\n after fc2 :", x.size())
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



def load_imagenet_tiny(traindir, testdir):
    transform = tvt.Compose([tvt.ToTensor(),tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #train_dataset = torchvision.datasets.ImageFolder(traindir,transform=transform)
    train_dataset = torchvision.datasets.ImageFolder(traindir,tvt.Compose([tvt.RandomResizedCrop(image_size[0]),transform]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    print("\n .... after traindataloader .......\n")
    #test_dataset = torchvision.datasets.ImageFolder(testdir,transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(testdir,tvt.Compose([tvt.RandomResizedCrop(image_size[0]),transform]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    print("\n ...... after testloader ....... \n")
    return train_loader, test_loader

def run_code_for_training(train_data_loader, net):        
    #filename_for_out = "performance_numbers_" + str(epochs) + ".txt"
    #FILE = open(filename_for_out, 'w')
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(epochs):  
        ##  We will use running_loss to accumulate the losses over 2000 batches in order
        ##  to present an averaged (over 2000) loss to the user.
        #print("\n")
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            #print(i)
            inputs, labels = data
            #print(i, labels)
            """
            if i % log_point == log_point-1:
            #if i % 2000 == 1999:
                print("\n\n[iter=%d:] Ground Truth:     " % (i+1) + 
                ' '.join('%5s' % labels[j] for j in range(batch_size)))
            """
            inputs = inputs.to(device)
            labels = labels.to(device)
            ##  Since PyTorch likes to construct dynamic computational graphs, we need to
            ##  zero out the previously calculated gradients for the learnable parameters:
            optimizer.zero_grad()
            # Make the predictions with the model:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            """
            if i % log_point == log_point-1:
                _, predicted = torch.max(outputs.data, 1)
                print("[iter=%d:] Predicted Labels: " % (i+1) + 
                 ' '.join('%5s' % predicted[j] for j in range(batch_size)))
                #self.display_tensor_as_image(torchvision.utils.make_grid(inputs, normalize=True), 
                #                        "see terminal for TRAINING results at iter=%d" % (i+1))
            """
            loss.backward()
            optimizer.step()
            ##  Present to the average value of the loss over the past 2000 batches:            
            running_loss += loss.item()
            if i % log_point == log_point-1:
            #if i % 2000 == 1999:    
#                    print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                #avg_loss = running_loss / float(2000)
                avg_loss = running_loss / float(log_point)
                print("[epoch:%d] loss: %.3f" % (epoch + 1, avg_loss))
                #FILE.write("%.3f\n" % avg_loss)
                #FILE.flush()
                running_loss = 0.0
    print("\nFinished Training\n")
    torch.save(net.state_dict(), path_saved_model)

def show_network_summary(net):
    print("\n\n\nprinting out the model:")
    print(net)
    print("\n\n\na summary of input/output for the model:")
    summary(net, (3,image_size[0],image_size[1]),-1, device='cpu')
    
def run_code_for_testing(test_data_loader, net):
    net.load_state_dict(torch.load(path_saved_model))
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(len(class_labels), len(class_labels))
    class_correct = [0] * len(class_labels)
    class_total = [0] * len(class_labels)
    with torch.no_grad():
        for i,data in enumerate(test_data_loader):
            ##  data is set to the images and the labels for one batch at a time:
            images, labels = data
            #print(i, labels)
            if i % log_point == log_point-1:
            #if i%2000 == 1999:
                print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%5s' % class_labels[labels[j]] 
                                                           for j in range(batch_size)))
            outputs = net(images)
            ##  max() returns two things: the max value and its index in the 10 element
            ##  output vector.  We are only interested in the index --- since that is 
            ##  essentially the predicted class label:
            _, predicted = torch.max(outputs.data, 1)
            for label,prediction in zip(labels,predicted):
                    confusion_matrix[label][prediction] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            ##  comp is a list of size batch_size of "True" and "False" vals
            comp = predicted == labels       
            for j in range(batch_size):
                label = labels[j]
                class_correct[label] += comp[j].item()
                class_total[label] += 1
    for j in range(len(class_labels)):
        print('Prediction accuracy for %5s : %2d %%' % (
                           class_labels[j], 100 * class_correct[j] / class_total[j]))
    print("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%" % 
                                                           (100 * correct / float(total)))
    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "         "
    for j in range(len(class_labels)):  out_str +=  "%7s" % class_labels[j]   
    print(out_str + "\n")
    for i,label in enumerate(class_labels):
        out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                                  for j in range(len(class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%6s:  " % class_labels[i]
        for j in range(len(class_labels)): out_str +=  "%7s" % out_percents[j]
        print(out_str)    


def main():

    #"""
    train_loader, test_loader = load_imagenet_tiny(traindir, testdir)
    print(len(train_loader), len(test_loader))
    #loading network
    spcon = SkipConnections() #using the ResNext SkipConnection
    model = spcon.BMEnet(skip_connections=True, depth=32)
    show_network_summary(model)
    #start training
    run_code_for_training(train_loader, model)
    run_code_for_testing(test_loader, model)
    #"""

if __name__== "__main__":
  main()