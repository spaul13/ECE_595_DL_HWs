#Using DLStudio Latest version (1.1.1)

#Task 4: In this task, noise level detection result is used intelligently to assign LOADnet2 models
#with different depth for different noise levels. Higher depth of LOADnet2 is used for dataset with
#higher noise level. 
#noise_level --> depth of LOADnet2
# 0%         -->    8
# 20%         -->   12
# 50%         -->   14
# 80%         -->   16
#with the noise level information beforehand for 50% noise level, the accuracy improved from 
#60% to 64% (only this improvement is recorded from this script).
#Here I am providing the accuracy improvement for DLStudio 1.1.1 and 1.1.0
#noise_level --> Accuracy improvement(depth)--DLS 1.1.1
# 20%         -->   4%(12)
# 50%         -->   4%(14)
# 80%         -->   3%(16)
#noise_level --> Accuracy improvement(depth)--DLS 1.1.0
# 20%         -->   3%(64)
# 50%         -->   5%(128)
# 80%         -->   9%(256)
#The accuracy improvement is recorded as before applying LOADnet2 with adaptive depth for adaptive
#noise level datasets to make LOADnet2 robust to noise levels. I hope if the depth of current 
#LOADnet2 (under version DLStudio 1.1.1) can be increased the accuracy improvement will be more.
    



import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import pymsgbox
import skimage
from scipy.signal import convolve2d
from skimage.restoration import estimate_sigma
import itertools
from sklearn.utils import shuffle
import operator



seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)


##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

#noise_level = []
dls = DLStudio(
                  dataroot = "H:\\DLStudio-1.1.0\\Examples\\data\\",
                  image_size = [32,32],
                  path_saved_model = "C:\\temp\\hw6_model",
                  momentum = 0.9,
                  learning_rate = 1e-5,
                  epochs = 2,
                  batch_size = 4,
                  classes = ('rectangle','triangle','disk','oval','star'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
detector = DLStudio.DetectAndLocalize( dl_studio = dls )
dataset_train_list = ["PurdueShapes5-10000-train.gz","PurdueShapes5-10000-train-noise-20.gz","PurdueShapes5-10000-train-noise-50.gz", "PurdueShapes5-10000-train-noise-80.gz"]
dataset_test_list = ["PurdueShapes5-1000-test.gz","PurdueShapes5-1000-test-noise-20.gz","PurdueShapes5-1000-test-noise-50.gz", "PurdueShapes5-1000-test-noise-80.gz"]
noise_level_list = [0,20,50,80]
noise_path_saved_model = "C:\\temp\\new_noise_detection_model"

class myPurdueShapes5Dataset(torch.utils.data.Dataset):
    def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
        super(myPurdueShapes5Dataset, self).__init__()
        self.noise_level = []
        if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                #print(len(self.dataset))
                for i in range(len(self.dataset)):
                    #print(i,len(self.dataset[i]))
                    if(len(self.dataset[i])<6):
                        self.dataset[i].append(0)
                    #print(i,len(self.dataset[i]))
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                #print(self.label_map)
                for i in range(len(self.dataset)):
                    self.dataset[i].append(0)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                print(self.class_labels)
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                for i in range(len(self.dataset)):
                    if(len(self.dataset[i])<6):
                        self.dataset[i].append(1)
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                for i in range(len(self.dataset)):
                    self.dataset[i].append(1)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                for i in range(len(self.dataset)):
                    if(len(self.dataset[i])<6):
                        self.dataset[i].append(2)
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                for i in range(len(self.dataset)):
                    self.dataset[i].append(2)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
            if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                      os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                print("\nLoading training data from the torch-saved archive")
                self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                for i in range(len(self.dataset)):
                    if(len(self.dataset[i])<6):
                        self.dataset[i].append(3)
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
            else: 
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                for i in range(len(self.dataset)):
                        self.dataset[i].append(3)
                torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform
        else:
            root_dir = dl_studio.dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if sys.version_info[0] == 3:
                self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
            else:
                self.dataset, self.label_map = pickle.loads(dataset)
            if dataset_file == "PurdueShapes5-1000-test.gz":
                for i in range(len(self.dataset)):
                    self.dataset[i].append(0)
            elif dataset_file == "PurdueShapes5-1000-test-noise-20.gz":
                for i in range(len(self.dataset)):
                    self.dataset[i].append(1)
            elif dataset_file == "PurdueShapes5-1000-test-noise-50.gz":
                for i in range(len(self.dataset)):
                    self.dataset[i].append(2)
            else:
                 for i in range(len(self.dataset)):
                    self.dataset[i].append(3)
            
            # reverse the key-value pairs in the label dictionary:
            self.class_labels = dict(map(reversed, self.label_map.items()))
            self.transform = transform
     
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        r = np.array( self.dataset[idx][0] )
        g = np.array( self.dataset[idx][1] )
        b = np.array( self.dataset[idx][2] )
        R,G,B = r.reshape(32,32), g.reshape(32,32), b.reshape(32,32)
        im_tensor = torch.zeros(3,32,32, dtype=torch.float)
        im_tensor[0,:,:] = torch.from_numpy(R) 
        im_tensor[1,:,:] = torch.from_numpy(G)
        im_tensor[2,:,:] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
        #print("\n inside new getitem \n")
        sample = {'image' : im_tensor, 
                  'bbox' : bb_tensor,
                  'label' : self.dataset[idx][4],
                  'noise_label' : self.dataset[idx][5]}
        if self.transform:
             sample = self.transform(sample)
        return sample
  
def custom_run_code_for_training_with_CrossEntropy_and_MSE_Losses(net, train_dataloader):        
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), 
                 lr=dls.learning_rate, momentum=dls.momentum)
    for epoch in range(dls.epochs):  
        running_loss_labeling = 0.0
        running_loss_regression = 0.0       
        for i, data in enumerate(train_dataloader):
            gt_too_small = False
            inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
            inputs = inputs.to(dls.device)
            labels = labels.to(dls.device)
            bbox_gt = bbox_gt.to(dls.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs_label = outputs[0]
            bbox_pred = outputs[1]
            if dls.debug_train and i % 500 == 499:
#                  if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                inputs_copy = inputs.detach().clone()
                inputs_copy = inputs_copy.cpu()
                bbox_pc = bbox_pred.detach().clone()
                bbox_pc[bbox_pc<0] = 0
                bbox_pc[bbox_pc>31] = 31
                bbox_pc[torch.isnan(bbox_pc)] = 0
                _, predicted = torch.max(outputs_label.data, 1)
                for idx in range(dls.batch_size):
                    i1 = int(bbox_gt[idx][1])
                    i2 = int(bbox_gt[idx][3])
                    j1 = int(bbox_gt[idx][0])
                    j2 = int(bbox_gt[idx][2])
                    k1 = int(bbox_pc[idx][1])
                    k2 = int(bbox_pc[idx][3])
                    l1 = int(bbox_pc[idx][0])
                    l2 = int(bbox_pc[idx][2])
                    print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                    print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                    inputs_copy[idx,0,i1:i2,j1] = 255
                    inputs_copy[idx,0,i1:i2,j2] = 255
                    inputs_copy[idx,0,i1,j1:j2] = 255
                    inputs_copy[idx,0,i2,j1:j2] = 255
                    inputs_copy[idx,2,k1:k2,l1] = 255                      
                    inputs_copy[idx,2,k1:k2,l2] = 255
                    inputs_copy[idx,2,k1,l1:l2] = 255
                    inputs_copy[idx,2,k2,l1:l2] = 255
#                        self.dl_studio.display_tensor_as_image(
#                              torchvision.utils.make_grid(inputs_copy, normalize=True),
#                             "see terminal for TRAINING results at iter=%d" % (i+1))
            loss_labeling = criterion1(outputs_label, labels)
            loss_labeling.backward(retain_graph=True)        
            loss_regression = criterion2(bbox_pred, bbox_gt)
            loss_regression.backward()
            optimizer.step()
            running_loss_labeling += loss_labeling.item()    
            running_loss_regression += loss_regression.item()                
            if i % 500 == 499:    
                avg_loss_labeling = running_loss_labeling / float(500)
                avg_loss_regression = running_loss_regression / float(500)
                print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                running_loss_labeling = 0.0
                running_loss_regression = 0.0

    print("\nFinished Training\n")
    return net

def custom_run_code_for_testing_detection_and_localization(net, test_dataloader, dataserver_train):
    #net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(len(dataserver_train.class_labels), 
                                   len(dataserver_train.class_labels))
    class_correct = [0] * len(dataserver_train.class_labels)
    class_total = [0] * len(dataserver_train.class_labels)
    net = net.to(device).float()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, bounding_box, labels = data['image'], data['bbox'], data['label']
            labels = labels.tolist()
            if dls.debug_test and i % 50 == 0:
                print("\n\n[i=%d:] Ground Truth:     " %i + ' '.join('%10s' % 
dataserver_train.class_labels[labels[j]] for j in range(dls.batch_size)))
            images = images.to(device).float()
            outputs = net(images)
            outputs_label = outputs[0]
            outputs_regression = outputs[1]
            outputs_regression[outputs_regression < 0] = 0
            outputs_regression[outputs_regression > 31] = 31
            outputs_regression[torch.isnan(outputs_regression)] = 0
            output_bb = outputs_regression.tolist()
            _, predicted = torch.max(outputs_label.data, 1)
            predicted = predicted.tolist()
            if dls.debug_test and i % 50 == 0:
                print("[i=%d:] Predicted Labels: " %i + ' '.join('%10s' % 
dataserver_train.class_labels[predicted[j]] for j in range(dls.batch_size)))
                for idx in range(dls.batch_size):
                    i1 = int(bounding_box[idx][1])
                    i2 = int(bounding_box[idx][3])
                    j1 = int(bounding_box[idx][0])
                    j2 = int(bounding_box[idx][2])
                    k1 = int(output_bb[idx][1])
                    k2 = int(output_bb[idx][3])
                    l1 = int(output_bb[idx][0])
                    l2 = int(output_bb[idx][2])
                    print("                    gt_bb:  [%d,%d,%d,%d]"%(j1,i1,j2,i2))
                    print("                  pred_bb:  [%d,%d,%d,%d]"%(l1,k1,l2,k2))
                    images[idx,0,i1:i2,j1] = 255
                    images[idx,0,i1:i2,j2] = 255
                    images[idx,0,i1,j1:j2] = 255
                    images[idx,0,i2,j1:j2] = 255
                    images[idx,2,k1:k2,l1] = 255                      
                    images[idx,2,k1:k2,l2] = 255
                    images[idx,2,k1,l1:l2] = 255
                    images[idx,2,k2,l1:l2] = 255
                
                #self.dl_studio.display_tensor_as_image(
                #      torchvision.utils.make_grid(images, normalize=True), 
                #      "see terminal for test results at i=%d" % i)
            for label,prediction in zip(labels,predicted):
                confusion_matrix[label][prediction] += 1
            total += len(labels)
            correct +=  [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
            comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
            for j in range(dls.batch_size):
                label = labels[j]
                class_correct[label] += comp[j]
                class_total[label] += 1
    print("\n")
    for j in range(len(dataserver_train.class_labels)):
        print('Prediction accuracy for %5s : %2d %%' % (
      dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
    print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" % 
                                                           (100 * correct / float(total)))
    print("\n\nDisplaying the confusion matrix:\n")
    out_str = "                "
    for j in range(len(dataserver_train.class_labels)):  
                         out_str +=  "%15s" % dataserver_train.class_labels[j]   
    print(out_str + "\n")
    for i,label in enumerate(dataserver_train.class_labels):
        out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                         for j in range(len(dataserver_train.class_labels))]
        out_percents = ["%.2f" % item.item() for item in out_percents]
        out_str = "%12s:  " % dataserver_train.class_labels[i]
        for j in range(len(dataserver_train.class_labels)): 
                                               out_str +=  "%15s" % out_percents[j]
        print(out_str)

def custom_load_PurdueShapes5_dataset(dataserver_train, dataserver_test ):        
    train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                       batch_size=dls.batch_size,shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                       batch_size=dls.batch_size,shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader

#defines the Noise level classification network
class Noise_classify_Net(nn.Module):
    #classification model_V3 (Final Version)
    def __init__(self):
        super(Noise_classify_Net, self).__init__()
        self.conv_seqn = nn.Sequential(
            # Conv Layer block 1:
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
        )
        self.fc_seqn = nn.Sequential(
            nn.Linear(8192, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 4),#0%,20%,50%,80%           
        )
    def forward(self, x):
        x = self.conv_seqn(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc_seqn(x)
        return x

#noise level classification training
def noiselevel_training(net, epochs, train_dataloader_1, train_dataloader_2, train_dataloader_3,train_dataloader_4):        
    net = copy.deepcopy(net)
    print("\n ....... Inside noiselevel_training() ........ \n")
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=dls.learning_rate, momentum=dls.momentum)
    train_iter_1 = itertools.cycle(train_dataloader_1)
    train_iter_2 = itertools.cycle(train_dataloader_2)
    train_iter_3 = itertools.cycle(train_dataloader_3)
    train_iter_4 = itertools.cycle(train_dataloader_4)

    for epoch in range(epochs):  
        running_loss_labeling = 0.0
        for i in range(2500):#2500
            try:
                data1, data2, data3, data4 = next(train_iter_1), next(train_iter_2), next(train_iter_3), next(train_iter_4)
                inputs_1, noise_labels_1 = data1['image'], data1['noise_label']
                inputs_2, noise_labels_2 = data2['image'], data2['noise_label']
                inputs_3, noise_labels_3 = data3['image'], data3['noise_label']
                inputs_4, noise_labels_4 = data4['image'], data4['noise_label']
                #print(inputs_1.shape, noise_labels_1.shape)
                inputs = torch.cat([inputs_1, inputs_2, inputs_3, inputs_4], dim=0)
                noise_labels = torch.cat([noise_labels_1, noise_labels_2, noise_labels_3, noise_labels_4], dim=0)
                #print(inputs.shape, noise_labels.shape)
                #print("\n Before shuffle :", noise_labels)
                inputs_np, noise_np = inputs.numpy(), noise_labels.numpy()
                x, y = shuffle(inputs_np, noise_np, random_state=0)
                inputs, noise_labels = torch.from_numpy(x),torch.from_numpy(y)
                #print("\n After shuffle :", noise_labels)
                inputs = inputs.to(dls.device)
                noise_labels = noise_labels.to(dls.device)
                optimizer.zero_grad()
                outputs = net(inputs).float()
                #print(outputs)
                loss_labeling = criterion(outputs, noise_labels)
                loss_labeling.backward()        
                optimizer.step()
                running_loss_labeling += loss_labeling.item()                   
                if i % 500 == 499:    
                    avg_loss_labeling = running_loss_labeling / float(500)
                    print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f " % (epoch + 1, i + 1, avg_loss_labeling))
                    running_loss_labeling = 0.0
            except:
                print("\n error \n")
                break
    print("\nFinished Training\n")
    noise_path_saved_model_new = noise_path_saved_model + "_" + str(epochs)
    torch.save(net.state_dict(), noise_path_saved_model_new)
    return net,noise_path_saved_model_new

#noise level detection code
def custom_noise_testing(net, test_dataloader, path):
    #net.load_state_dict(torch.load(path))
    correct = 0
    correct_label = [0,0,0,0]
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, noise_labels = data['image'], data['noise_label']
            images = images.to(device)
            outputs = net(images)
            actual = noise_labels.cpu().numpy()
            #print(outputs)
            for i in range(4):
                #print(torch.argmax(outputs[i]), actual)
                predicted_label = torch.argmax(outputs[i]).cpu().numpy()
                #print(predicted_label, torch.argmax(outputs[i]).cpu(), actual[i])
                correct_label[int(predicted_label)]+=1
                if(torch.argmax(outputs[i]).cpu().numpy() == actual[i]):
                    correct+=1
    max_index, max_value = max(enumerate(correct_label), key=operator.itemgetter(1))
    print("\n Testing accuracy = %f "%(correct/1000))
    print("\n the predicted noise label for overall dataset is = %d" %max_index)
    print("\n part of confusion matrix \n")
    print(correct_label)
    return max_index, correct_label 
            

            
def main():

    #detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    #train_dataloader, test_dataloader = custom_load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    #we can increase the depth or can add more skipblocks
    noise_model = Noise_classify_Net()
    dls.show_network_summary(noise_model)
    noise_epochs = 20
    test_dataloader_list = []
    #loading all start
    #"""
    dataserver_train = myPurdueShapes5Dataset(train_or_test = 'train',dl_studio = dls, dataset_file = dataset_train_list[0])
    dataserver_test = myPurdueShapes5Dataset( train_or_test = 'test',dl_studio = dls,dataset_file = dataset_test_list[0])
    train_dataloader_1, test_dataloader_1 = custom_load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    dataserver_train = myPurdueShapes5Dataset(train_or_test = 'train',dl_studio = dls, dataset_file = dataset_train_list[1])
    dataserver_test = myPurdueShapes5Dataset( train_or_test = 'test',dl_studio = dls,dataset_file = dataset_test_list[1])
    train_dataloader_2, test_dataloader_2 = custom_load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    dataserver_train = myPurdueShapes5Dataset(train_or_test = 'train',dl_studio = dls, dataset_file = dataset_train_list[2])
    dataserver_test = myPurdueShapes5Dataset( train_or_test = 'test',dl_studio = dls,dataset_file = dataset_test_list[2])
    train_dataloader_3, test_dataloader_3 = custom_load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    #"""
    dataserver_train = myPurdueShapes5Dataset(train_or_test = 'train',dl_studio = dls, dataset_file = dataset_train_list[3])
    dataserver_test = myPurdueShapes5Dataset( train_or_test = 'test',dl_studio = dls,dataset_file = dataset_test_list[3])
    train_dataloader_4, test_dataloader_4 = custom_load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    #loading all ends
    #total_dataloader = MyLoader([train_dataloader_1,train_dataloader_2, train_dataloader_3, train_dataloader_4])
    noise_model, path = noiselevel_training(noise_model, noise_epochs, train_dataloader_1, train_dataloader_2, train_dataloader_3,train_dataloader_4)
    noise_model = noise_model.to(device)
    print("\n testing on testdataloader_3 \n")
    max_index, conf_3 = custom_noise_testing(noise_model,test_dataloader_3, path)

    print("\n ............ Final Detection starts ......... \n")    
    print("\n testing on dataset with 50% noise level \n")
    if(max_index == 0):
        model1 = detector.LOADnet2(skip_connections=True, depth=8)
        print("\n ... training for depth 8 .... \n")
        model1 = custom_run_code_for_training_with_CrossEntropy_and_MSE_Losses(model1, train_dataloader_1)
        custom_run_code_for_testing_detection_and_localization(model1, test_dataloader_1, dataserver_train)
    elif(max_index == 1):
        model2 = detector.LOADnet2(skip_connections=True, depth=12)
        print("\n ... training for depth 12 .... \n")
        model2 = custom_run_code_for_training_with_CrossEntropy_and_MSE_Losses(model2, train_dataloader_2)
        custom_run_code_for_testing_detection_and_localization(model2, test_dataloader_2, dataserver_train)
    elif(max_index == 2):
        model3 = detector.LOADnet2(skip_connections=True, depth=14)
        print("\n ... training for depth 14 .... \n")
        model3 = custom_run_code_for_training_with_CrossEntropy_and_MSE_Losses(model3, train_dataloader_3)
        custom_run_code_for_testing_detection_and_localization(model3, test_dataloader_3, dataserver_train)
    else:
        model4 = detector.LOADnet2(skip_connections=True, depth=16)
        print("\n ... training for depth 16 .... \n")
        model4 = custom_run_code_for_training_with_CrossEntropy_and_MSE_Losses(model4, train_dataloader_4)
        custom_run_code_for_testing_detection_and_localization(model4, test_dataloader_4, dataserver_train)
    

if __name__== "__main__":
  main()

