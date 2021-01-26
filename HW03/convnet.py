import torch, torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tvt 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1
#data loading
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data_loc = datasets.CIFAR10(root='data\\', train=True, download=True, transform=transform)
test_data_loc = datasets.CIFAR10(root='data\\', train=False, download=True, transform=transform)

train_data_loader = torch.utils.data.DataLoader(train_data_loc, batch_size=4,shuffle=True, num_workers=2)
test_data_loader = torch.utils.data.DataLoader(test_data_loc, batch_size=1, shuffle=False, num_workers=2)

#in order to test
torch.manual_seed(0)
np.random.seed()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TemplateNet(nn.Module):
    def __init__(self):
        super(TemplateNet, self).__init__()
        #Task 1
        #self.conv1 = nn.Conv2d(3, 128, 3) ## (A)
        #self.conv2 = nn.Conv2d(128, 128, 3) ## (B)
        #add padding with first layer only
        self.conv1 = nn.Conv2d(3, 128, 3, padding=(1,1)) ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3)
        #self.conv2 = nn.Conv2d(128, 128, 3, padding=(1,1)) ## (B) just one padding needed each side
        self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(28800, 1000) ## (C)
        #self.fc1 = nn.Linear(4608, 1000) ## (C)#without zero padding
        self.fc1 = nn.Linear(6272, 1000) ## (C)#with zero padding to first conv layers
        #self.fc1 = nn.Linear(8192, 1000) ## (C)#with zero padding to both conv layers
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        #uncomment this for task2
        x = self.pool(F.relu(self.conv2(x))) ## (D)initially was [128,15,15] but after having the convolution with valid mode input will be reduced
        #print(x.size())
        #task 1
        #x = x.view(-1, 28800) ## (E)
        #x = x.view(-1, 4608)#this is for reduced size [128,6,6] but once zeropadding added size [128,8,8]#without zero padding
        x = x.view(-1, 6272)#with zero padding to first conv layer [128,7,7]
        #x = x.view(-1, 8192)#with zero padding to both conv layers [128,8,8]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def generate_confusion_matrix(net):
    #conf_matrix = np.empty(shape=(10,10))
    conf_matrix = np.zeros((10,10))
    for i, data in enumerate(test_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        #print(outputs[0])
        predicted_class =torch.argmax(outputs[0]).cpu().numpy()
        actual_label = labels.cpu().numpy()[0]
        #conf_matrix[int(predicted_class)][int(actual_label)]+=1
        conf_matrix[int(actual_label)][int(predicted_class)]+=1
        print(predicted_class,actual_label, conf_matrix[int(predicted_class)][int(actual_label)])
    print("\n ========= priniting the confusion matrix =============== \n")
    for i in range(10):
        for j in range(10):
            print("\n confusion_matrix[%d][%d] = %d \n"%(i,j,conf_matrix[i][j]))
    print(torch.from_numpy(conf_matrix))

def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" %(epoch + 1, i + 1, running_loss / float(2000)))
                running_loss = 0.0
    return net

def main():
    net1 = TemplateNet()
    net2 = run_code_for_training(net1)
    generate_confusion_matrix(net2)
    
	

if __name__ == "__main__":
    main()