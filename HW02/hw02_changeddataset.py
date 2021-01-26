import torch, torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as tvt
import matplotlib.pyplot as plt

transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Downloading/Louding CIFAR10 data
trainset  = CIFAR10(root="data//", train=True , download=True, transform = transform)
testset   = CIFAR10(root="data//", train=False, download=True, transform = transform)
classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}


# Separating trainset/testset data/label
#"""
"""
#CPU version
x_train  = trainset.data
x_test   = testset.data
y_train  = trainset.targets
y_test   = testset.targets
"""
#GPU version
x_train  = trainset.train_data
x_test   = testset.test_data
y_train  = trainset.train_labels
y_test   = testset.test_labels

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fs = 30
num_classes = 2 #only cat & dog

# Define a function to separate CIFAR classes by class index
def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]
    
    return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc = transform):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        return bin_index, index_wrt_class


def train_dnn(train_data, test_data):
	# D_out = 2 o/p classes
	D_in, H1, H2, D_out = 3 * 32 * 32, 1000, 256, 2
	print("\n current device type: " + str(device))
	# Randomly initialize weights
	w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
	w2 = torch.randn(H1, H2, device=device, dtype=dtype)
	w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
	learning_rate = 1e-12
	xlist, loss_list = [], []
	num_epochs = 1000
	epoch_log_step = 1

	for epoch in range(num_epochs):
		for i, data in enumerate(train_data):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			x = inputs
			x = inputs.view(x.size(0), -1)  
			y = labels.view(-1,1)
			h1 = x.mm(w1)  # In numpy, you would say h1 = x.dot(w1)
			h1_relu = h1.clamp(min=0)
			h2 = h1_relu.mm(w2)
			h2_relu = h2.clamp(min=0)
			y_pred = h2_relu.mm(w3)
			

		#y = y.view(-1,1)# Compute and print loss
		print(x.size(), h1.size(), h1_relu.size(), h2.size(), h2_relu.size(), y_pred.size(), y.size(), labels.size())
		loss = (y_pred.float() - y.float()).pow(2).sum().item()
		if epoch % epoch_log_step == 0:
			print("Epoch %d: %f"%(epoch, loss))
			loss_list.append(loss)
			xlist.append(epoch)

		# Backpropagate the error for the next epoch
		y_error = y_pred.float() - y.float()
		grad_w3 = h2_relu.t().mm(2 * y_error)  #Gradient of Loss w.r.t w3
		h2_error = 2.0 * y_error.mm(w3.t())  # backpropagated error to the h2 hidden layer
		h2_error[h2 < 0] = 0  # To backprop error, zero those elements where fwd prop values to the same layer are negative
		grad_w2 = h1_relu.t().mm(2 * h2_error)  #Gradient of Loss w.r.t w2
		h1_error = 2.0 * h2_error.mm(w2.t())  # backpropagated error to the h1 hidden layer
		h1_error[h1 < 0] = 0  # We set those elements of the backpropagated error
		grad_w1 = x.t().mm(2 * h1_error)  #Gradient of Loss w.r.t w2
		# Update weights using gradient descent
		w1 -= learning_rate * grad_w1
		w2 -= learning_rate * grad_w2
		w3 -= learning_rate * grad_w3

	# print("Training Summary every %s epochs upto %s epochs" % str(epoch_step_plot), str(num_epochs))
	print("Loss: %s" % str(loss_list))
	test_dnn(w1, w2, w3, test_data)
	plot_loss(xlist, loss_list)

def plot_loss(xlist, ylist):
	plt.plot(xlist,ylist)
	plt.xlabel('Training Iterations', fontsize = fs)
	plt.ylabel('Loss', fontsize = fs)
	#plt.legend(loc='best',fontsize = fs)
	plt.xticks(fontsize = fs)
	plt.yticks(fontsize = fs)
	plt.show()

def test_dnn(w1, w2, w3, test_data):
	acc_preds = 0
	for i, data in enumerate(test_data):
		inputs, labels = data
		# print(torch.Size(labels))
		inputs = inputs.to(device)
		labels = labels.to(device)
		x = inputs.view(inputs.size(0), -1)  # -1 means that that dimension is inferred by the other specified dimensions
		# x = inputs
		y = labels
		h1 = x.mm(w1)  # In numpy, you would say h1 = x.dot(w1)
		h1_relu = h1.clamp(min=0)
		h2 = h1_relu.mm(w2)
		h2_relu = h2.clamp(min=0)
		y_pred = h2_relu.mm(w3)

		# Discretize the 2 classes using argmax
		y_pred_argmax = torch.argmax(y_pred)
		
		#for 0-->cat, 1-->dog
		if y_pred_argmax.cpu().numpy() == y.cpu().numpy()[0]:
			acc_preds = acc_preds + 1
		
		"""
		#for one-hot encoding
		y_loc = np.where(y.numpy() == 1)[0]
		#print(y_pred, y, y_loc, y_pred_argmax.numpy())
		if y_pred_argmax.numpy() == y_loc[0]:
			acc_preds = acc_preds + 1
		"""
	test_acc = acc_preds/(i+1)
	print(i, acc_preds, test_acc)
	print("Test Accuracy:", test_acc)


# ================== Usage ================== #
def main():
	# Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
	cat_dog_trainset = DatasetMaker([get_class_i(x_train, y_train, classDict['cat']), get_class_i(x_train, y_train, classDict['dog'])],
        transform)
	cat_dog_testset  = DatasetMaker([get_class_i(x_test , y_test , classDict['cat']), get_class_i(x_test , y_test , classDict['dog'])],
        transform)
	
	kwargs = {'num_workers': 2, 'pin_memory': False}
	# Create datasetLoaders from trainset and testset
	trainsetLoader   = DataLoader(cat_dog_trainset, batch_size=5, shuffle=True , **kwargs)
	testsetLoader    = DataLoader(cat_dog_testset , batch_size=5, shuffle=False, **kwargs)
	print(len(trainsetLoader),len(testsetLoader))
	train_dnn(trainsetLoader, testsetLoader)

if __name__== "__main__":
  main()