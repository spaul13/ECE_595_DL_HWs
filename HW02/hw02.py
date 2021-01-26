import torch, torchvision
import torchvision.datasets as datasets
import torchvision.transforms as tvt 
import matplotlib.pyplot as plt
import numpy as np

dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fs = 30
num_classes = 2 #only cat & dog

def train_dnn(train_data, test_data):
	# D_out = 2 o/p classes
	D_in, H1, H2, D_out = 3 * 32 * 32, 1000, 256, 2
	#print("\n current device type: " + str(device))
	# Randomly initialize weights
	w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
	w2 = torch.randn(H1, H2, device=device, dtype=dtype)
	w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
	learning_rate = 1e-12
	xlist, loss_list = [], []
	num_epochs = 2500
	epoch_log_step = 100

	for epoch in range(num_epochs):
		for i, data in enumerate(train_data):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			x = inputs
			x = inputs.view(x.size(0), -1)  
			y = labels
			h1 = x.mm(w1)  # In numpy, you would say h1 = x.dot(w1)
			h1_relu = h1.clamp(min=0)
			h2 = h1_relu.mm(w2)
			h2_relu = h2.clamp(min=0)
			y_pred = h2_relu.mm(w3)
        
        	

		#print(y_pred.float() - y.float())
		#print(x.size(), h1.size(), h1_relu.size(), h2.size(), h2_relu.size(), y_pred.size(), y.size(), labels.size())
		loss = (y_pred.float() - y.float()).pow(2).sum().item()
		if epoch % epoch_log_step == 99:
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
	#print("Loss: %s" % str(loss_list))
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
		
		#print(y_pred_argmax.cpu().numpy(),y.cpu().numpy()[0])
		if y_pred_argmax.cpu().numpy() == y.cpu().numpy()[0]:
			acc_preds = acc_preds + 1
		
		"""
		#for one-hot encoding
		y_loc = np.where(y.numpy() == 1)[0]
		#print(y_pred, y, y_loc, y_pred_argmax.numpy())
		if y_pred_argmax.numpy() == y_loc[0]:
			acc_preds = acc_preds + 1
		"""
	test_acc = (acc_preds*100)/(i+1)
	#print(i, acc_preds, test_acc)
	print("\n Test Accuracy: %f" %test_acc)

def one_hot_embedding(labels, num_classes):
    #y = torch.eye(num_classes) 
    #return y[labels]
	x = torch.arange(0,num_classes)
	y = torch.nn.functional.one_hot(x)
	return y[labels]

def DataLoading():
	classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	train_data_loc = datasets.CIFAR10(root='data\\', train=True, download=True, transform=transform)
	test_data_loc = datasets.CIFAR10(root='data\\', train=False, download=True, transform=transform)

	trainloader = torch.utils.data.DataLoader(train_data_loc, batch_size=1,shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(test_data_loc, batch_size=1, shuffle=False, num_workers=2)
	# get some random training images
	dataiter = iter(testloader)
	counter = 0
	trainloader_new, testloader_new = [],[]
	while True:
		try:
			images, labels = dataiter.next()
			counter+=1
			#print(counter, labels,classes[labels.numpy()[0]])
			if (classes[labels.numpy()[0]] == 'cat'):
				testloader_new.append([images, torch.tensor([0])])
				#one-hot encoding
				#print('cat ' + str(one_hot_embedding(0, num_classes)))
				#testloader_new.append([images, one_hot_embedding(0, num_classes)])
			if (classes[labels.numpy()[0]] == 'dog'):
				testloader_new.append([images, torch.tensor([1])])
				#one-hot encoding
				#print('dog ' + str(one_hot_embedding(1, num_classes)))
				#testloader_new.append([images, one_hot_embedding(1, num_classes)])
		except:
			break
	#"""
	traindataiter = iter(trainloader)
	while True:
		try:
			images, labels = traindataiter.next()
			counter+=1
			#print(counter, labels,classes[labels.numpy()[0]])
			if (classes[labels.numpy()[0]] == 'cat'):
				trainloader_new.append([images, torch.tensor([0])])
				#trainloader_new.append([images, one_hot_embedding(0, num_classes)])
			if (classes[labels.numpy()[0]] == 'dog'):
				trainloader_new.append([images, torch.tensor([1])])
				#trainloader_new.append([images, one_hot_embedding(1, num_classes)])
		except:
			break
	#train()
	#print(len(trainloader_new),len(testloader_new))
	#print("\n ======== \n training the DNN started \n ========= \n")
	return trainloader_new, testloader_new
	#"""



def main():
	trainloader, testloader = DataLoading()
	train_dnn(trainloader, testloader)
	#DataLoading()

if __name__== "__main__":
  main()
