#!/usr/bin/env python

"""
For task1, to make the network learn about the sentiments, I tried
(a)simply passing the word index thro' review_to_index_tensor()
(b)passing the word index to a word2vec network
(c)passing one-hot repr of word index to word2vec network

By default script uses one-hot representation of the word indices
in the sorted vocabulary. Which predicts everything as positive sentiments.

Trained on "sentiment_dataset_train_40.tar.gz" dataset
for 2 epochs.

while (a) passing the word index improves the classification accuracy.
while (b) and (c) employ two encoding on the input word. I found 
(a) provides the best classification accuracy during testing. 
Although loss fairly oscillates around 0.68-0.71
"""

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
import time


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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_iter_log = 100

class TextClassification(nn.Module):             
    """
    The purpose of this inner class is to be able to use the DLStudio module for simple 
    experiments in text classification.  Consider, for example, the problem of automatic 
    classification of variable-length user feedback: you want to create a neural network
    that can label an uploaded product review of arbitrary length as positive or negative.  
    One way to solve this problem is with a recurrent neural network in which you use a 
    hidden state for characterizing a variable-length product review with a fixed-length 
    state vector.  This inner class allows you to carry out such experiments.
    """
    def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        super(TextClassification, self).__init__()
        dls = dl_studio
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    class SentimentAnalysisDataset(torch.utils.data.Dataset):
        def __init__(self, dl_studio, train_or_test, dataset_file):
            super(TextClassification.SentimentAnalysisDataset, self).__init__()
            self.train_or_test = train_or_test
            root_dir = dl_studio.dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if train_or_test is 'train':
                if sys.version_info[0] == 3:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                self.categories = sorted(list(self.positive_reviews_train.keys()))
                self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                self.indexed_dataset_train = []
                for category in self.positive_reviews_train:
                    for review in self.positive_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 1])
                for category in self.negative_reviews_train:
                    for review in self.negative_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 0])
                random.shuffle(self.indexed_dataset_train)
                print("\n train_datasaver length = %d\n"%len(self.indexed_dataset_train))
            elif train_or_test is 'test':
                if sys.version_info[0] == 3:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                self.vocab = sorted(self.vocab)
                self.categories = sorted(list(self.positive_reviews_test.keys()))
                self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                self.indexed_dataset_test = []
                for category in self.positive_reviews_test:
                    for review in self.positive_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 1])
                for category in self.negative_reviews_test:
                    for review in self.negative_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 0])
                random.shuffle(self.indexed_dataset_test)
                print("\n test_datasaver length = %d\n"%len(self.indexed_dataset_test))

        def get_vocab_size(self):
            return len(self.vocab)

        def one_hotvec_for_word(self, word):
            word_index =  self.vocab.index(word)
            hotvec = torch.zeros(1, len(self.vocab))
            hotvec[0, word_index] = 1
            return hotvec
        
        def review_to_index_tensor(self,review):
            review_tensor = torch.zeros(len(review), 1)
            for i,word in enumerate(review):
                review_tensor[i,:] = self.vocab.index(word)
            return review_tensor

        def review_to_tensor(self, review):
            review_tensor = torch.zeros(len(review), len(self.vocab))
            for i,word in enumerate(review):
                review_tensor[i,:] = self.one_hotvec_for_word(word)
            return review_tensor

        def sentiment_to_tensor(self, sentiment):
            """
            Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
            sentiment and 1 for positive sentiment.  We need to pack this value in a
            two-element tensor.
            """        
            sentiment_tensor = torch.zeros(2)
            if sentiment is 1:
                sentiment_tensor[1] = 1
            elif sentiment is 0: 
                sentiment_tensor[0] = 1
            sentiment_tensor = sentiment_tensor.type(torch.long)
            return sentiment_tensor

        def __len__(self):
            if self.train_or_test is 'train':
                return len(self.indexed_dataset_train)
            elif self.train_or_test is 'test':
                return len(self.indexed_dataset_test)

        def __getitem__(self, idx):
            #print("\n Inside getitem() length is = %d \n"%len(self.indexed_dataset_train))
            if(self.train_or_test=='train'):
                sample = self.indexed_dataset_train[idx] 
            if(self.train_or_test=='test'):
                sample = self.indexed_dataset_test[idx] 
            review = sample[0]
            #print("\n review = " + str(review))
            review_category = sample[1]
            #print("\n review_category = " + str(review_category))
            review_sentiment = sample[2]
            #print("\n 1.review_sentiment = " + str(review_sentiment))
            review_sentiment = self.sentiment_to_tensor(review_sentiment)
            #print("\n 2.review_sentiment = " + str(review_sentiment))
            #review_tensor = self.review_to_tensor(review)
            review_tensor = self.review_to_index_tensor(review)
            #print("\n review tensor =" +str(review_tensor))
            #print(review_tensor.size(), len(review), len(self.vocab))
            #print("\n ============== \n")
            category_index = self.categories.index(review_category)
            sample = {'review'       : review_tensor, 
                      'category'     : category_index, # should be converted to tensor, but not yet used
                      'sentiment'    : review_sentiment }
            return sample

    def load_SentimentAnalysisDataset(self, dataserver_train, dataserver_test ):   
        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                    batch_size=dls.batch_size,shuffle=True, num_workers=1)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                           batch_size=dls.batch_size,shuffle=False, num_workers=1)

    class GRUnet(nn.Module):
        """
        Source: https://blog.floydhub.com/gru-with-pytorch/
        with the only modification that the final output of forward() is now
        routed through LogSoftmax activation. 
        """
        def __init__(self, input_size, hidden_size, output_size, n_layers, drop_prob=0.2):
            super(TextClassification.GRUnet, self).__init__()
            self.hidden_size = hidden_size
            self.n_layers = n_layers
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=drop_prob)
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.logsoftmax = nn.LogSoftmax(dim=1)
            
        def forward(self, x, h):
            out, h = self.gru(x, h)
            out = self.fc(self.relu(out[:,-1]))
            out = self.logsoftmax(out)
            return out, h

        def init_hidden(self, batch_size):
            weight = next(self.parameters()).data
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
            return hidden

    def save_model(self, model):
        "Save the trained model to a disk file"
        torch.save(model.state_dict(), dls.path_saved_model)

    def run_code_for_training_for_text_classification_with_gru(self, net, hidden_size): 
        filename_for_out = "performance_numbers_" + str(dls.epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(device)
        ##  Note that the GREnet now produces the LogSoftmax output:
        criterion = nn.NLLLoss()
#            criterion = nn.MSELoss()
#            criterion = nn.CrossEntropyLoss()
        accum_times = []
        optimizer = optim.SGD(net.parameters(), 
                     lr=dls.learning_rate, momentum=dls.momentum)
        for epoch in range(dls.epochs):  
            print("")
            running_loss = 0.0
            start_time = time.clock()
            for i, data in enumerate(self.train_dataloader):    
                review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                #print("\n inside training = " + str(review_tensor.size()))
                review_tensor = review_tensor.to(device)
                sentiment = sentiment.to(device)
                ## The following type conversion needed for MSELoss:
                ##sentiment = sentiment.float()
                optimizer.zero_grad()
                hidden = net.init_hidden(1).to(device)
                for k in range(review_tensor.shape[1]):
                    output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                ## If using NLLLoss, CrossEntropyLoss
                loss = criterion(output, torch.argmax(sentiment, 1))
                ## If using MSELoss:
                ## loss = criterion(output, sentiment)     
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if i % num_iter_log == num_iter_log-1:    
                    avg_loss = running_loss / float(num_iter_log)
                    current_time = time.clock()
                    time_elapsed = current_time-start_time
                    print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.3f" % (epoch+1,i+1, time_elapsed,avg_loss))
                    accum_times.append(current_time-start_time)
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
        print("Total Training Time: {}".format(str(sum(accum_times))))
        print("\nFinished Training\n")
        self.save_model(net)


    def run_code_for_testing_text_classification_with_gru(self, net, hidden_size):
        net.load_state_dict(torch.load(dls.path_saved_model))
        classification_accuracy = 0.0
        negative_total = 0
        positive_total = 0
        confusion_matrix = torch.zeros(2,2)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                hidden = net.init_hidden(1)
                for k in range(review_tensor.shape[1]):
                    output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                predicted_idx = torch.argmax(output).item()
                gt_idx = torch.argmax(sentiment).item()
                if i % 100 == 99:
                    print("   [i=%d]    predicted_label=%d       gt_label=%d\n\n" % (i+1, predicted_idx,gt_idx))
                if predicted_idx == gt_idx:
                    classification_accuracy += 1
                if gt_idx is 0: 
                    negative_total += 1
                elif gt_idx is 1:
                    positive_total += 1
                confusion_matrix[gt_idx,predicted_idx] += 1
        out_percent = np.zeros((2,2), dtype='float')
        print("\n\nNumber of positive reviews tested: %d" % positive_total)
        print("\n\nNumber of negative reviews tested: %d" % negative_total)
        print("\n\nDisplaying the confusion matrix:\n")
        out_str = "                      "
        out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
        print(out_str + "\n")
        for i,label in enumerate(['true negative', 'true positive']):
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            out_str = "%12s:  " % label
            for j in range(2):
                out_str +=  "%18s" % out_percent[i,j]
            print(out_str)






dls = DLStudio(
#                  dataroot = "/home/kak/TextDatasets/",
                  dataroot = "H:\\DLStudio-1.1.3\\Examples\\data\\",
                  path_saved_model = "C:\\temp\\RNN_task1",
                  momentum = 0.9,
#                  learning_rate =  0.004,
                  learning_rate =  1e-4,
                  epochs = 2,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  debug_train = 1,
                  debug_test = 1,
                  use_gpu = True,
              )


text_cl = TextClassification( dl_studio = dls )
dataserver_train = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'train',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_train_3.tar.gz",
#                                dataset_file = "sentiment_dataset_train_200.tar.gz", 
                                 dataset_file = "sentiment_dataset_train_40.tar.gz", 
                                                                      )
dataserver_test = TextClassification.SentimentAnalysisDataset(
                                 train_or_test = 'test',
                                 dl_studio = dls,
#                                dataset_file = "sentiment_dataset_test_3.tar.gz",
#                                dataset_file = "sentiment_dataset_test_200.tar.gz",
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
                                                                  )
text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()
hidden_size = 512
output_size = 2                            # for positive and negative sentiments
n_layers = 2

#model = text_cl.GRUnet(vocab_size, hidden_size, output_size, n_layers)
model = text_cl.GRUnet(1, hidden_size, output_size,n_layers)#passing the index of words only

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

def main():
    ## TRAINING:
    print("\nStarting training --- BE VERY PATIENT, PLEASE!  The first report will be at 100th iteration. May take around 5 minutes.\n")
    text_cl.run_code_for_training_for_text_classification_with_gru(model, hidden_size)

    ## TESTING:
    text_cl.run_code_for_testing_text_classification_with_gru(model, hidden_size)


if __name__ == "__main__":
	main()


