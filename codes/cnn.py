# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:40:11 2019

@author: Loïc
"""
import torch
import torch.nn as nn



class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        #First layer : convolution 
        self.first_conv = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        #second layer : convolution with 5x5 kernel, stride 1, 50 output channels and no zeros padding
        self.second_conv = torch.nn.Conv1d(20, 50, 5, stride = 1)
        #third layer : full connected layer with 500 neurons
        self.first_fully_c = torch.nn.Linear(50,500)
        #forth layer : fully connected layer with 10 neurons
        self.second_fully_c = torch.nn.Linear(500, 10)
        
    #operations made through the different layers of the neural network
    def forward(self, I):
        #activation of the first layer
        I = torch.nn.functional.relu(self.first_conv(I))
        #pooling of first layer
        I = torch.nn.functional.max_pool1d(I, 5, stride=2)
        #activation of the second layer
        I = torch.nn.functional.relu(self.second_conv(I))
        #pooling of the second layer
        I = torch.nn.functional.max_pool2d(I, 5, stride=2)
        #activation of the third layer
        I = torch.nn.functional.relu(self.first_fully_c(I))
        #activation of forth layer
        I = torch.nn.functional.log_softmax(self.second_fully_c(I), dim=1)
        return(I)
    
cnn = CNN()