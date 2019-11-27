#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
#import GPUtil


# In[2]:


class CNN(torch.nn.Module):
    
    #definition of the convolutions and fully connected layers of the neural network
    def __init__(self):
        super(CNN, self).__init__()
        #First layer : convolution with 5x5 kernel, stride 1, 20 output channels and no zero padding
        self.first_conv = torch.nn.Conv2d(1, 20, 5, stride=1)
        #second layer : convolution with 5x5 kernel, stride 1, 50 output channels and no zeros padding
        self.second_conv = torch.nn.Conv2d(20, 50, 5, stride = 1)
        #third layer : full connected layer with 500 neurons
        self.first_fully_c = torch.nn.Linear(4*4*50,500)
        #forth layer : fully connected layer with 10 neurons
        self.second_fully_c = torch.nn.Linear(500, 10)
    
    #operations made through the different layers of the neural network
    def forward(self, I):
        #activation of the first layer
        I = torch.nn.functional.relu(self.first_conv(I))
        #pooling of first layer
        I = torch.nn.functional.max_pool2d(I, (2,2), stride=2)
        #activation of the second layer
        I = torch.nn.functional.relu(self.second_conv(I))
        #pooling of the second layer
        I = torch.nn.functional.max_pool2d(I, (2,2), stride=2)
        I = I.view(-1, 4*4*50)
        #activation of the third layer
        I = torch.nn.functional.relu(self.first_fully_c(I))
        #activation of forth layer
        I = torch.nn.functional.log_softmax(self.second_fully_c(I), dim=1)
        return(I)
    
cnn = CNN()


# In[3]:


def initialization(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight)
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.uniform_(m.weight)

def training(trainloader, model, device, optimizer, criterion):
    loss_mean = 0
    model.train()
    for batch, (data, labels) in enumerate (trainloader):
        #use of the GPU
        data, labels = data.to(device), labels.to(device)
        #initialization for gradient computation
        optimizer.zero_grad()
        #result of the input through the neural network
        result = model(data)
        #computation of the loss
        loss = criterion(result, labels)
        loss_mean += loss
        #computation of the gradient
        loss.backward()
        #update
        optimizer.step()
        torch.cuda.empty_cache()
    return(loss_mean)


def validation(trainloader, model, device, size_validation, criterion):
    loss_mean = 0
    i = 0
    model.eval()
    for batch, (data, labels) in enumerate (trainloader):
        data, labels = data.to(device), labels.to(device)
        result = model(data)
        loss = criterion(result, labels)
        loss_mean +=loss
        if i > 5000:
            break;
        torch.cuda.empty_cache()
    return(loss_mean)

def test(testloader, model, device, criterion):
    loss_mean = 0
    i=0
    model.eval()
    for batch, (data, labels) in enumerate (testloader):
        i+=1
        data, labels = data.to(device), labels.to(device)
        result = model(data)
        loss = criterion(result, labels)
        loss_mean += loss
        torch.cuda.empty_cache()
    return(loss_mean)
    
    
    
    
    
    


# In[5]:



device = torch.device("cuda")

#parameters to set
bsize = 32#batch size for training set
test_bsize = 16#batch size for test set
learning_rate = 0.01#learning rate
nb_epoch = 2# number of epochs
size_validation = 5000#size of the validation set

#transforming the images into tensor and normalizing them (as shown in the pytorch tutorial)
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))])


#creating the training set
trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize,shuffle=True, num_workers = 1)

#creating the test set
testset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsize,shuffle=True, num_workers = 1)


    
    
    
#creation of the model and use of cuda
model = cnn.to(device)
#model = cnn
model.apply(initialization)
#creation of the optimizer for the SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#definition of the loss function
criterion = torch.nn.NLLLoss()


final_training_loss = []
final_validation_loss = []
for epoch in range (0,60):
    training_loss = training(trainloader, model, device, optimizer, criterion)
    #validation_loss = validation(trainloader, model, device, size_validation, criterion)
    print(training_loss)
    #print(validation_loss)
    print(epoch)
    final_training_loss.append(training_loss)
    #final_validation_loss.append(validation_loss)
test(testloader, model, device, criterion)


# In[37]:


plt.plot(np.arange(2000), training_loss[0:2000])
#plt.ylim([2,2.5])
plt.title('loss for lr = 0.0001')
plt.xlabel('iterations')
plt.ylabel('loss')
training_loss[0]


# In[23]:


torch.cuda.empty_cache()
