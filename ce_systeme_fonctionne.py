# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:22:59 2019
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@author: Loïc
"""
from codes.speechdataset import SpeechDataset
from codes.twoLayerNet import TwoLayerNet

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import variable


#Batch size (permet de travailler avec plusieurs sample en même temps )
"""
attention ! il faut vérifier que sa donne un résultat entier nb de fichier 
divisé par batch_size (enfin je pense)
"""
batch_size=10

#get the workspace path
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
#DL training set
train_bruit_path = cwd+'/data/data_train_bruit'
train_path = cwd+'/data/data_train'
trainset = SpeechDataset(train_bruit_path, train_path, transform=['reshape','normalisation','train','tensor_cuda'])
#train_bruit_set = fulltrainset[1]
#training set loader
"""
Le data loader est une fonction qui permet d'importer les données de manière itératifs, 
le but étant de procéder aux calculs sur une nombre d'échantillons restreint (égale au
batch_size).
- shuffle : importer les échantillons de manière aléatoire
- num_workers : nombre de coeur du processeur utilisés 
"""
trainloader=torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

#DL test set
test_bruit_path = cwd+'/data/data_test_bruit'
test_path = cwd+'/data/data_test'
testset =  SpeechDataset(test_bruit_path, test_path, transform=['reshape','normalisation','test','tensor_cuda'])

#training set loader
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

#function to show signal
def sigshow(matrice):
    #a faire construire un vecteur t en fonction de length et fs
    matrice=matrice.numpy()
    for i in range(len(matrice)):
        plt.subplot(len(matrice),1,i+1)
        plt.plot(matrice[i])
        
    plt.show()
    
#data set as iterator
dataiter=iter(trainloader)

#nombre de batches totales présent dans notre sytèmes
n_batches=len(dataiter)

#get next batch
data = dataiter.next()
#matrice contenant les x premiers spectro de ref
reference = data.get("signal")
#matrice contenant les x premiers spectro bruité correspondant
bruit = data.get("signal_noised")

#show signaux référence
#sigshow(reference)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = batch_size, data.get("signal").size(1), 100, data.get("signal").size(1)

# Create random Tensors to hold inputs and outputs
#x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)
model.cuda()

#learning rate
learning_rate = 0.001


#nb d'iter → nombre epoch
n_iterations = 1

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')

#decente par gradient, avoir si on prend autre chose
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
loss_vector = np.zeros(n_batches)
for epoch in range(n_iterations):
    
    print(epoch)
    #Average loss during training
    average_loss = 0.0
    
    #iter chaque batches
    for i, data in enumerate(trainloader, 0):
        y = data.get("signal")
        x = data.get("signal_noised")
#        x=x.to(torch.device("cuda:0"))
#        y=y.to(torch.device("cuda:0"))
        #init grad
        optimizer.zero_grad()
        
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
    
        # Compute and print loss
        loss = criterion(y_pred, y)
        loss_vector[i]=loss.item()
        
        #backward → calcul les grads
        loss.backward()
        
        #optimise → applique les grad trouvées au différent params (update weights)
        optimizer.step()
        
        #maj loss → pas sur de la syntaxe
        average_loss += loss.data

    if epoch % 100 == 99:
            print(epoch, loss.item())
#test=y_pred[0]
#test=test.cpu().detach().numpy()
#ref=y[0]
#ref=ref.cpu().detach().numpy()