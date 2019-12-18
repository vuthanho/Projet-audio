# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:22:59 2019
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@author: Loïc
"""
from codes.speechdataset import SpeechDataset
from codes.twoLayerNet import FCN

import os
import matplotlib.pyplot as plt
import numpy as np
import torch



#Batch size (permet de travailler avec plusieurs sample en même temps )
"""
attention ! il faut vérifier que sa donne un résultat entier nb de fichier 
divisé par batch_size (enfin je pense)
"""
batch_size=40

#get the workspace path
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()
#DL training set
train_bruit_path = cwd+'/data/data_train_bruit'
train_path = cwd+'/data/data_train'
trainset = SpeechDataset(train_bruit_path, train_path, transform=['reshape','cut&sousech','normalisation','train','tensor_cuda'])

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

#data set as iterator
dataiter=iter(trainloader)

#nombre de batches totales présent dans notre sytèmes
n_batches=len(dataiter)

# Construct our model by instantiating the class defined above
model = FCN()
model.double().cuda()

#learning rate
learning_rate = 0.001


#nb d'iter → nombre epoch
n_iterations = 2
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.L1Loss()
torch.backends.cudnn.enabled = True
#decente par gradient, avoir si on prend autre chose
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
loss_vector = np.zeros(n_iterations)


for epoch in range(n_iterations):
    
    print(epoch)
    data_train = dataiter.next()
    x,y=data_train
    dataiter=iter(trainloader)
    x=x.to(torch.device("cuda:0"))
    y=y.to(torch.device("cuda:0"))
    
    #Average loss during training
    average_loss = 0.0
    
    #init grad
    optimizer.zero_grad()
        
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(torch.squeeze(y_pred), torch.squeeze(y))
    loss_vector[epoch]=loss.item()
    print(loss_vector[epoch])
    #backward → calcul les grads
    loss.backward()
        
    #optimise → applique les grad trouvées au différent params (update weights)
    optimizer.step()
        
    #maj loss → pas sur de la syntaxe
    average_loss += loss.data



data_test_iter=iter(testloader)
data_test = data_test_iter.next()
x,y,a=data_test
x=x.to(torch.device("cuda:0"))
y=y.to(torch.device("cuda:0"))
a=a.to(torch.device("cuda:0"))
y_pred = model(x)


from scipy.io import wavfile #for audio processing
import scipy.signal as sig
from math import floor

def signal_reconsctructed(y_pred,a,indice):
    fs=16000
    nperseg = floor(0.03*fs)
    noverlap=nperseg//2
    module_s=y_pred[indice][0].cpu().detach().numpy()
    phase_s=a[indice][0].cpu().detach().numpy()
    Zxx = module_s*(np.exp(1j*phase_s))
    _,reconstructed = sig.istft(Zxx, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
    reconstructed = np.int16(reconstructed/np.amax(np.absolute(reconstructed))*2**15)
    wavfile.write('signal_denoise_'+str(indice)+'.wav',fs,reconstructed)

#signal_reconsctructed(y_pred,a,0)