# -*- coding: utf-8 -*-

"""
Created on Tue Nov 26 16:22:59 2019
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
@author: Olivier
@author: Loïc

paaser en db
normaliser fréquence par fréquence (ligne) les psectro -moy puis / o²
"""
from codes.speechdataset import SpeechDataset
from codes.twoLayerNet import FCN

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import math

#Batch size (permet de travailler avec plusieurs sample en même temps )
"""
attention ! il faut vérifier que sa donne un résultat entier nb de fichier 
divisé par batch_size (enfin je pense)
"""
<<<<<<< HEAD
batch_size= 25
=======
batch_size=20

>>>>>>> master
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
testset =  SpeechDataset(test_bruit_path, test_path, transform=['reshape','cut&sousech','normalisation','test','tensor_cuda'])

#training set loader
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

#data set as iterator
dataiter=iter(trainloader)

#nombre de batches totales présent dans notre sytèmes
n_batches=len(dataiter)

# Construct our model by instantiating the class defined above
model = FCN()
def init_normal(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.dirac_(m.weight, std=0.01)
model.apply(init_normal)
model.double().cuda()

#learning rate
learning_rate = 1e-3

#nb d'iter → nombre epoch
<<<<<<< HEAD
n_iterations = 5000

=======
n_iterations = 1000
>>>>>>> master
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
torch.backends.cudnn.enabled = True
#decente par gradient, avoir si on prend autre chose
# optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
optimizer = torch.optim.Adadelta(model.parameters(),lr=learning_rate)
loss_vector = np.zeros(n_iterations)

# for epoch in range(n_iterations):
#     print(epoch)

for epoch in range(n_iterations):
    
    print(epoch,flush=True)
    sys.stdout.flush()
    data_train = dataiter.next()
    x,y=data_train
    x=x.to(torch.device("cuda:0"))
    y=y.to(torch.device("cuda:0"))
        
    # Forward pass: Compute predicted y by passing x to the model
    for subpart in range (124-8):

        #init grad
        optimizer.zero_grad()

        x_temp=x[:,:, :,subpart:subpart+8]
        y_temp=y[:,:, :,subpart:subpart+1]
        y_pred_temp = model(x_temp)
        
        # Compute and print loss
        loss = criterion(torch.squeeze(y_pred_temp), torch.squeeze(y_temp))
        loss_vector[epoch]=loss.item()
    
        #backward → calcul les grads
        loss.backward()
            
        #optimise → applique les grad trouvées au différent params (update weights)
        optimizer.step()
        
        #save le resultat
        if epoch==n_iterations-1:
            if subpart == 0:
                y_pred=y_pred_temp
            else:
                y_pred=torch.cat((y_pred,y_pred_temp), 3)
                
    print(loss_vector[epoch],flush=True)
    sys.stdout.flush()
    if epoch==n_iterations-1:
        for b in range(batch_size):
            module_s=y_pred[b][0].cpu().detach().numpy()
            module_x=x[b][0].cpu().detach().numpy()
            module_z=y[b][0].cpu().detach().numpy()
            plt.figure()
            plt.subplot(131)
            plt.imshow(module_s) 
            plt.subplot(132)
            plt.imshow(module_x) 
            plt.subplot(133)
            plt.imshow(module_z) 
            plt.show()
            rsb = RSB(module_s,module_x,module_z)
            print(rsb)
            
        plt.figure()
        plt.plot(loss_vector)
        #plt.show()


        
    #print(loss_vector[epoch])
    dataiter=iter(trainloader)
    
#save model & optimizer : https://pytorch.org/tutorials/beginner/saving_loading_models.html
torch.save(model.state_dict(), "C:/Users/Loïc/Documents/3A/deep learning/model_nom")
torch.save(optimizer.state_dict(), "C:/Users/Loïc/Documents/3A/deep learning/optimizer_b25_5000")
#load
#model_load = FCN()
#model_load.load_state_dict(torch.load("C:/Users/Loïc/Documents/3A/deep learning/model_nom"))
#model_load.eval()

# data_test_iter=iter(testloader)
# data_test = data_test_iter.next()
# x,y,a=data_test
# x=x.to(torch.device("cuda:0"))
# y=y.to(torch.device("cuda:0"))
# a=a.to(torch.device("cuda:0"))
# y_pred = model(x)


# from scipy.io import wavfile #for audio processing
# import scipy.signal as sig
# from math import floor

def signal_reconsctructed(y_pred,a,indice):
     fs=8000
     nperseg = 256
     noverlap=nperseg//2
     module_s=y_pred[indice][0].cpu().detach().numpy()
     phase_s=a[indice][0].cpu().detach().numpy()
     Zxx = module_s*(np.exp(1j*phase_s))
     _,reconstructed = sig.istft(Zxx, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
     reconstructed = np.int16(reconstructed/np.amax(np.absolute(reconstructed))*2**15)
     wavfile.write('signal_denoise_'+str(indice)+'.wav',fs,reconstructed)

# signal_reconsctructed(y_pred,a,0)
    
def RSB(prediction,bruit,reference):
    reference=reference[:,0:116]
#    s_min = prediction.min()
#    s_max = prediction.max()
#    prediction = np.divide(prediction-s_min,s_max-s_min)
    
    bruit=bruit[:,0:116]
    s_minb = bruit.min()
    s_maxb = bruit.max()
    bruit = np.divide(bruit-s_minb,s_maxb-s_minb)
    
    bruit=np.abs(bruit-reference)#puissance du bruit
    bruit=np.sum(bruit,0)/129#moyenne de la puissance du bruit à chaque instant
    
    prediction=np.sum(np.abs(prediction),0)/129
    
    rsb=(np.sum(prediction)/116)/(np.sum(bruit)/116)#moyenne de la puissance du 
    rsb = 10 * math.log10(rsb)
    
    return rsb


#load
model_load = FCN()
model_load.load_state_dict(torch.load("C:/Users/Loïc/Documents/3A/deep learning/b25_5000"))
model_load.double().cuda()
model_load.eval()

batch_size=1
test_bruit_path = train_bruit_path
test_path = train_path
testset =  SpeechDataset(test_bruit_path, test_path, transform=['reshape','cut&sousech','normalisation','test','tensor_cuda'])
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
data_test_iter=iter(testloader)
data_test = data_test_iter.next()

loss_test_vector = np.zeros(n_iterations)
nb=1
for epoch in range(nb):
    
    print(epoch,flush=True)
    sys.stdout.flush()
    data_test = data_test_iter.next()
    x_test,y_test,a_test=data_test
    x_test=x_test.to(torch.device("cuda:0"))
    y_test=y_test.to(torch.device("cuda:0"))
    a_test=a_test.to(torch.device("cuda:0"))
    
    # Forward pass: Compute predicted y by passing x to the model
    for subpart in range (124-8):

        x_temp=x_test[:,:, :,subpart:subpart+8]
        y_temp=y_test[:,:, :,subpart:subpart+1]
        y_pred_temp = model_load(x_temp)
        
        # Compute and print loss
        loss_test = criterion(torch.squeeze(y_pred_temp), torch.squeeze(y_temp))
        loss_test_vector[epoch]=loss_test.item()

        
        #save le resultat
        if subpart == 0:
            y_pred=y_pred_temp
        else:
            y_pred=torch.cat((y_pred,y_pred_temp), 3)
                
    print(loss_test_vector[epoch],flush=True)
    sys.stdout.flush()
    if epoch==nb-1:
        for b in range(batch_size):
            module_s_test=y_pred[b][0].cpu().detach().numpy()
            module_x_test=x_test[b][0].cpu().detach().numpy()
            module_z_test=y_test[b][0].cpu().detach().numpy()
            plt.figure()
            plt.subplot(131)
            plt.imshow(module_s_test) 
            plt.subplot(132)
            plt.imshow(module_x_test) 
            plt.subplot(133)
            plt.imshow(module_z_test) 
            plt.show()
            
        plt.figure()
        plt.plot(loss_test_vector)
        #plt.show()


        
    #print(loss_vector[epoch])
    data_test_iter=iter(testloader)
    
    
for param_tensor in model_load.state_dict():
    print(param_tensor, "\t", model_load.state_dict()[param_tensor].size())