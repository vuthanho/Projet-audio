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

#get the workspace path
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

#DL test set
test_bruit_path = cwd+'/data/data_test_bruit'
test_path = cwd+'/data/data_test'
testset =  SpeechDataset(test_bruit_path, test_path, transform=['reshape','cut&sousech','normalisation','test','tensor_cuda'])

# test set loader
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

#load model
model_load = FCN()
model_load.load_state_dict(torch.load(cwd+"\\saved\\b25_5000"))
model_load.double().cuda()
model_load.eval()

# A PARTIR DE LA FAUT METTRE EN ORDRE

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