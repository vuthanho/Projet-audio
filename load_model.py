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
#from scipy.io import wavfile #for audio processing
import scipy.signal as sig

def signal_reconsctructed(module_s,phase_s,indice):
     fs=8000
     nperseg = 256
     noverlap=nperseg//2
     Zxx = module_s*(np.exp(1j*phase_s))
     _,reconstructed = sig.istft(Zxx, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
     reconstructed = np.int16(reconstructed/np.amax(np.absolute(reconstructed))*2**15)
#     wavfile.write('signal_denoise_'+str(indice)+'.wav',fs,reconstructed)
     return reconstructed

def psnr(img1,img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    result = 10 * math.log10(1. / mse)
    return result

batch_size=1
nb=1680
display_spectro=False
display_psnr=True

#get the workspace path
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

#DL test set
test_bruit_path = cwd+'/data/data_test_bruit'
test_path = cwd+'/data/data_test'

#Affichage de ce qu'il contient
#for param_tensor in model_load.state_dict():
#    print(param_tensor, "\t", model_load.state_dict()[param_tensor].size())

#load model
model_load = FCN()
model_load.load_state_dict(torch.load(cwd+"\\saved\\model_b5_10000_ASVG"))
model_load.double().cuda()
model_load.eval()

# A PARTIR DE LA FAUT METTRE EN ORDRE

train_bruit_path = cwd+'/data/data_train_bruit'
train_path = cwd+'/data/data_train'
#test_bruit_path = train_bruit_path
#test_path = train_path
criterion = torch.nn.MSELoss(reduction='sum')
testset =  SpeechDataset(test_bruit_path, test_path, transform=['reshape','cut&sousech','normalisation','test','tensor_cuda'])
testloader=torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
data_test_iter=iter(testloader)
data_test = data_test_iter.next()


loss_test_vector = np.zeros(nb)
psnr_debruite_vector = np.zeros(nb)
psnr_bruite_vector = np.zeros(nb)

myplot = plt
myplot.figure(frameon=True)
myplot.ion()

for epoch in range(nb):
    
    print("epoch " + str(epoch),flush=True)
    sys.stdout.flush()
    data_test = data_test_iter.next()
    x_test,y_test,a_test=data_test
    x_test=x_test.to(torch.device("cuda:0"))
    y_test=y_test.to(torch.device("cuda:0"))
    a_test=a_test.to(torch.device("cuda:0"))
    loss_mean_vector = np.zeros(124-8)
    # Forward pass: Compute predicted y by passing x to the model
    for subpart in range (124-8):

        x_temp=x_test[:,:, :,subpart:subpart+8]
        y_temp=y_test[:,:, :,subpart:subpart+1]
        y_pred_temp = model_load(x_temp)
        
        # Compute and print loss
        loss_test = criterion(torch.squeeze(y_pred_temp), torch.squeeze(y_temp))
        loss_mean_vector[subpart]=loss_test.item()

        
        #save le resultat
        if subpart == 0:
            y_pred=y_pred_temp
        else:
            y_pred=torch.cat((y_pred,y_pred_temp), 3)
     
    loss_test_vector[epoch]=np.mean(loss_mean_vector)         
    print("Loss moyen = " +str(loss_test_vector[epoch]),flush=True)
    sys.stdout.flush()
    if epoch==nb-1:
        for b in range(batch_size):
            module_s_test=y_pred[b][0].cpu().detach().numpy()
            module_x_test=x_test[b][0].cpu().detach().numpy()
            module_z_test=y_test[b][0].cpu().detach().numpy()
            phase_z_test=a_test[b][0].cpu().detach().numpy()
            phase_z_test=phase_z_test[0:116]
            
            if display_spectro==True:
                plt.figure()
                plt.subplot(131)
                plt.imshow(module_s_test) 
                plt.subplot(132)
                plt.imshow(module_x_test) 
                plt.subplot(133)
                plt.imshow(module_z_test) 
                plt.show()
    module_s_test=y_pred[batch_size-1][0].cpu().detach().numpy()
    module_x_test=x_test[batch_size-1][0].cpu().detach().numpy()
    module_z_test=y_test[batch_size-1][0].cpu().detach().numpy()        
    psnr_debruite_db=psnr(module_s_test,module_z_test[:,0:116])
    psnr_bruite_db=psnr(module_x_test,module_z_test)
    psnr_debruite_vector[epoch]=psnr_debruite_db
    psnr_bruite_vector[epoch]=psnr_bruite_db
    dbmax= psnr_debruite_vector.max()
    dbmin= psnr_bruite_vector.min()
    msemax=loss_test_vector.max()
    msemin=loss_test_vector.min()
    
    if display_psnr==True:
        myplot.subplot(211)
        myplot.cla()
        myplot.plot(psnr_debruite_vector,label="Débruité") 
        myplot.plot(psnr_bruite_vector,label="bruité")
        myplot.legend()
        myplot.xlim(0,epoch+0)
        myplot.ylim(dbmin,dbmax)
        myplot.xlabel("epoch")
        myplot.ylabel("PSNR dB")
        myplot.subplot(212)
        myplot.cla()
        myplot.plot(loss_test_vector)
        myplot.xlim(0,epoch+0)
        myplot.ylim(msemin,msemax)
        myplot.xlabel("epoch")
        myplot.ylabel("MSE")
        myplot.pause(0.01)
        
    data_test_iter=iter(testloader)
    
