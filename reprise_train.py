# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 08:16:52 2020

@author: Loïc
"""

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
from scipy.io import wavfile #for audio processing
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





#Variable a set

batch_size=5
n_iterations =100
display_spectro=False
display_psnr=True


#get the workspace path
dir_path = os.path.dirname(os.path.realpath(__file__))
cwd = os.getcwd()

#Affichage de ce qu'il contient
#for param_tensor in model_load.state_dict():
#    print(param_tensor, "\t", model_load.state_dict()[param_tensor].size())

#load model
model_load = FCN()
model_load.load_state_dict(torch.load(cwd+"\\saved\\model_b5_7000_ASVG"))
model_load.double().cuda()
model_load.eval()


torch.backends.cudnn.enabled = True
criterion = torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.ASGD(model_load.parameters())
optimizer.load_state_dict(torch.load(cwd+"\\saved\\optimizer_b5_7000_ASVG"))


# A PARTIR DE LA FAUT METTRE EN ORDRE

train_bruit_path = cwd+'/data/data_train_bruit'
train_path = cwd+'/data/data_train'

trainset = SpeechDataset(train_bruit_path, train_path, transform=['reshape','cut&sousech','normalisation','train','tensor_cuda'])
trainloader=torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

#data set as iterator
dataiter=iter(trainloader)

loss_vector = np.zeros(n_iterations)
psnr_debruite_vector = np.zeros(n_iterations)
psnr_bruite_vector = np.zeros(n_iterations)

myplot = plt
myplot.figure(frameon=True)
myplot.ion()


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
        y_pred_temp = model_load(x_temp)
        
        # Compute and print loss
        loss = criterion(torch.squeeze(y_pred_temp), torch.squeeze(y_temp))
        loss_vector[epoch]=loss.item()
    
        #backward → calcul les grads
        loss.backward()
            
        #optimise → applique les grad trouvées au différent params (update weights)
        optimizer.step()
        
        #save le resultat
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

    module_s=y_pred[0][0].cpu().detach().numpy()
    module_x=x[0][0].cpu().detach().numpy()
    module_z=y[0][0].cpu().detach().numpy()
    psnr_debruite_db=psnr(module_s,module_z[:,0:116])
    psnr_bruite_db=psnr(module_x,module_z)
    psnr_debruite_vector[epoch]=psnr_debruite_db
    psnr_bruite_vector[epoch]=psnr_bruite_db
    
    myplot.subplot(211)
    myplot.cla()
    myplot.plot(psnr_debruite_vector,label="Débruité") 
    myplot.plot(psnr_bruite_vector,label="bruité")
    myplot.legend()
    myplot.xlim(0,epoch+0)
    myplot.xlabel("epoch")
    myplot.ylabel("PSNR dB")
    myplot.subplot(212)
    myplot.cla()
    myplot.plot(loss_vector)
    myplot.xlim(0,epoch+0)
    if epoch>50:
        myplot.ylim(0,4.0)
    myplot.xlabel("epoch")
    myplot.ylabel("MSE")
    myplot.pause(0.01)


        
    #print(loss_vector[epoch])
    dataiter=iter(trainloader)
    
#save model & optimizer : https://pytorch.org/tutorials/beginner/saving_loading_models.html
#torch.save(model_load.state_dict(), cwd+"\\saved\\model_b5_7000_ASVG")
#torch.save(optimizer.state_dict(), cwd+"\\saved\\optimizer_b5_7000")