# -*- coding: utf-8 -*-
"""
Script contenant des fonctions utiles
"""

import numpy as np
import math
import torch

def psnr(a1,a2):
    """
    si a1 n'est pas un int16, alors a1 doit être compris entre -1 et 1
    """
    mse = np.mean( (a1 - a2) ** 2 )
    if mse == 0:
        return 100
    if type(a1[0])==np.int16:
        max_intensity = float(2**15)
    else:
        max_intensity = 1.0
    return 20 * math.log10(max_intensity / math.sqrt(mse))

def reverse_psnr(g,a,b):
    """
    si a n'est pas un int16, alors a doit être compris entre -1 et 1
    """
    # gamma is returned such that psnr(a,a+gamma*b)=g
    sigma = math.sqrt(np.mean( np.power(b,2) ))
    if type(a[0])==np.int16:
        max_intensity = float(2**15)
    else:
        max_intensity = 1.0
    return max_intensity/sigma*10**(-g/20)

def reshape(signal, max_len):
    "add zero padding"
    zero=np.zeros(max_len-len(signal))
    resultat=np.concatenate((signal,zero))
    return resultat

def cut_ech(signal):
    #cut 2s
    resultat=signal[0:2*16000]
    #sous ech 8khz
    resultat=resultat[0:len(resultat):2]
    return resultat

def totensor(signal):
    return torch.from_numpy(signal).float()

def totensor_cuda(signal):
    print("To Do")
    
def normalise(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean/np.amax(np.absolute(tensor_minusmean))
