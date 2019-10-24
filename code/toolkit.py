# -*- coding: utf-8 -*-
"""
Script contenant des fonctions utiles
"""

import numpy as np
import math
import torch

def psnr(a1,a2):
    mse = np.mean( (a1 - a2) ** 2 )
    if mse == 0:
        return 100
    if type(a1[0])==np.int16:
        max_intensity = float(2**15)
    else:
        max_intensity = 1.0
    return 20 * math.log10(max_intensity / math.sqrt(mse))

def reverse_psnr(g,a,b):
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

def totensor(signal):
    return torch.from_numpy(signal)

def totensor_cuda(signal):
    print("To Do")
    
def normalise(signal):
    print("To Do")
