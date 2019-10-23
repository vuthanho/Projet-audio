# -*- coding: utf-8 -*-
"""
Script contenant des fonctions utiles
"""

import numpy as np
import math

def psnr(a1,a2):
    mse = numpy.mean( (a1 - a2) ** 2 )
    if mse == 0:
        return 100
    if type(a1[0])==np.int16:
        max_intensity = 16384.0
    else:
        max_intensity = 1.0
    return 20 * math.log10(max_intensity / math.sqrt(mse))

def reverse_psnr(g,a1,a2):
    # a1 is fixed and a2 is returned such that psnr(a1,a2)=g
    return 

def reshape(signal, max_len):
    "add zero padding"
    zero=np.zeros(max_len-len(signal))
    resultat=np.concatenate((signal,zero))
    return resultat

