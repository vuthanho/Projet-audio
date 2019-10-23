# -*- coding: utf-8 -*-
"""
Script contenant des fonctions utiles
"""
import numpy as np

def reshape(signal, max_len):
    "add zero padding"
    zero=np.zeros(max_len-len(signal))
    resultat=np.concatenate((signal,zero))
    return resultat