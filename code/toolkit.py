# -*- coding: utf-8 -*-
"""
Script contenant des fonctions utiles
"""
import numpy as np

import numpy as np
import math

def psnr(a1,a2):
    mse = numpy.mean( (a1 - a2) ** 2 )
    if mse == 0:
        return 100
    max_intensity = 255.0
    return 20 * math.log10(max_intensity / math.sqrt(mse))


def reshape(signal, max_len):
    "add zero padding"
    zero=np.zeros(max_len-len(signal))
    resultat=np.concatenate((signal,zero))
    return resultat