# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:42:19 2019

@author: Loïc
"""

import os
import torch
import numpy as np

t=torch.tensor([[1,2],[3,4]])
x=t.numpy()

t=torch.tensor([[1,2],[3,4]],device = "cuda:0")
print(t)
x=t.to("cpu").numpy()
print(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)