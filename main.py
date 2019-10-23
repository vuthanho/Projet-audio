# -*- coding: utf-8 -*-
"""
Main script du projet
"""

from code.speechdataset import SpeechDataset
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
train_bruit_path = dir_path+'/data/data_train_bruit'
train_path = dir_path+'/data/data_train'

data_train_bruit = SpeechDataset(train_bruit_path, train_path, transform=['reshape','tensor'])
sample = data_train_bruit[2]
print("nombre de fichier : ",len(data_train_bruit))
nb=data_train_bruit.max_len_function()
print("max_len : ",nb)
print("len signal : ",len(sample['signal']))

dataloader = torch.utils.data.DataLoader(data_train_bruit, batch_size=4, shuffle=True, num_workers=4)