# -*- coding: utf-8 -*-
"""
Main script du projet
"""

from code.speechdataset import SpeechDataset

data_train_bruit = SpeechDataset("C:/Users/Loïc/Documents/3A/projet-audio/data/data_train_bruit",transform="reshape")
sample = data_train_bruit[2]
print("nombre de fichier : ",len(data_train_bruit))
nb=data_train_bruit.max_len()
print("max_len : ",nb)
print("len signal : ",len(sample['signal']))

