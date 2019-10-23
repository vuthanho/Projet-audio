# -*- coding: utf-8 -*-
"""
Main script du projet
"""

from code.speechdataset import SpeechDataset

data_train_bruit = SpeechDataset("C:/Users/Loïc/Documents/3A/projet audio/data/Data_train_bruit")
sample = data_train_bruit[2]
print(len(data_train_bruit))
nb=data_train_bruit.max_len()
print(nb)