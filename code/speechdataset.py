# -*- coding: utf-8 -*-
"""
Classe permettant de load les données. Penser pour être utilisé avec un
itérateur comme le dataloader de pytorch
"""
import os
import torch
from scipy.io import wavfile #for audio processing
import code.toolkit as toolkit

class SpeechDataset(object):
    """SpeechDataset dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the file .wav.
            transform (callable, optional): Optional transform to be applied
                on a sample(file).
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        for root, _, files in os.walk(self.root_dir):
            #files est une liste contenant tous les fichiers
            return len(files)

    
    def max_len(self):
        nb=0
        for file in os.listdir(self.root_dir):
            file_name = os.path.join(self.root_dir, file)
            fs, signal = wavfile.read(file_name)
            if nb<len(signal):
                nb=len(signal)
        return nb
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        i=0
        for file in os.listdir(self.root_dir):
            if i==idx:
                file_name = os.path.join(self.root_dir, file)
            i=i+1
        
        fs, signal = wavfile.read(file_name)
        

        if self.transform=='reshape':
            signal = toolkit.reshape(signal, self.max_len())
            
        sample = {'signal': signal}

        return sample