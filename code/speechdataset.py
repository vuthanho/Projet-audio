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

    def __init__(self, root_dir_noise, root_dir, transform=[None]):
        """
        Args:
            root_dir_noise (string): Directory with all the file .wav.
            transform (callable, optional): Optional transform to be applied
                on a sample(file).
        """
        self.root_dir_noise = root_dir_noise
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = self.max_len_function()

    def __len__(self):
        for root, _, files in os.walk(self.root_dir_noise):
            #files est une liste contenant tous les fichiers
            return len(files)

    
    def max_len_function(self):
        nb=0
        for file in os.listdir(self.root_dir_noise):
            file_name = os.path.join(self.root_dir_noise, file)
            fs, signal = wavfile.read(file_name)
            if nb<len(signal):
                nb=len(signal)
        return nb
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        i=0
        for file in os.listdir(self.root_dir_noise):
            if i==idx:
                file_name_noised = os.path.join(self.root_dir_noise, file)
                file_name= os.path.join(self.root_dir, file)
            i=i+1
        
        fs_noised, signal_noised = wavfile.read(file_name_noised)
        fs, signal = wavfile.read(file_name)

        if 'reshape' in self.transform:
            signal_noised = toolkit.reshape(signal_noised, self.max_len)
            signal = toolkit.reshape(signal, self.max_len)
        
        if 'tensor' in self.transform:
            signal_noised = toolkit.totensor(signal_noised)
            signal = toolkit.totensor(signal)
            
        if 'tensor_cuda' in self.transform:
            print("To Do")
            
        if 'normalisation' in self.transform:
            print("To Do")
            
        sample = {'signal_noised': signal_noised, 'signal' : signal}

        return sample