# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 08:36:01 2019

@author: Loïc
"""
import os
import torch
from scipy.io import wavfile #for audio processing

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
            return len(files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        i=0
        for file in os.listdir(self.root_dir):
            if i==idx:
                file_name = os.path.join(self.root_dir, file)
            i=i+1
        
        fs, signal = wavfile.read(file_name)
        sample = {'signal': signal}

        if self.transform:
            sample = self.transform(sample)

        return sample