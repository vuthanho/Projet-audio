 # -*- coding: utf-8 -*-
"""
Classe permettant de load les données. Penser pour être utilisé avec un
itérateur comme le dataloader de pytorch
Le but est de récupérer pour le train le module du spectrogramme
et pour le test le module et la phase

exemple :
    -testset = SpeechDataset(test_bruit_path, test_path, transform=['reshape','normalisation','test','tensor_cuda'])
    -trainset = SpeechDataset(train_bruit_path, train_path, transform=['reshape','normalisation','train','tensor_cuda'])
"""
import os
import torch
from scipy.io import wavfile #for audio processing
from scipy.signal import spectrogram
from codes import toolkit
from math import floor

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
        
        _, signal_noised = wavfile.read(file_name_noised)
        _, signal = wavfile.read(file_name)

        if 'reshape' in self.transform:
            signal_noised = toolkit.reshape(signal_noised, self.max_len)
            signal = toolkit.reshape(signal, self.max_len)
            
        if 'normalisation' in self.transform:
            signal_noised = toolkit.normalise(signal_noised)
            signal = toolkit.normalise(signal)
        
        if 'train' in self.transform:
            fs=16000
            nperseg = floor(0.03*fs)
            noverlap=nperseg//2
            _,_,signal_noised = spectrogram(signal_noised, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            _,_,signal = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            sample = {'signal_noised': signal_noised, 'signal' : signal}
        
        if 'test' in self.transform:
            fs=16000
            nperseg = floor(0.03*fs)
            noverlap=nperseg//2
            _,_,signal_noised = spectrogram(signal_noised, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            _,_,signal = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            _,_,angle_noised = spectrogram(signal_noised, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='angle')
            sample = {'signal_noised': signal_noised, 'signal' : signal, 'angle' : angle_noised}
        
        if 'tensor' in self.transform:
            signal_noised = toolkit.totensor(signal_noised)
            signal = toolkit.totensor(signal)
        
        if 'tensor_cuda' in self.transform:
            signal_noised = toolkit.totensor(signal_noised)
            signal = toolkit.totensor(signal)
            signal_noised=signal_noised.to(torch.device("cuda:0"))
            signal=signal.to(torch.device("cuda:0"))

        return sample
    # istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
