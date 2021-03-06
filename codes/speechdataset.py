 # -*- coding: utf-8 -*-
"""
@author: Olivier
@author: Loïc
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
import numpy as np
from scipy.io import wavfile #for audio processing
from scipy.signal import spectrogram
from codes import toolkit
from math import floor
import numpy as np

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
            
        if 'cut&sousech' in self.transform:
            signal_noised = toolkit.cut_ech(signal_noised)
            signal = toolkit.cut_ech(signal)
            
        if 'normalisation' in self.transform:
            signal_noised = toolkit.normalise(signal_noised)
            signal = toolkit.normalise(signal)
        
        if 'train' in self.transform:
            fs=8000 # Car sous échantillonnage ?
            nperseg = 256#floor(0.03*fs)
            noverlap=nperseg//2
            _,_,signal_noised = spectrogram(signal_noised, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            _,_,signal = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
#            signal_noised = 2 * signal_noised / np.sum(np.hanning(nperseg))
#            signal = 2 * signal / np.sum(np.hanning(nperseg))
            
#            #centré reduit par bande de f
#            for k in range(np.shape(signal)[0]):
#                signal[k,:]=(signal[k,:]-np.mean(signal[k,:])) /np.std(signal[k,:])
#                signal_noised[k,:]=(signal_noised[k,:]-np.mean(signal_noised[k,:]))/np.std(signal_noised[k,:])
#           
            #centré reduit du spectro → aucun sens physique
#            signal=(signal-np.mean(signal)) /np.std(signal)
#            signal_noised=(signal_noised-np.mean(signal_noised))/np.std(signal_noised)
#            
            #min max par bande de f
#            for k in range(np.shape(signal)[0]):
#                signal[k,:]=(signal[k,:]-np.min(signal[k,:])) /(np.max(signal[k,:])-np.min(signal[k,:]))
#                signal_noised[k,:]=(signal_noised[k,:]-np.min(signal_noised[k,:])) /(np.max(signal_noised[k,:])-np.min(signal_noised[k,:]))
#           
#            min max du spectro → aucun sens physique
            signal=(signal-np.min(signal))/(np.max(signal)-np.min(signal))
            signal_noised=(signal_noised-np.min(signal_noised))/(np.max(signal_noised)-np.min(signal_noised))
            
            # sample = {'signal_noised': signal_noised, 'signal' : signal}

            # Normalisation du spectre
            s_min = signal.min()
            s_max = signal.max()
            n_min = signal_noised.min()
            n_max = signal_noised.max()

            signal = np.divide(signal-s_min,s_max-s_min)
            signal_noised = np.divide(signal_noised-n_min,n_max-n_min)

            signal = signal[None,...]
            signal_noised = signal_noised[None,...]
            sample = [signal_noised,signal]
        
        if 'test' in self.transform:
            fs=8000
            nperseg = 256 #floor(0.03*fs)
            noverlap=nperseg//2
            temp=signal_noised
            _,_,signal_noised = spectrogram(signal_noised, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            _,_,signal = spectrogram(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='magnitude')
            _,_,angle_noised = spectrogram(temp, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=None, detrend=False, return_onesided=True, scaling='spectrum', axis=-1, mode='angle')
            # sample = {'signal_noised': signal_noised, 'signal' : signal, 'angle' : angle_noised}

            # Normalisation du spectre
            s_min = signal.min()
            s_max = signal.max()
            n_min = signal_noised.min()
            n_max = signal_noised.max()

            signal = np.divide(signal-s_min,s_max-s_min)
            signal_noised = np.divide(signal_noised-n_min,n_max-n_min)

            signal = signal[None,...]
            signal_noised = signal_noised[None,...]
            #angle_noised = angle_noised[None,...]
            sample = [signal_noised,signal,angle_noised]
        
        if 'tensor' in self.transform:
            signal_noised = toolkit.totensor(signal_noised)
            signal = toolkit.totensor(signal)
        
        # if 'tensor_cuda' in self.transform:
        #     signal_noised = toolkit.totensor(signal_noised)
        #     signal = toolkit.totensor(signal)
        #     signal_noised=signal_noised.to(torch.device("cuda:0"))
        #     signal=signal.to(torch.device("cuda:0"))

        return sample
    # istft(Zxx, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, input_onesided=True, boundary=True, time_axis=-1, freq_axis=-2)
