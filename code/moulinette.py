# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:23:20 2019

@author: Loïc
"""

"""
Script permettant d'extraire les data .WAV

fonction 'chacha20' pour trier
fonction 'salsa20' pour supprimer les doublons

DL sph2pipe pour réaliser la conversion  NIST SPHERE file en vraie .wav file :
https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools

lancer un CMD pour faire la conversion
>> cd chemin
>> forfiles /s /m *.wav /c "cmd /c sph2pipe -f wav @file @fnameRIFF.wav"
"""

import os
import shutil

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import winsound
import random


#Fonction
###############################################################################
def chacha20(chemin,nouveau_chemin):
    i=0
    for folder in os.listdir(chemin):
        if os.path.isdir(chemin+"/"+folder):
            for subfolder in os.listdir(chemin+"/"+folder):
                if os.path.isdir(chemin+"/"+folder+"/"+subfolder):
                    chemin_file = chemin+"/"+folder+"/"+subfolder
                    for file in os.listdir(chemin_file):
                        if file[-3::] == 'WAV':
                            source = chemin_file+"/"+file
                            destination = nouveau_chemin+"/"+folder+"_"+subfolder+"_"+file
                            shutil.copy(source, destination)
                            i=i+1
    print(i, "Fichier .WAV copié")

def salsa20(chemin):
    i=0
    for file in os.listdir(chemin):
        if os.path.isfile(chemin+"/"+file):
            if file[-8::] != 'RIFF.wav':
                os.remove(chemin+"/"+file)
                i=i+1
    print(i, "Fichier .WAV supprimé")
    

def bruit_random(chemin_bruit,chemin_soure,chemin_resultat):
    i=0
    fbruit, bruit = wavfile.read(chemin_bruit)
    N_bruit = len(bruit)
    for file in os.listdir(chemin_soure):
        if os.path.isfile(chemin_soure+"/"+file):
            if file[-8::] == 'RIFF.wav':
                fs, s1 = wavfile.read(chemin_soure+"/"+file)
                s = np.array(s1)
                s = s / max(abs(s))
                N_signal = len(s)
                N_chunck = N_bruit // N_signal
                r1=random.randint(1, N_chunck-1)
                r2=random.randint(1, N_chunck-1)
                bruit_random = bruit[r1*N_signal:(r1+1)*N_signal] + bruit[r2*N_signal:(r2+1)*N_signal]
                bruit_random = bruit_random / max(abs(bruit_random))
                signal_bruite = s + bruit_random
                signal_bruite=np.array(signal_bruite, dtype='int16')
                wavfile.write(chemin_resultat+"/"+file, fs, signal_bruite)
                i=i+1
    print(i, "Fichier RIFF.wav bruités")
    
#Script
###############################################################################

#Trie
# Ne pas oublier de mettre des / et pas des \
    
# acquisition du current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

chemin_test = dir_path+'/Raw_data/TIMIT_TEST'
nouveau_chemin_test = dir_path+'/data/data_test'

chemin_train = dir_path+'/Raw_data/TIMIT_TRAIN'
nouveau_chemin_train = dir_path+'/data/data_train'

#chacha20(chemin_test,nouveau_chemin_test)
#chacha20(chemin_train,nouveau_chemin_train)

#Suppression doublon
#salsa20(chemin)



#affichage signal
# chemin_file = "C:/Users/Loïc/Documents/3A/projet audio/Data_train/DR1_FDAW0_SA1RIFF.wav"
# fs, s1 = wavfile.read(chemin_file)
# s = np.array(s1)
# s = s / max(abs(s))

# plt.plot(s)
# plt.xlabel("Sample")
# plt.ylabel("amplitude normalisée)")
# plt.title("CHACHA20 best name ever")
# plt.show()

#lecture via carte son
#winsound.PlaySound(chemin_file,winsound.SND_FILENAME)

#bruitage
"""
Attention la fréquence d'échantillonage du signal du bruit et des voix ne sont 
pas égaux donc j'ai pris la plus petit : 16kHz celle des voix pour construire
des signaux bruités.
"""
# chemin_bruit=dir_path+"/data/babble.wav"
# chemin_soure=dir_path+"/data/data_test"
# chemin_resultat=dir_path+"/data/data_test_bruit"
# bruit_random(chemin_bruit,chemin_soure,chemin_resultat)

# chemin_soure=dir_path+"/data/data_train"
# chemin_resultat=dir_path+"/data/data_train_bruit"
# bruit_random(chemin_bruit,chemin_soure,chemin_resultat)





