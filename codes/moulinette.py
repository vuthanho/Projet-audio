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
>> forfiles /p path_to_files /s /m *.wav /c "cmd /c path_to_sph2pipe -f wav @file @fnameRIFF.wav"
forfiles /p D:\Cours\3A\Projet_audio\data\data_test /s /m *.wav /c "cmd /c D:\Cours\3A\Projet_audio\sph2pipe_v2.5\sph2pipe -f wav @file @fnameRIFF.wav"
"""

import os
import shutil

from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import winsound
import random
import math
from toolkit import psnr, reverse_psnr



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
            if (file[-8::] != 'RIFF.wav') | (file[-12::] == 'RIFFRIFF.wav'):
                os.remove(chemin+"/"+file)
                i=i+1
    print(i, "Fichier .WAV supprimé")
    

def bruit_random(PSNR,chemin_bruit,chemin_soure,chemin_resultat):
    i=0
    fbruit, bruit = wavfile.read(chemin_bruit)
    N_bruit = len(bruit)
    for file in os.listdir(chemin_soure):
        if os.path.isfile(chemin_soure+"/"+file):
            if file[-8::] == 'RIFF.wav':
                fs, s1 = wavfile.read(chemin_soure+"/"+file)

                # Conversion des données en float 64 pour pouvoir les manipuler
                s = np.array(s1,dtype=np.float64)
                bruit = np.array(bruit,dtype=np.float64)

                # Normalisation de s
                s=1/(2**15)*s

                # Segmentation du bruit à ajouter pour qu'il soit de la même dimension que le signal
                N_signal = len(s)
                N_chunck = N_bruit // N_signal
                r1=random.randint(1, N_chunck-1)
                r2=random.randint(1, N_chunck-1)
                bruit_random = bruit[r1*N_signal:(r1+1)*N_signal] + bruit[r2*N_signal:(r2+1)*N_signal]
                bruit_random = bruit_random / math.sqrt(np.mean( np.power(bruit_random,2) ))

                # Ajout du bruit en fonction du PSNR choisi
                g = reverse_psnr(PSNR,s,bruit_random)
                signal_bruite = s + g*bruit_random

                # Préparation à la conversion en int16
                # Exception si signal_bruite sature
                if max(abs(signal_bruite))>1:
                    signal_bruite=2**15/max(abs(signal_bruite))*signal_bruite
                else:
                    signal_bruite=2**15*signal_bruite
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
# print(dir_path)

# chemin_test = dir_path+'/../Raw_data/TIMIT_TEST'
nouveau_chemin_test = dir_path+'/../data/data_test'

# chemin_train = dir_path+'/../Raw_data/TIMIT_TRAIN'
nouveau_chemin_train = dir_path+'/../data/data_train'

# chacha20(chemin_test,nouveau_chemin_test)
# chacha20(chemin_train,nouveau_chemin_train)

#Suppression doublon
# salsa20(nouveau_chemin_test)
# salsa20(nouveau_chemin_train)


#affichage signal
# chemin_file = "C:/Users/Loïc/Documents/3A/projet audio/data_train/DR1_FDAW0_SA1RIFF.wav"
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
pas égaux donc j'ai pris la plus petite : 16kHz celle des voix pour construire
des signaux bruités.
"""
chemin_bruit=dir_path+"/../data/babble.wav"
# Le psnr correspond est choisi de telle façon à ce que psnr(s,s+g*b) = PSNR
# où s et b sont normalisés
PSNR = 35

chemin_soure=dir_path+"/../data/data_test"
chemin_resultat=dir_path+"/../data/data_test_bruit"
bruit_random(PSNR,chemin_bruit,chemin_soure,chemin_resultat)

chemin_soure=dir_path+"/../data/data_train"
chemin_resultat=dir_path+"/../data/data_train_bruit"
bruit_random(PSNR,chemin_bruit,chemin_soure,chemin_resultat)

