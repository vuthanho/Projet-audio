# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

#import wave

#wave.open("TIMIT_TRAIN/DR1/FCJF0/SA1.WAV",'rb')
#s1=open("babble.wav",'rb')



###############################



#import wave
#import binascii
#
#NomFichier = "babble.wav"
#Monson = wave.open(NomFichier,'r')	# instanciation de l'objet Monson
#
#print("\nNombre de canaux :",Monson.getnchannels())
#print("Taille d'un échantillon (en octets):",Monson.getsampwidth())
#print("Fréquence d'échantillonnage (en Hz):",Monson.getframerate())
#print("Nombre d'échantillons :",Monson.getnframes())
#print("Type de compression :",Monson.getcompname())
#
#TailleData = Monson.getnchannels()*Monson.getsampwidth()*Monson.getnframes()
#
#print("Taille du fichier (en octets) :",TailleData + 44)
#print("Nombre d'octets de données :",TailleData)
#
#print("\nAffichage d'une plage de données (dans l'intervalle 0 -",Monson.getnframes()-1,")")
#echDebut = 1
#echFin = 10
#
#print("\nN° échantillon	Contenu")
#
#Monson.setpos(echDebut)
#plage = echFin - echDebut + 1
#for i in range(0,plage):
#    print(Monson.tell(),'\t\t',binascii.hexlify(Monson.readframes(1)))
#
#Monson.close()



from scipy.io import wavfile #for audio processing
import numpy as np
import matplotlib.pyplot as plt
import winsound

#fs, s1 = wavfile.read("babble.wav")
#fs, s1 = wavfile.read("TIMIT_TRAIN/DR1/FCJF0/SA1.WAV")
fs, s1 = wavfile.read("SA1RIFF.wav")
s = np.array(s1)
s = s / max(abs(s))

plt.plot(s)
plt.xlabel("Sample")
plt.ylabel("amplitude normalisée)")
plt.title("CHACHA20 best name ever")
plt.show()

winsound.PlaySound(s,winsound.SND_FILENAME)

#########################################################


