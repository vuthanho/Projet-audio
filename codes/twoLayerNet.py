# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:08:20 2019

@author: Loïc
"""

import torch

# Fully Convolutionnal Network

class FCN(torch.nn.Module): 
    #Constructeur qui définis nos couches
    def __init__(self):
        """
        Ce script se base sur l'interprétation du papier https://www.isca-speech.org/archive/Interspeech_2017/pdfs/1465.PDF
        faite par matlab dans leur example https://www.mathworks.com/help/deeplearning/examples/denoise-speech-using-deep-learning-networks.html
        """
        super(FCN, self).__init__()
        self.firstlayer = torch.nn.Sequential(
            torch.nn.Conv2d(1,18,(9,7),padding=(9//2,7//2)),
            torch.nn.BatchNorm2d(18),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(18,30,(5,1),padding=(5//2,1//2)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(30,8,(9,1),padding=(9//2,1//2)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True)
        )
        self.hiddenlayer = torch.nn.Sequential(
            torch.nn.Conv2d(8,18,(9,1),padding=(9//2,1//2)),
            torch.nn.BatchNorm2d(18),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(18,30,(5,1),padding=(5//2,1//2)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(30,8,(9,1),padding=(9//2,1//2)),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True)
        )
        self.lastlayer = torch.nn.Conv2d(8,1,(129,1),padding=(129//2,1//2))
        
    #forward : calcul a partir de l'entrée la sortie du réseau en appliquant les différentes couches successif définis dans le constructeur
    def forward(self, x):
        x = self.firstlayer(x)
        x = self.hiddenlayer(x)
        x = self.lastlayer(x)
        return x


