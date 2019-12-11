# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:08:20 2019

@author: Loïc
"""

import torch
import matplotlib.pyplot as plt

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
            torch.nn.Conv2d(1,18,(9,8))
        )
        
    #forward : calcul a partir de l'entrée la sortie du réseau en appliquant les différentes couches successif définis dans le constructeur
    def forward(self, x):
        return x


class TwoLayerNet(torch.nn.Module):
    #Constructeur qui définis nos couches
    def __init__(self, D_in, H, D_out):
        """
        - N = batch size; D_in = input dimension;
        - H = hidden dimension; D_out = output dimension.
        """
        super(TwoLayerNet, self).__init__()
        self.first_conv = torch.nn.Conv2d(D_in, H, 5)
        self.second_conv = torch.nn.Conv2d(H, D_out, 5)
        # self.first_fully_c = torch.nn.Linear(H,1)
        
    #forward : calcul a partir de l'entrée la sortie du réseau en appliquant les différentes couches successif définis dans le constructeur
    def forward(self, x):
#        x=x.to(torch.device("cuda:0"))
        x = torch.nn.functional.relu(self.first_conv(x))
        for i in range(0,10):
            plt.pcolormesh(x[i][0].cpu().detach().numpy())
            plt.show()
        x = torch.nn.functional.relu(self.second_conv(x))
        # x = x.view(-1, 24363600)
        # x = torch.nn.functional.relu(self.first_fully_c(x))

        return x


