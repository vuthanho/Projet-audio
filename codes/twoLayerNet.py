# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 22:08:20 2019

@author: Loïc
"""

import torch


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
        x = torch.nn.functional.relu(self.second_conv(x))
        # x = x.view(-1, 24363600)
        # x = torch.nn.functional.relu(self.first_fully_c(x))

        return x


