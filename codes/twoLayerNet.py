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
        self.linear1 = torch.nn.Linear(D_in, H).cuda()
        self.linear2 = torch.nn.Linear(H, D_out).cuda()
        
    #forward : calcul a partir de l'entrée la sortie du réseau en appliquant les différentes couches successif définis dans le constructeur
    def forward(self, x):
#        x=x.to(torch.device("cuda:0"))
        h_relu = self.linear1(x).relu()
        y_pred = self.linear2(h_relu)
        return y_pred


