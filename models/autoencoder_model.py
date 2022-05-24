# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:28:55 2018

@author: Mou
"""

from torch.nn import Module, Sequential, Linear, Conv2d, LeakyReLU, Sigmoid, MaxPool2d, ConvTranspose2d
    
class mlaem_model(Module):
   def __init__(self):
        super(mlaem_model, self).__init__()
        self.encoder = Sequential(
            Linear(224, 128),
            LeakyReLU(0.1),
            Linear(128, 64),
            LeakyReLU(0.1),
            Linear(64, 16),
            LeakyReLU(0.1),
            Linear(16, 4)
            )
        
        self.decoder_linearpart = Sequential(
            Linear(4, 224, bias=False),
            )
        
        self.decoder_nonlinearpart = Sequential(
            Linear(224, 224, bias=True),
            Sigmoid(),
            Linear(224, 224, bias=True)
            )

            
   def forward(self, x):
        x_latent = self.encoder(x)
        x_latent = x_latent.abs()
        x_latent = x_latent.t()/x_latent.sum(1)
        x_latent = x_latent.t()
        x_linear = self.decoder_linearpart(x_latent)
        x = self.decoder_nonlinearpart(x_linear)
        return x, x_latent
    
   def get_endmember(self, x):
        endmember = self.decoder_linearpart(x)
        return endmember 
    

    
    
    
    