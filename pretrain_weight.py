# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 09:39:49 2018

@author: Mou
"""

import torch
from torch.nn import Module, Sequential, Linear, LeakyReLU, Sigmoid
from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable

# Define the network

class encoder_net(Module):
    def __init__(self):
        super(encoder_net, self).__init__()
        self.encoder = Sequential(
            Linear(224, 64),
            LeakyReLU(0.1),
            Linear(64, 16),
            LeakyReLU(0.1),
            Linear(16, 4)
            )
           
    def forward(self, x):
        x = self.encoder(x)
        return x

class decoder_net(Module):
    def __init__(self):
        super(decoder_net, self).__init__()
        self.decoder = Sequential(
            Linear(4, 16),
            LeakyReLU(0.1),
            Linear(16, 64),
            LeakyReLU(0.1),
            Linear(64, 224)
            )
           
    def forward(self, x):
        x = self.decoder(x)
        return x
    
class decoder_nonlinear(Module):
    def __init__(self):
        super(decoder_nonlinear, self).__init__()
        self.decoder = Sequential(
            Linear(224, 224),
            Sigmoid(),
            Linear(224, 224, bias=True)
            )
           
    def forward(self, x):
        x = self.decoder(x)
        return x

def pretrain_encoder(hsi,abundance):
    
    GPU_NUMS = 1
    learning_rate = 1e-3
    EPOCH = 1501
    
    hsi = torch.from_numpy(hsi)
    hsi = Variable(hsi).cuda()

    abundance = torch.from_numpy(abundance)
    abundance = abundance.t()
    abundance = Variable(abundance).cuda()

    model = encoder_net()
    model = model.cuda() if GPU_NUMS > 0 else model

    criterion = MSELoss()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    for epoch in range(1, EPOCH):
        output = model(hsi)
        loss = criterion(output, abundance) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(epoch,loss.item())
   
    torch.save(model.encoder.state_dict(), 'pretrain_encoder.pth')
    return model.encoder.state_dict()


def pretrain_weight(W_init,num_endmember):
    
    GPU_NUMS = 1
    learning_rate = 1e-3
    EPOCH = 401

    W_init = W_init.t()
    W_init = W_init.cuda()

    model = decoder_net()
    model = model.cuda() if GPU_NUMS > 0 else model

    criterion = MSELoss()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    code_onehot = torch.eye(num_endmember)  
    code_onehot = Variable(code_onehot).cuda()


    for epoch in range(1, EPOCH):
        output = model(code_onehot)
        loss = criterion(output, W_init) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(epoch,loss.item())
   
    torch.save(model.decoder.state_dict(), 'pretrain_decoder.pth')
    return model.decoder.state_dict()

def pretrain_dec_nonlipart(hsi):
    
    GPU_NUMS = 1
    learning_rate = 1e-3
    EPOCH = 4001
    
    hsi = torch.from_numpy(hsi)
    hsi = Variable(hsi).cuda()

    model = decoder_nonlinear()
    model = model.cuda() if GPU_NUMS > 0 else model

    criterion = MSELoss()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(1, EPOCH):
        output = model(hsi)
        loss = criterion(output, hsi) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch,loss.item())
   
    torch.save(model.decoder.state_dict(), 'pretrain_decoder_nonlinear.pth')
    return model.decoder.state_dict()