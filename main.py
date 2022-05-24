# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:57:41 2018

@author: Mou
"""

import os
import scipy.io as sio
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from torch.autograd import Variable

from models.autoencoder_model import mlaem_model

from pretrain_weight import pretrain_weight, pretrain_dec_nonlipart

from utils.ProgressBar import ProgressBar


model_name = 'mlaem_model'
workspace = 'D:\\workspace\\AE_HSU_M1'
GPU_NUMS = 1
EPOCH = 40
BATCH_SIZE = 1024
learning_rate = 1e-3
num_endmember = 4
num_repeat = 1
SNR = 10
model_type = 'linear'

## --------------------- functions -------------------------------------------

    

## --------------------- Load the data ---------------------------------------
adundance_name = 'abundance_%ddb' %SNR 
file_path = os.path.join(workspace, "data", "%s.mat" % adundance_name)
datamat = sio.loadmat(file_path)
adundance = datamat[adundance_name]

hsi_name = 'data_%s_%ddb' %(model_type,SNR)
file_path = os.path.join(workspace, "data", "%s.mat" % hsi_name)
datamat = sio.loadmat(file_path)
hsi = datamat[hsi_name]

endmember_name = 'weight_%s_%ddb' %(model_type,SNR)
file_path = os.path.join(workspace, "data", "%s.mat" % endmember_name)
datamat = sio.loadmat(file_path)
W_init = datamat[endmember_name]
W_init = torch.from_numpy(W_init)

if np.size(adundance,1)!=np.size(hsi):
    adundance = np.transpose(adundance)
    hsi = np.transpose(hsi)
## ---------------------------------------------------------------------------


##迭代多次
    
for iter in range(1, num_repeat+1):
    
## --------------------- initialize the network ------------------------------
    model = mlaem_model()
    model.decoder_linearpart[0].weight.data = W_init
    dec_nonlipart = pretrain_dec_nonlipart(hsi)
    model.decoder_nonlinearpart.load_state_dict(dec_nonlipart)

    
    model = model.cuda() if GPU_NUMS > 0 else model


    print(model)

    criterion = MSELoss()

## ------------------- Fine Tune ---------------------------------------------
    if model_name=='mlaem_model':
        params1 = map(id, model.decoder_linearpart.parameters())
        params2 = map(id, model.decoder_nonlinearpart.parameters())
        ignored_params = list(set(params1).union(set(params2)))      
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters()) 
        optimizer = Adam([
                {'params': base_params},
                {'params': model.decoder_linearpart.parameters(), 'lr': 1e-5}, 
                {'params': model.decoder_nonlinearpart.parameters(), 'lr': 1e-5 }, 
                ], lr=learning_rate, weight_decay=1e-5)
    else:
        ignored_params = list(map(id, model.decoder.parameters()))      
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters()) 
        optimizer = Adam([
                {'params': base_params},
                {'params': model.decoder.parameters(), 'lr': 1e-4}        
                ], lr=learning_rate, weight_decay=1e-5)

    
    vector_all = []
    code_onehot = torch.eye(num_endmember)  
    code_onehot = Variable(code_onehot).cuda()

    data_loader = DataLoader(hsi, batch_size=BATCH_SIZE, shuffle=False)

    proBar = ProgressBar(EPOCH, len(data_loader), "Loss:%.5f")


    for epoch in range(1, EPOCH+1):
        for data in data_loader:
            pixel = data
            pixel = Variable(pixel).cuda() if GPU_NUMS > 0 else Variable(pixel)
            # ===================forward=====================

            output, vector = model(pixel)
            
            loss_reconstruction = criterion(output, pixel)
            loss = loss_reconstruction
        
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            proBar.show(epoch, loss.item())
            # ===================log========================
            if epoch == EPOCH-1:
                vector_temp = vector.cpu().data
                vector_temp = vector_temp.numpy()
                vector_all = np.append(vector_all,vector_temp)

    torch.save(model.state_dict(), 'sim_autoencoder.pth')

    vector_all = vector_all.reshape(-1,num_endmember)
    name_vector_all = 'vector_all_%s_%ddb_%d.mat' %(model_type,SNR,iter) 
    sio.savemat(name_vector_all,{'vector_all':vector_all}) 


    endmember = model.get_endmember(code_onehot)
    endmember = endmember.cpu().data
    endmember = endmember.numpy()
    name_endmember = 'endmember_pre_%s_%ddb_%d.mat' %(model_type,SNR,iter) 
    sio.savemat(name_endmember,{'endmember':endmember})




