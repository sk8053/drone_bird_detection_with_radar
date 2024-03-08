# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:34:09 2023

@author: gangs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from classification_models import Res_Convnet
import pickle
from utils import conf, DEVICE
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
if (os.path.isfile('data/total_data/bird_data_total.pkl') and  os.path.isfile('data/total_data/mavik_data_total.pkl')) is False:
  download_data()
  
model = Res_Convnet(input_channels = 1, num_classes = 2)

device = DEVICE
print(f'================ device is {device} =====================')
n_iter = conf.n_iter_conv
SNRs = conf.SNRs_conv
n_sample_size = conf.n_sample_size_conv

model.load_state_dict(torch.load('checkpoint/checkpoints_res_conv.pth')['model'])
model.to(device)
model.eval()

### attenuation factor ranging from [a1, a2]
a2 = conf.a2
a1 = conf.a1
################

print(f'n_iter = {n_iter}')
print(f'n_sample_size = {n_sample_size}')
print(f'SNR list = {SNRs}')
print(f'a1 and a2 = {a1} and {a2}')
print('=============================================')

print("Num params: ", sum(p.numel() for p in model.parameters()))

with open('data/total_cut_data/bird_data_total.pkl', 'rb') as f:
  bird_data_total = pickle.load(f)[:,None]
with open('data/total_cut_data/mavik_data_total.pkl', 'rb') as f:
  mavik_data_total = pickle.load(f)[:,None]


I = np.random.permutation(len(mavik_data_total))
mavik_data_total = mavik_data_total[I][:len(bird_data_total)]


print (bird_data_total.shape, mavik_data_total.shape)


# set the range of aplitudes as [0,1]
mavik_data_total /= (np.max(np.abs(mavik_data_total)))
bird_data_total /= (np.max(np.abs(bird_data_total)))

bird_data_total = torch.tensor(bird_data_total, dtype = torch.complex32)
mavik_data_total = torch.tensor(mavik_data_total, dtype = torch.complex32)

total_data = torch.concat([bird_data_total, mavik_data_total], dim = 0)

mavik_label = np.repeat([1], len(mavik_data_total))
bird_label = np.repeat([0], len(bird_data_total))
total_label = np.append(bird_label, mavik_label)

total_data = torch.tensor(total_data, dtype = torch.complex32)
total_label = torch.tensor(total_label, dtype = torch.float32)


acc_list = []
snr_list = []

snr_to_acc = []
cm_list = []
for snr in tqdm(SNRs, desc = 'SNRs', ascii=True):
    acc_sum = 0
    cm_sum = 0
    for iter_ in range(n_iter):
        
        # shuffle data every iteration
        I= torch.randperm(len(total_data))
        # after shuffling, take the first n samples
        total_data2 = total_data[I][:n_sample_size]       
        # attenuation effect in far distance
        att = (torch.sqrt(torch.rand(1)*a2 +a1)).type(torch.complex32)
        total_data2 = att*total_data2
        
        amp = total_data2.abs().mean()
        noise_std = amp/(10**(0.05*snr))
        # compute SNR over n samples
        #avg_snr = 20*torch.log10(amp/noise_std)
        #avg_snr = avg_snr.detach().cpu().numpy()
        
        
        total_label2 = total_label[I][:n_sample_size].to(device).type(torch.float32)
        
        noise_power = (torch.normal(0,1,size = total_data2.size()) + 1j*torch.normal(0,1,size = total_data2.size()))*noise_std/torch.sqrt(torch.tensor(2))
        total_data2 += noise_power
     
        
        y_pred_values = model(total_data2.abs().type(torch.float32).to(device))
        _, y_pred = torch.max(y_pred_values, 1)
        acc = torch.sum(y_pred==total_label2)/n_sample_size
        cm = confusion_matrix(y_pred.detach().cpu().numpy(), total_label2.detach().cpu().numpy(), labels = [0,1]) # bird 0, drone 1
        cm_sum += cm
        acc_sum += acc.detach().cpu()

    cm_sum = cm_sum/cm_sum.sum(0)[None]
    cm_list.append(cm_sum)
    snr_to_acc.append(acc_sum/n_iter)
    
plt.scatter(SNRs, snr_to_acc)
snr_to_acc_data = np.column_stack((SNRs, snr_to_acc))

np.savetxt('data/plotting_data/test_data_conv.txt', snr_to_acc)

plt.savefig('figures/test_accuracy_conv_net.png')

#### confusion matrix ###
cm_list = np.array(cm_list)

h00 = cm_list[:,0, 0]
h01 = cm_list[:,0, 1]
h10 = cm_list[:,1, 0]
h11 = cm_list[:,1, 1]

plt.figure()
plt.plot(SNRs, h00, label = 'h00')
plt.plot(SNRs, h01, label = 'h01')
plt.plot(SNRs, h10, label = 'h10')
plt.plot(SNRs, h11, label = 'h11')
plt.legend()
plt.grid()
plt.savefig('figures/false_alarm_conv.png')
plt.close()

np.savetxt('data/plotting_data/h00_conv.txt', h00)
np.savetxt('data/plotting_data/h01_conv.txt', h01)
np.savetxt('data/plotting_data/h10_conv.txt', h10)
np.savetxt('data/plotting_data/h11_conv.txt', h11)