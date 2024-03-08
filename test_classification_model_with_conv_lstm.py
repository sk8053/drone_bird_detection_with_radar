# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:04:21 2023

@author: seongjoon kang
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from classification_models import ConvLSTM
from utils import conf, DEVICE
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils.download_data import download_data
import os
if (os.path.isfile('data/total_data/bird_data_total.pkl') and  os.path.isfile('data/total_data/mavik_data_total.pkl')) is False:
  download_data()

def collect_data_in_bin(data, bin_size = 5, class_label = 1):
    data_new = []
    target_new = []
    L, W = data.shape[-2], data.shape[-1]
    for i in range(len(data)-bin_size):
        data_i = data[i:i+bin_size]
        data_i = data_i.reshape(1,bin_size, L,W)
        
        data_new.append(data_i)
        target_new.append(class_label)
    data_new = np.array(data_new).reshape(-1,bin_size, L,W)
    target_new = np.array(target_new)
    
    return data_new, target_new


time_bin_size = conf.time_bin_size_conv_lstm
n_iter = conf.n_iter_conv_lstm
n_hidden = conf.n_hidden_conv_lstm
n_sample_size = conf.n_sample_size_conv_lstm
SNRs = conf.SNRs_conv_lstm
device = DEVICE
### attenuation factor ranging from [a1, a2]
a2 = conf.a2
a1 = conf.a1
################

print(f'n_iter = {n_iter}')
print(f'n_hidden = {n_hidden}')
print(f'n_sample_size = {n_sample_size}')
print(f'SNR list = {SNRs}')
print(f'time bin size = {time_bin_size}')
print(f'================ device is {device} =====================')
print(f'a1 and a2 are {a1} and {a2}')

model = ConvLSTM(input_dim=1, hidden_dim=n_hidden, kernel_size=(3,3), num_layers=5, batch_first=True)
model.to(device)
model.load_state_dict(torch.load(f'checkpoint/checkpoint_convlstm_{n_hidden}.pth')['model'])
model.eval()
print("Num params: ", sum(p.numel() for p in model.parameters()))



with open('data/total_cut_data/bird_data_total.pkl', 'rb') as f:
  bird_data_total = pickle.load(f)[:,None]
with open('data/total_cut_data/mavik_data_total.pkl', 'rb') as f:
  mavik_data_total = pickle.load(f)[:,None]
  
I = np.random.permutation(len(mavik_data_total))
mavik_data_total = mavik_data_total[I][:len(bird_data_total)]
print (bird_data_total.shape, mavik_data_total.shape)

mavik_data_total /= np.max(np.abs(mavik_data_total))
bird_data_total /= np.max(np.abs(bird_data_total))

angle_max = np.pi

bird_data_total, bird_label = collect_data_in_bin(bird_data_total, bin_size = time_bin_size, class_label = 0)
mavik_data_total, mavik_label = collect_data_in_bin(mavik_data_total, bin_size = time_bin_size, class_label = 1)

total_data = np.vstack((bird_data_total, mavik_data_total))
total_label = np.append(bird_label, mavik_label)

total_data = torch.tensor(total_data, dtype = torch.complex32)
total_label = torch.tensor(total_label, dtype = torch.complex32)

snr_to_acc = []
cm_list = []
for snr in tqdm(SNRs, desc = 'SNRs', ascii=True):
    acc_sum = 0
    cm_sum = 0
    for i in range(n_iter):
        I= torch.randperm(len(total_data))
        total_data2 = total_data[I][:n_sample_size]
        total_label2 = total_label[I][:n_sample_size]
        total_label2 = total_label2.to(device).type(torch.float32)
        
        # attenuation effect in far distance
        att = (torch.sqrt(torch.rand(1)*a2 + a1)).type(torch.complex32)
        total_data2 = att*total_data2
        
        # compute SNR over samples
        #print((total_data2.abs()).mean(-1).mean(-1).squeeze())
        amp = total_data2.abs().mean()
        noise_std = amp/(10**(0.05*snr))
        
        # add noise power
        noise_power = (torch.normal(0,1,size = total_data2.size()) + 1j*torch.normal(0,1,size = total_data2.size()))*noise_std/torch.sqrt(torch.tensor(2))
        total_data2 = total_data2 + noise_power
        total_data2 = torch.angle(total_data2)/angle_max
        total_data2 = total_data2[:,:,None]
        
        _,_, y_pred_value= model(total_data2.to(device))
        
        _, y_pred = torch.max(y_pred_value, 1)
        acc = torch.sum(y_pred==total_label2)/n_sample_size
        cm = confusion_matrix(y_pred.detach().cpu().numpy(), total_label2.detach().cpu().numpy(), labels = [0,1]) # bird 0, drone 1
        cm_sum += cm
        acc_sum += acc.detach().cpu()

   
    cm_sum = cm_sum/cm_sum.sum(0)[None]
    cm_list.append(cm_sum)
    snr_to_acc.append(acc_sum/n_iter)

plt.scatter(SNRs, snr_to_acc)
snr_to_acc_data = np.column_stack((SNRs, snr_to_acc))

np.savetxt('data/plotting_data/test_data_conv_lstm.txt', snr_to_acc)
plt.savefig('figures/test_accuracy_conv_lstm_net.png')

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
plt.savefig('figures/false_alarm_convlstm.png')
plt.close()

np.savetxt('data/plotting_data/h00_conv_lstm.txt', h00)
np.savetxt('data/plotting_data/h01_conv_lstm.txt', h01)
np.savetxt('data/plotting_data/h10_conv_lstm.txt', h10)
np.savetxt('data/plotting_data/h11_conv_lstm.txt', h11)
