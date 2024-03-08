# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:49:30 2023

@author: seongjoon kang
"""
import torch 
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
import numpy as np
import pickle
from classification_models import MyDataSet, ConvLSTM
from utils import conf, DEVICE
from utils.download_data import download_data
import os
if (os.path.isfile('data/total_data/bird_data_total.pkl') and  os.path.isfile('data/total_data/mavik_data_total.pkl')) is False:
  download_data()


with open('data/total_data/bird_data_total.pkl', 'rb') as f:
  bird_data_total = pickle.load(f)[:,None]
with open('data/total_data/mavik_data_total.pkl', 'rb') as f:
  mavik_data_total = pickle.load(f)[:,None]


img_size = bird_data_total.shape[-2:]
batch_size = conf.batch_size_conv_lstm

time_bin_size = conf.time_bin_size_conv_lstm # the size of time bin

device = DEVICE
print(f'================ device is {device} =====================')
lr = conf.lr_conv_lstm
n_epoch = conf.n_epoch_conv_lstm
n_hidden = conf.n_hidden_conv_lstm

print(f'learning rate: {lr}')
print(f'n_hidden: {n_hidden}')

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
        

# take angle information for bird and drone
bird_data_total = np.angle(bird_data_total)
mavik_data_total = np.angle(mavik_data_total)

#I = np.random.permutation(len(mavik_data_total))
mavik_data_total = mavik_data_total[:len(bird_data_total)]

print (bird_data_total.shape, mavik_data_total.shape)

# set the range of data as <1
mavik_data_total /= np.max(mavik_data_total)
bird_data_total /= np.max(bird_data_total)
# create the labels of mavik (1) and bird (0)
mavik_lavel = np.repeat([1], len(mavik_data_total)) # label of drone
bird_label = np.repeat([0], len(bird_data_total)) # label of bird
#total_data = np.vstack((bird_data_total, mavik_data_total))

plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(np.squeeze(bird_data_total[1]))
ax1.set_title('phases from bird')
ax2.imshow(np.squeeze(mavik_data_total[2]))
ax2.set_title('phases from drone')
plt.savefig('figures/drone_and_bird_phase.png')
plt.close()

# collect data over time bin
bird_data_total, bird_label = collect_data_in_bin(bird_data_total, bin_size = time_bin_size, class_label = 0)
mavik_data_total, mavik_label = collect_data_in_bin(mavik_data_total, bin_size = time_bin_size, class_label = 1)

total_data = np.append(bird_data_total, mavik_data_total, axis = 0)
total_label = np.append(bird_label, mavik_lavel)

# add the dimension of channel
total_data = total_data[:,:,None]
# shuffle data 
I = np.random.permutation(len(total_data))
total_data = total_data[I]
total_label = total_label[I]
# change to tensor
total_data = torch.tensor(total_data, dtype=torch.float32)
total_label = torch.tensor(total_label, dtype = torch.float32)

# split data by train,validation, and test
train_size = int(0.3*len(total_data)) 

L = len(total_data)
train_data_set = MyDataSet(image = total_data[:train_size], target= total_label[:train_size])
val_data_set = MyDataSet(image = total_data[train_size:L], target= total_label[train_size:L])

train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True,drop_last=False )
val_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=True,drop_last=False )

print('shape is number of frame, time, channels, height and width')
print('train size: ', train_data_set.image.shape, train_data_set.target.shape)
print('validation data size: ', val_data_set.image.shape, val_data_set.target.shape)


####################################################################################################
######################################################################################################

conv_lstm = ConvLSTM(input_dim=1, hidden_dim=n_hidden, kernel_size=(3,3), num_layers=5,
                 batch_first=True, bias=True, return_all_layers=False, img_size = img_size)
model = conv_lstm
model.to (device)
print("Num params: ", sum(p.numel() for p in model.parameters()))

# Initialize the criterion
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

max_patience = 100
patience_counter =0
#def train_loop(model, criterion, optimizer,  train_loader, val_loader, n_epoch=50):

###### model train and test ##################
best_val = 0.0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
patience_counter = 0
# Training
for t in tqdm(range(n_epoch)):
  
  epoch_t_acc = 0
  epoch_t_loss = 0

  model.train()
  # Loop over the training set
  for train_data, targets in train_data_loader:
    # Put the inputs and targets on the write device
    train_data = train_data.to(device)
    targets = torch.tensor(targets, dtype = torch.int64)
    targets = targets.to (device)
    # Feed forward to get the logits
   
    _,_, y_pred = model.forward(train_data)
    #y_pred = y_pred[:,0]
   
    score, predicted = torch.max(y_pred, 1)
    
    # Compute the loss and accuracy
    loss = criterion ( y_pred,targets)
    
    # zero the gradients before running
    # the backward pass.
    optimizer.zero_grad()
    # Backward pass to compute the gradient
    # of loss w.r.t our learnable params. 
    loss.backward()
    optimizer.step()
  
    train_accuracy = (targets == predicted).sum().float() / len(targets)

    epoch_t_acc += train_accuracy.item()
    epoch_t_loss += loss.item()

  train_losses.append(epoch_t_loss/len(train_data_loader))
  train_accuracies.append(epoch_t_acc/len(train_data_loader))

# Switch the model to eval mode
  model.eval()
## evaluate validation accuracy with validation data ####
  v_acc = 0
  v_loss = 0  
  with torch.no_grad():
      # TLoop over the validation set 
      for val_data, val_targets in val_data_loader:
          # Put the inputs and targets on the write device
          val_data = val_data.to(device)
          val_targets = torch.tensor(val_targets, dtype = torch.int64)
          val_targets = val_targets.to(device)
          # Feed forward to get the logits
          _, _, y_pred_val = model (val_data)
          score, y_pred = torch.max(y_pred_val, 1)
          # Compute the loss and accuracy
          loss_val = criterion (y_pred_val,val_targets)
          accuracy_val = (val_targets == y_pred).sum().float() / len(val_targets)
          # Keep track of accuracy and loss
          v_acc += accuracy_val.item()
          v_loss += loss_val.item()

  val_losses.append(v_loss/len(val_data_loader))
  val_accuracies.append(v_acc/len(val_data_loader))

  if val_accuracies[-1] > best_val:
    best_val = val_accuracies[-1]
    patience_counter = 0
    print(f'------ model is saved ------{best_val}')
    # Save best model, optimizer, epoch_number
    torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': t,
            }, f'checkpoint/checkpoint_convlstm_{n_hidden}.pth')
  else:
    patience_counter += 1    
    if patience_counter > max_patience: 
      break
  print("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACCURACY]: %.3f" % (t, train_losses[-1], train_accuracies[-1]))
  print("[EPOCH]: %i, [VAL LOSS]: %.6f, [VAL ACCURACY]: %.3f \n" % (t, val_losses[-1] ,val_accuracies[-1]))
########### evaluate the test accuracy with test data ####################
    
  model.train()

plt.figure()
plt.plot(train_accuracies,'r*-', label = 'train accuracy')
plt.plot(val_accuracies,'bo-', label = 'validation accuracy')
plt.xlabel('epoch', fontsize = 16)
plt.ylabel('accuracy', fontsize = 16)
plt.xticks(np.arange(n_epoch, step=4), fontsize = 14)
plt.yticks (fontsize = 14)
plt.grid()
plt.legend(fontsize = 14)

plt.savefig("figures/accuracy_using_phase.png",dpi = 400)
np.savetxt('data/plotting_data/train_accuraciy_conv_lstm.txt', train_accuracies)
np.savetxt('data/plotting_data/val_accuraciy_conv_lstm.txt', val_accuracies)
