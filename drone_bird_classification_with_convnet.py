# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:16:38 2023

@author: seongjoon kang
"""
import torch 
import matplotlib.pyplot as plt

import random
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
seed = 12345
random.seed(seed)
torch.manual_seed(seed)
import numpy as np
import pickle
from classification_models import MyDataSet, Res_Convnet
from utils import conf, DEVICE
from utils.download_data import download_data
import os
if (os.path.isfile('data/total_data/bird_data_total.pkl') and  os.path.isfile('data/total_data/mavik_data_total.pkl')) is False:
  download_data()

n_epoch = conf.n_epoch_conv
lr = conf.lr_conv
batch_size = conf.batch_size_conv
### attenuation factor ranging from [a1, a2]
a2 = conf.a2
a1 = conf.a1
################
print(f'learning rate: {lr}')
print(f'a1 and a2: {a1},{a2}')
device = DEVICE
print(f'================ device is {device} =====================')

with open('data/total_data/bird_data_total.pkl', 'rb') as f:
  bird_data_total = pickle.load(f)[:,None]
with open('data/total_data/mavik_data_total.pkl', 'rb') as f:
  mavik_data_total = pickle.load(f)[:,None]


bird_data_total = np.abs(bird_data_total)
mavik_data_total = np.abs(mavik_data_total)
print (bird_data_total.shape, mavik_data_total.shape)

# set the range of data as [0,1]
mavik_data_total /= np.max(mavik_data_total)
bird_data_total /= np.max(bird_data_total)
# create the labels of mavik (1) and bird (0)
bird_label = np.repeat([0], len(bird_data_total))
mavik_lavel = np.repeat([1], len(mavik_data_total))

total_data = np.vstack((bird_data_total, mavik_data_total))
# let's permute bird and drone data
I = np.random.permutation(np.arange(len(total_data)))
total_data = total_data[I]
total_label = np.append(bird_label, mavik_lavel)
total_label = total_label[I]

plt.figure()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(np.squeeze(bird_data_total[1]))
ax1.set_title('bird')
ax2.imshow(np.squeeze(mavik_data_total[2]))
ax2.set_title('drone')
plt.savefig('figures/drone_and_bird_amplitude.png')
plt.close()


total_data = torch.tensor(total_data, dtype=torch.float32)

# split data by train,validation, and test
train_size = int(0.7*len(total_data)) 

L = len(total_data)
train_data_set = MyDataSet(image = total_data[:train_size], target= total_label[:train_size])
val_data_set = MyDataSet(image = total_data[train_size:L], target= total_label[train_size:L])

train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True,drop_last=False )
val_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=True,drop_last=False )

####################################################################################################
####################################################################################################

model = Res_Convnet(input_channels=1, num_classes=2)
print("Num params: ", sum(p.numel() for p in model.parameters()))
model.to (device)
# Initialize the criterion
criterion = torch.nn.CrossEntropyLoss()
# Initialize the SGD optimizer with lr 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #torch.optim.RMSprop(model.parameters(), lr=lr)
max_patience = 100

###### model train and test ##################

best_val = 0.0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training
for t in tqdm(range(n_epoch)):
  epoch_t_acc = 0.0 
  epoch_t_loss = 0.0
  # Set the model to train mode
  #model.to(device)       
  model.train()
  # Loop over the training set
  for train_data, targets in train_data_loader:
    # Put the inputs and targets on the write device
    
    # signals attenuation is considered by multiplying a random number from Uni[a1, a2]
    f = torch.rand(1)*a2 + a1
    train_data = f*train_data
        
    train_data = train_data.to(device)
    targets = targets.to (device)
    targets = torch.tensor(targets, dtype = torch.int64)
  
    # Feed forward to get the logits
    y_pred = model.forward(train_data)
   
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
  v_acc = 0
  v_loss = 0  
  with torch.no_grad():
      # TLoop over the validation set 
      for val_data, targets in val_data_loader:
          # Put the inputs and targets on the write device
          #signals attenuation is considered by multiplying a random number from Uni[0.2, 1]
          f = torch.rand(1)*a2 +a1
          val_data = val_data*f
          val_data = val_data.to(device)
          targets = targets.to(device)
          targets = torch.tensor(targets, dtype = torch.int64)
          # Feed forward to get the logits
          y_pred_val = model(val_data)
          score, predicted = torch.max(y_pred_val, 1)
          # Compute the loss and accuracy
          loss_val = criterion ( y_pred_val,targets)
          accuracy_val = (targets == predicted).sum().float() / len(targets)
          
          # Keep track of accuracy and loss
          v_acc += accuracy_val.item()
          v_loss += loss_val.item()

  val_losses.append(v_loss/len(val_data_loader))
  val_accuracies.append(v_acc/len(val_data_loader))

  if val_accuracies[-1] > best_val:
    best_val = val_accuracies[-1]
    patience_counter = 0
    print(f'-------------- model is saved -------------{np.round(best_val,2)}')
    # Save best model, optimizer, epoch_number
    torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': t,
            }, 'checkpoint/checkpoints_res_conv.pth')
  else:
    patience_counter += 1    
    if patience_counter > max_patience: 
      break
  print("[EPOCH]: %i, [TRAIN LOSS]: %.6f, [TRAIN ACCURACY]: %.3f" % (t, train_losses[-1], train_accuracies[-1]))
  print("[EPOCH]: %i, [VAL LOSS]: %.6f, [VAL ACCURACY]: %.3f \n" % (t, val_losses[-1] ,val_accuracies[-1]))
    
  model.train()

   
# Run the training loop using this model
#train_losses, train_acc, val_losses, val_acc = train_loop(model, criterion, optimizer,  train_data_loader, test_data_loader,n_epoch)
plt.figure()
plt.plot(train_accuracies,'r*-', label = 'train accuracy')
plt.plot(val_accuracies,'bo-', label = 'validation accuracy')
plt.xlabel('epoch', fontsize = 16)
plt.ylabel('accuracy', fontsize = 16)
plt.xticks(np.arange(n_epoch, step=4), fontsize = 14)
plt.yticks (fontsize = 14)
plt.grid()
plt.legend(fontsize = 14)
plt.savefig("figures/accuracy_using_amplitude.png",dpi = 400)

np.savetxt('data/plotting_data/train_accuraciy_conv.txt', train_accuracies)
np.savetxt('data/plotting_data/val_accuraciy_conv.txt', val_accuracies)


'''
    if 0:
        ## draw confusion matrix over all validation data ###
        val_data = torch.tensor(test_data_image, dtype = torch.float32)
        targets = torch.tensor(test_data_targets, dtype = torch.int64)
    
        
        val_data = val_data*(torch.rand(val_data.shape)*0.8+0.2)
        val_data, targets = val_data.to(device), targets.to(device)
        y_pred_val = model(val_data)
        _, y_pred = torch.max(y_pred_val, 1)
        
        #f = torch.rand(1)*0.8 +0.2
        cm = confusion_matrix(y_pred.detach().cpu().numpy(), 
                              targets.detach().cpu().numpy(), 
                              labels=[0,1])
        
        class_labels = ['bird', 'drone']
        cm = cm/cm.sum(0)[None]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_labels)
        disp.plot(cmap='binary', colorbar = False)
        
              
        plt.xticks(range(len(class_labels)), class_labels, fontsize=14)
        plt.yticks(range(len(class_labels)), class_labels, fontsize=14, rotation =90)
    
        plt.xlabel('Precticted label', fontsize = 15)
        plt.ylabel('True label', fontsize = 15, rotation = 90)
        plt.show()
    '''