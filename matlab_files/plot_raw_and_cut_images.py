# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:07:56 2021

@author: seongjoon kang
"""

import pickle
import matplotlib.pyplot as plt
plt.rcParams["font.serif"] = "Times New Roman"
import numpy as np
import glob
from tqdm import tqdm
import os

file_list = glob.glob('data/raw_data/*.pkl')
#mavik_data = np.empty(shape=[0, 32, 32])
#bird_data = np.empty(shape=[0, 32, 32])

mavik_data = np.empty(shape=[0, 20, 30])
bird_data = np.empty(shape=[0, 20, 30])

dir_ =  'data/raw_data/'

################################### Bionic Bird Image #########################################

file ='bird_tripod_az_90_tilt_-45_50pct.pkl' #file.split('data/raw_data/')[1].split('.')[0]

file_dir = os.path.join(dir_, file)
with open(file_dir, 'rb') as f:
    data = pickle.load(f)

data_ = np.flip(data[500].T, axis =0)
plt.figure()
plt.imshow(abs(data_), cmap = 'jet')
y, x = data_.shape
print(y,x)
plt.xticks(np.arange(x, step=21.211), np.arange(-90, 120,step = 30) , fontsize = 16, fontname = 'Times New Roman')
plt.yticks(168-np.array([8.0,   28,  48,  68,  88,  108, 128, 148, 168]),
           np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), fontsize = 16, fontname = 'Times New Roman')

#plt.xlabel(r'Angle [$^\circ$]', fontsize = 16, fontname = 'Times New Roman')
#plt.ylabel('Distance [m]', fontsize = 16, fontname = 'Times New Roman')
#plt.title('Bionic bird', fontsize = 17,fontweight = 'bold', fontname = 'Times New Roman')
plt.savefig(f'data/radar_images/new/origin_{file}.png',  bbox_inches="tight", dpi = 800)
plt.close()
###############################################################
########## cut-image ##################
data = data[:, 60:80, :][:, :, 60:90]
data_ = np.flip(data[500].T, axis=0)

plt.figure()
plt.imshow(np.abs(data_), cmap= 'jet')
plt.tight_layout()
plt.yticks([])
plt.xticks([])
plt.savefig(f'data/radar_images/new/cut_{file}.png', bbox_inches="tight", dpi =800)
plt.close()


######################################### Mavik Image ######################################

file ='mavik_tripod_az_90_tilt_-45_rotation.pkl' #file.split('data/raw_data/')[1].split('.')[0]

file_dir = os.path.join(dir_, file)
with open(file_dir, 'rb') as f:
    data = pickle.load(f)

data_ = np.flip(data[500].T, axis =0)
plt.figure()
plt.imshow(abs(data_), cmap = 'jet')
y, x = data_.shape

plt.xticks(np.arange(x, step=21.211), np.arange(-90, 120,step = 30) , fontsize = 16, fontname = 'Times New Roman')
plt.yticks(168-np.array([8.0,   28,  48,  68,  88,  108, 128, 148, 168]),
           np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), fontsize = 16, fontname = 'Times New Roman')

#plt.xlabel(r'Angle [$^\circ$]', fontsize = 16, fontname = 'Times New Roman')
#plt.ylabel('Distance [m]', fontsize = 16, fontname = 'Times New Roman')
#plt.title('DJI mavik', fontsize = 17,fontweight = 'bold', fontname = 'Times New Roman')
plt.savefig(f'data/radar_images/new/origin_{file}.png',  bbox_inches="tight", dpi = 800)


########## cut-image ##################
data = data[:, 60:80, :][:, :, 60:90]
data_ = np.flip(data[500].T, axis=0)

plt.figure()
plt.imshow(np.abs(data_), cmap= 'jet')
plt.tight_layout()
plt.yticks([])
plt.xticks([])
plt.savefig(f'data/radar_images/new/cut_{file}.png', bbox_inches="tight", dpi =800)

plt.show()







