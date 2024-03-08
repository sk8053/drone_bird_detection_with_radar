# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:07:56 2021

@author: seongjoon kang
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm

file_list = glob.glob('data/*.pkl')
#mavik_data = np.empty(shape=[0, 32, 32])
#bird_data = np.empty(shape=[0, 32, 32])

mavik_data = np.empty(shape=[0, 20, 30])
bird_data = np.empty(shape=[0, 20, 30])

for file in tqdm(file_list):

    with open(file, 'rb') as f:
        data = pickle.load(f)

    s = file.find('\\')
    e = file.find('.pkl')
    file = file[s + 1:e]

    #data = data[:, 54:86, :][:, :, 59:91]
    data = data[:, 60:80, :][:, :, 60:90]
    
    plt.figure()
    plt.imshow(np.abs(data[100]))
    plt.savefig(f'data/RCS_plots/{file}.png')
    
    bird = file.find('bird') # read bird
    mavik = file.find('mavik') # read mavik
    p3p = file.find('p3p') # read drone

    if bird != -1:
        with open(f'data/cut_data/cut_{file}.pkl', 'wb') as f:
            pickle.dump(data, f)
        bird_data = np.vstack((bird_data, data))
    elif mavik != -1:
        with open(f'data/cut_data/cut_{file}.pkl' , 'wb') as f:
            pickle.dump(data, f)
        mavik_data = np.vstack((mavik_data, data))
    else:
        with open(f'data/cut_data/cut_{file}.pkl' , 'wb') as f:
            pickle.dump(data, f)
        mavik_data = np.vstack((mavik_data, data))

with open('data/total_cut_data/bird_data_total.pkl', 'wb') as f:
    pickle.dump(bird_data, f)

with open('data/total_cut_data/mavik_data_total.pkl', 'wb') as f:
    pickle.dump(mavik_data, f)



