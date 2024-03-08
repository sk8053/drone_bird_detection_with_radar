import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import pickle
import os
from tqdm import tqdm

#mode = 'rotation' # 'static'
mode = 'static'
angle_list = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]
#angle_list = [112.5]
for angle in tqdm(angle_list):

    file_name1 = f'bird_{angle}_50pct'
    file_name2 = f'bird_{angle}_still'

    with open(f'data/{file_name1}.pkl', 'rb') as f:
        data_rot = pickle.load(f)
    with open(f'data/{file_name2}.pkl', 'rb') as f:
        data_sta = pickle.load(f)

    if mode == 'rotation':
        data = data_rot
        save_file_name = f'moving_{angle}.txt'
    else:
        data = data_sta
        save_file_name = f'static_{angle}.txt'

    max_v = np.max(abs(data[100]))
    I = np.where(max_v == abs(data[100]))
    I = (I[0] - 1, I[1] +1)

    plt.figure()
    plt.plot(np.angle(data[:,I[0], I[1]],  deg = True))
    plt.grid()
    plt.title('phase change of bird')

    #plt.figure()
    #plt.imshow(abs(data[10]))
    #plt.scatter([I[1]], [I[0]], c = 'r')
    plt.savefig(f'data/sample_data/phase change of bird_{save_file_name}.png')
    np.savetxt(f'data/sample_data/bird_angle_{save_file_name}', np.angle(data[:, I[0], I[1]], deg = True))


    file_name1 = f'mavik_{angle}_rotation'
    file_name2 = f'mavik_{angle}_still'
    with open(f'data/{file_name1}.pkl', 'rb') as f:
        data_rot = pickle.load(f)
    with open(f'data/{file_name2}.pkl', 'rb') as f:
        data_sta = pickle.load(f)

    if mode == 'rotation':
        data = data_rot
        save_file_name = f'moving_{angle}.txt'
    else:
        data = data_sta
        save_file_name = f'static_{angle}.txt'

    max_v = np.max(abs(data[0]))
    I = np.where(max_v == abs(data[0]))
    I = (I[0]-1, I[1]+1)

    plt.figure()
    plt.plot(np.angle(data[:, I[0], I[1]], deg = True))
    plt.grid()
    plt.title('phase change of drone')

    #plt.figure()
    #plt.imshow(abs(data[0]))
    #plt.scatter([I[1]], [I[0]], c='r')

    plt.savefig(f'data/sample_data/phase change of mavik_{save_file_name}.png')
    np.savetxt(f'data/sample_data/mavik_angle_{save_file_name}',np.angle(data[:, I[0], I[1]], deg=True))
    #plt.show()
    #exit()
