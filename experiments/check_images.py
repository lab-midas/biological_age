import matplotlib.pyplot as plt
import os
import tqdm
import h5py
import pandas as pd
import numpy as np
from pathlib import Path


def load_brain_data(data, key, group, ukb=True):
    fhandle = h5py.File(data, 'r')
    group_str = group + '/' if group else ''
    try:
        if ukb:
            keyh5 = key + '_2'
        else:
            keyh5 = key
        data = fhandle[f'{group_str}{keyh5}']
    except:
        if ukb:
            keyh5 = key + '_3'
        else:
            keyh5 = key
        data = fhandle[f'{group_str}{keyh5}']
    plt.imsave(f'/home/raecker1/test/{key}_brain.png', data[:,:,data.shape[2]//2], cmap='gray')


def load_abdominal_data(data, key, group, contrast, organ, ukb=True):
    if not os.path.isdir(data):
        fhandle = h5py.File(data, 'r')

        if '/' in key:  # combined kidney set
            keyh5 = key
            key = key.replace('/', '_')
        else:
            keyh5 = key
        group_str = group + '/' if group else ''
        group_str_inner = group_str + contrast + '/'
        data = fhandle[f'{group_str_inner}{keyh5}']
    else:
        fhandle = h5py.File(os.path.join(data, f'{contrast}_{key}.h5'), 'r')
        data = fhandle['image']
    plt.imsave(f'/home/raecker1/test/{key}_{organ}_{contrast}.png', data[:, :, data.shape[2] // 2], cmap='gray')


def load_heart_data(data, key, group, ukb=True):
    print(f'load {key}')
    fhandle = h5py.File(data, 'r')
    group_str = group + '/' if group else ''

    for idx in [2, 3]:
        keyh5 = key + '_' + str(idx) + '_sa'
        # if Path(self.datapath).joinpath(keyh5 + '_sa.h5').exists():
        if f'{group_str}{keyh5}' in fhandle:
            break
        else:
            keyh5 = key
    data = fhandle[f'{group_str}{keyh5}']
    print(data.shape)
    data = np.squeeze(data[:, :, data.shape[2]//2, :])  # take a random slice -> data: X x Y x Time
    print(data.shape)
    plt.imsave(f'/home/raecker1/test/{key}_heart.png', data[:, :, 0], cmap='gray')


def load_fundus_data(data, key):
    fhandle = h5py.File(data, 'r')
    group_str, key = key.split('/')
    orientation = group_str
    group_str += '/'

    # coin = np.random.randint(2, size=1)
    # if coin:
    #    group_str = 'left/'
    # else:
    #    group_str = 'right/'

    for idx in range(4):
        keyh5 = key + '_' + str(idx)
        if f'{group_str}{keyh5}' in fhandle:
            break
        else:
            keyh5 = ''
    data = fhandle[f'{group_str}{keyh5}']
    plt.imsave(f'/home/raecker1/test/{key}_{group_str[:-1]}_fundus.png', data)

if __name__ == '__main__':
    organ = 'brain'
    keys = ['1000295', '1002035', '1002551', '1008408', '1008801', '5143978', '4964011']
    #key = '4675538'
    """keys = ['2837023', '3753282', '4803283', '4663445', '2693985', '1839110', '4525038', 
            '4451253', '5923528', '2693186', '3987000', '5857948', '1421691', '1608783', 
            '4617426', '3632322', '1255391', '1835179', '2590183', '1468308', '1710978', '1544436']"""
    for key in keys:
        if organ == 'brain':
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_brain_preprocessed.h5'
            load_brain_data(data, key, group='image', ukb=True)
        elif organ == 'heart':
            #data = '/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_heart_preprocessed.h5'
            #load_heart_data(data, key, group='image', ukb=True)
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_heart_preprocessed.h5'
            load_heart_data(data, key, group='image', ukb=True)
        elif organ == 'liver':
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_liv_preprocessed_masked.h5'
            load_abdominal_data(data, key, organ='liver', contrast='wat', group='', ukb=True)
        elif organ == 'spleen':
            #data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_spl_preprocessed_masked.h5'
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_spl_preprocessed_masked'
            load_abdominal_data(data, key, organ='spleen', contrast='wat', group='', ukb=True)
        elif organ == 'pancreas':
            #data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_pnc_preprocessed_masked.h5'
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_pnc_preprocessed_masked'
            load_abdominal_data(data, key, organ='pancreas', contrast='wat', group='', ukb=True)
        elif organ == 'kidneys':
            """data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_kidney_preprocessed_masked.h5'
            load_abdominal_data(data, f'{key}/left', organ='kidney', contrast='wat', group='', ukb=True)
            load_abdominal_data(data, f'{key}/right', organ='kidney', contrast='wat', group='', ukb=True)"""
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_lkd_preprocessed_masked'
            load_abdominal_data(data, key, organ='lkd', contrast='wat', group='', ukb=True)
            data = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_rkd_preprocessed_masked'
            load_abdominal_data(data, key, organ='rkd', contrast='wat', group='', ukb=True)
        elif organ == 'fundus':
            data = '/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_fundus_preprocessed.h5'
            load_fundus_data(data, f'left/{key}')
            load_fundus_data(data, f'right/{key}')


