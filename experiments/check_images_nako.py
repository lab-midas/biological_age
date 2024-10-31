import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import nibabel as nib

def load_brain_data(data, key, group, ukb=False):
    fhandle = h5py.File(data, 'r')
    group_str = group + '/' if group else ''
    keyh5 = key
    data = fhandle[f'{group_str}{keyh5}'][:]
    if np.isnan(data).any():
        print('---------------------------------')
        print(f'{key}: robex or fcm normalization failed')
        print('---------------------------------')

    #plt.imsave(f'/home/raeckev1/{key}_brain.png', data[:,data.shape[1]//2,:], cmap='gray')


def load_abdominal_data(data, key, group, contrast, organ, ukb=True):
    fhandle = h5py.File(data, 'r')
    if '/' in key:  # combined kidney set
        keyh5 = key
        key = key.replace('/', '_')
    else:
        keyh5 = key
    group_str = group + '/' if group else ''
    group_str_inner = group_str + contrast + '/'
    data = fhandle[f'{group_str_inner}{keyh5}']

    mask_path = Path('/mnt/qdata/share/raeckev1/nako_30k/abdominal_MRI/seg/')
    mask = nib.load(mask_path.joinpath(key, 'prd.nii.gz'))
    bounding_boxes = pd.read_csv(Path(mask_path).joinpath('bounding_boxes_abdomen.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]
    sel_shape = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50], 'pnc': [80, 50, 50]}
    contrasts = ['fat', 'in', 'opp', 'water']
    box = bounding_boxes.loc[bounding_boxes['pat'] == int(pat)][class_name].values
    box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
    center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
    center = [int(x) for x in center]
    mask_crop = crop(img_data, sel_shape_curr, center)

    plt.imsave(f'/home/raeckev1/{key}_{organ}_{contrast}.png', data[:, :, data.shape[2] // 2], cmap='gray')


def load_heart_data(data, key, group, ukb=True):
    #print(f'load {key}')
    fhandle = h5py.File(data, 'r')
    group_str = group + '/' if group else ''

    keyh5 = key
    data = fhandle[f'{group_str}{keyh5}']
    #print(data.shape)
    data = np.squeeze(data[:, :, np.random.randint(np.shape(data)[2], size=1), :])  # take a random slice -> data: X x Y x Time
    if data.shape[-1] != 25:
        print('---------------------------------')
        print(f'{key} != 25 cardiac time frames')
        print(data.shape)
        print('---------------------------------')
    plt.imsave(f'/home/raeckev1/{key}_heart.png', data[:, :, 0], cmap='gray')


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
    organ = 'heart'
    key_path = f'/mnt/qdata/share/raeckev1/nako_30k/interim/keys/train_{organ}_mainly_healthy.dat'
    keys = [l.strip() for l in Path(key_path).open().readlines()]
    #keys = ['114584', '107033', '119628', '111409', '126766', '127219', '115750']
    for key in tqdm(keys):
        if organ == 'brain':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_brain_preprocessed.h5'
            load_brain_data(data, key, group='image', ukb=True)
        elif organ == 'heart':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_heart_preprocessed.h5'
            load_heart_data(data, key, group='image', ukb=True)
        elif organ == 'liver':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_liv_preprocessed.h5'
            load_abdominal_data(data, key, organ='liver', contrast='wat', group='', ukb=True)
        elif organ == 'spleen':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_spl_preprocessed.h5'
            load_abdominal_data(data, key, organ='spleen', contrast='wat', group='', ukb=True)
        elif organ == 'pancreas':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_pnc_preprocessed.h5'
            load_abdominal_data(data, key, organ='pancreas', contrast='wat', group='', ukb=True)
        elif organ == 'kidneys':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_kidney_preprocessed.h5'
            load_abdominal_data(data, f'{key}/left', organ='kidney', contrast='wat', group='', ukb=True)
            load_abdominal_data(data, f'{key}/right', organ='kidney', contrast='wat', group='', ukb=True)
        elif organ == 'fundus':
            data = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_fundus_preprocessed.h5'
            load_fundus_data(data, f'left/{key}')
            load_fundus_data(data, f'right/{key}')


