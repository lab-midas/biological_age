import os
from sys import getsizeof
from pathlib import Path

import logging
import collections
#import torch
import time
import h5py
import dotenv
import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as AbstractDataset
dotenv.load_dotenv()
from tqdm import tqdm

class SliceDataset(AbstractDataset):

    def __init__(self,
                 data,
                 info,
                 labels=['age'],
                 image_group='image',
                 preload=True,
                 zoom=None,
                 transform=None):
        """Dataset of slices drawn from a numpy array. 

        The dataframe object should contain the following columns (index=slice number,
        subject key, labels ('age','sex'))
        Args:
            data (np.array): Data array (can be memmapped), number of slices x H x W
            info (pandas.DataFrame): Data frame containing the meta information. One row per slice.
            labels (list): Label columns names.
            image_group (str): Group name of the image datasets. Defaults to 'image'.
            preload (bool): Preload dataset to memory. Defaults to True.
            zoom (float): Zoom image. Defaults to None. 
            transform (class, optional): Tranformation per Image. Defaults to None.
        """

        super().__init__()

        # copy over
        self.info = info
        self.data = data
        self.labels = labels
        self.transform = transform
        self.preload = preload
        self.zoom = zoom

        hf = h5py.File(self.data, mode='r')
        if self.preload:
            t0 = time.perf_counter()
            print('loading data to memory ...')
            self.ds = hf['image'][self.info.index][:]
            print(f'finished {self.ds.nbytes/1e6:.2f} MB - {time.perf_counter() - t0:.2f}s ')
        else:
            self.ds = hf['image']

    def __len__(self):
        return len(self.info)   

    def __getitem__(self, i):
        # subject
        key = self.info.iloc[i]['key']
        sl = self.info.iloc[i]['position']
        pos = self.info.iloc[i].name
        img = self.ds[i] if self.preload else self.ds[pos]
        img = img.astype(np.float32)
        if self.zoom:
            img = scipy.ndimage.zoom(img, self.zoom)

        sample = {'data':  img[np.newaxis, np.newaxis, :, :],
                  'label': self.info.iloc[i][self.labels].tolist(),
                  'position': sl,
                  'key':   key}

        # data augmentation
        # data tensor format BxCxHxWxD (B=C=1)
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)

        return sample


class FundusDataset(AbstractDataset):
    def __init__(self,
                 data,
                 keys,
                 info,
                 group,
                 column='label',
                 preload=False,
                 meta=False,
                 ukb=True,
                 transform=None):

        super().__init__()

        # copy over
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self.preload = preload
        self.meta = meta

        self.logger.info('opening dataset ...')
        if ukb:
            info_df = pd.read_csv(info, index_col=0, usecols=[1,2,3,4,5], dtype={'key': 'string', column: np.float32})
        else:
            info_df = pd.read_csv(info, index_col=0, dtype={'key': 'string', column: np.float32})
        self.keys = [l.strip() for l in Path(keys).open().readlines()] if isinstance(keys, str) else keys

        fhandle = h5py.File(data, 'r')

        def load_data():
            for key in tqdm(self.keys):
                group_str, key = key.split('/')
                group_str += '/'
                label = info_df.loc[key][column]
                sex = info_df.loc[key]['sex']

                #coin = np.random.randint(2, size=1)
                #if coin:
                #    group_str = 'left/'
                #else:
                #    group_str = 'right/'

                for idx in range(4):
                    keyh5 = key + '_' + str(idx)
                    if f'{group_str}{keyh5}' in fhandle:
                        break
                    else:
                        keyh5 = ''

                if self.preload:
                    data = fhandle[f'{group_str}{keyh5}'][:]  # x, y, RGB
                else:
                    data = fhandle[f'{group_str}{keyh5}']

                sample = {'data': data,
                          'label': label,
                          'key': key,
                          'sex': sex}
                yield sample

        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]
        sample = {'data': np.transpose(ds['data'][:][np.newaxis, ...].astype(np.float32), (0, -1, 1, 2)),
                  'label': ds['label'],
                  'key': ds['key']}
        # data augmentation
        # data tensor format B x C X H X W (B=1, C=3)
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)
        if self.meta:
            sample['position'] = ds['sex']
        return sample
