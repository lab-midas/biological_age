import logging
import collections
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset as AbstractDataset


class BrainDataset(AbstractDataset):
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
        self.logger.info(info)
        if ukb:
            info_df = pd.read_csv(info, index_col=0, usecols=[1,2,3,4,5], dtype={'key': 'string', column: np.float32})
        else:
            info_df = pd.read_csv(info, index_col='key', dtype={'key': 'string', column: np.float32})
        self.keys = [l.strip() for l in Path(keys).open().readlines()] if isinstance(keys, str) else keys

        self.logger.info('loading h5 dataset ...')
        self.logger.info(data)
        fhandle = h5py.File(data, 'r')

        def load_data():
            for key in tqdm(self.keys):
                label = info_df.loc[key][column]
                sex = info_df.loc[key]['sex']
                group_str = group + '/' if group else ''

                try:
                    if ukb:
                        keyh5 = key + '_2'
                    else:
                        keyh5 = key

                    if self.preload:
                        data = fhandle[f'{group_str}{keyh5}'][:]
                    else:
                        data = fhandle[f'{group_str}{keyh5}']
                except:
                    if ukb:
                        keyh5 = key + '_3'
                    else:
                        keyh5 = key

                    if self.preload:
                        data = fhandle[f'{group_str}{keyh5}'][:]
                    else:
                        data = fhandle[f'{group_str}{keyh5}']

                sample = {'data': data,
                          'label': label,
                          'key': key,
                          'sex': sex,
                          'orientation': ''}
                yield sample

        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]
        sample = {'data':   ds['data'][:][np.newaxis, np.newaxis, ...].astype(np.float32),
                  'label':  ds['label'],
                  'key':    ds['key'],
                  'orientation': ds['orientation']}
        
        # data augmentation
        # data tensor format B x C X H X W X D
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)

        if self.meta:
            sample['position'] = ds['sex']
        return sample


class HeartDataset(AbstractDataset):
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
        self.logger.info(info)

        if ukb:
            info_df = pd.read_csv(info, index_col=0, usecols=[1,2,3,4,5], dtype={'key': 'string', column: np.float32})
        else:
            info_df = pd.read_csv(info, index_col='key', dtype={'key': 'string', column: np.float32})

        self.keys = [l.strip() for l in Path(keys).open().readlines()] if isinstance(keys, str) else keys

        self.logger.info('loading h5 dataset ...')
        fhandle = h5py.File(data, 'r')

        self.logger.info(data)

        def load_data():
            for key in tqdm(self.keys):
                label = info_df.loc[key][column]
                sex = info_df.loc[key]['sex']
                group_str = group + '/' if group else ''

                if ukb:
                    for idx in [2, 3]:
                        keyh5 = key + '_' + str(idx) + '_sa'
                        if f'{group_str}{keyh5}' in fhandle:
                            break
                        else:
                            keyh5 = ''
                else:
                    keyh5 = key

                if self.preload:
                    data = fhandle[f'{group_str}{keyh5}'][:]  # keyh5_sa for multiple h5 files
                else:
                    data = fhandle[f'{group_str}{keyh5}']

                data = np.squeeze(data[:, :, np.random.randint(np.shape(data)[2], size=1), :])  # take a random slice -> data: X x Y x Time
                sample = {'data': data,
                          'label': label,
                          'key': key,
                          'sex': sex,
                          'orientation': ''}
                yield sample

        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]
        sample = {'data':   ds['data'][:][np.newaxis, np.newaxis, ...].astype(np.float32),
                  'label':  ds['label'],
                  'key':    ds['key'],
                  'orientation': ds['orientation']}
        
        # data augmentation
        # data tensor format B x C X H X W X D
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)

        if self.meta:
            sample['position'] = ds['sex']
        return sample


class AbdomenDataset(AbstractDataset):
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
        self.logger.info(info)
        if ukb:
            info_df = pd.read_csv(info, index_col=0, usecols=[1,2,3,4,5], dtype={'key': 'string', column: np.float32})
        else:
            info_df = pd.read_csv(info, index_col='key', dtype={'key': 'string', column: np.float32})

        self.keys = [l.strip() for l in Path(keys).open().readlines()] if isinstance(keys, str) else keys

        if ukb:
            self.contrasts = ['fat', 'inp', 'opp', 'wat']
        else:
            self.contrasts = ['fat', 'in', 'opp', 'water']

        self.logger.info('loading h5 dataset ...')
        self.filehandle = data

        if self.preload:
            fhandle = h5py.File(data, 'r')

        self.logger.info(data)

        def load_data():
            for key in tqdm(self.keys):
                if '/' in key:  # combined kidney set
                    keyh5 = key
                    key = key.split('/')[0]

                    if ukb:
                        orientation = keyh5.split('/')[1]
                    else:
                        orientation =  '/' + keyh5.split('/')[1]
                else:
                    keyh5 = key
                    orientation = ''

                try:
                    label = info_df.loc[key][column]
                    sex = info_df.loc[key]['sex']
                except:
                    print(key)
                    continue

                group_str = group + '/' if group else ''
                data = []

                for contrast in self.contrasts:
                    group_str_inner = group_str + contrast + '/'

                    if ukb:
                        if self.preload:
                            data.append(fhandle[f'{group_str_inner}{keyh5}'][:])
                        else:
                            data.append(f'{group_str_inner}{keyh5}')
                    else:
                        try:
                            if self.preload:
                                data.append(fhandle[f'{group_str_inner}{keyh5}'][:])
                            else:
                                data.append(f'{group_str_inner}{keyh5}')
                        except:
                            keyh5 = key + '_30' + orientation
                            #print(keyh5)
                            if self.preload:
                                data.append(fhandle[f'{group_str_inner}{keyh5}'][:])
                            else:
                                data.append(f'{group_str_inner}{keyh5}')

                if not ukb:
                    orientation = orientation[1:] if orientation != '' else orientation

                sample = {'data': data,
                          'label': label,
                          'key': key,
                          'sex': sex,
                          'orientation': orientation}
                yield sample

        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]

        if self.preload:
            datasample = np.stack(ds['data'], axis=0)[np.newaxis, ...].astype(np.float32)
        else:
            datasample = []
            for contrast in self.contrasts:
                datasample.append(h5py.File(self.filehandle, 'r')[ds['data'][self.contrasts.index(contrast)]])
            datasample = np.stack(datasample, axis=0)[np.newaxis, ...].astype(np.float32)

        sample = {'data':   datasample,
                  'label':  ds['label'],
                  'key':    ds['key'],
                  'orientation': ds['orientation']}
        
        # data augmentation
        # data tensor format B x C X H X W X D
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)
        
        if self.meta:
            sample['position'] = ds['sex']
        return sample