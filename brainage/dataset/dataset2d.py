from pathlib import Path
import logging
import collections
import h5py
import dotenv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as AbstractDataset
dotenv.load_dotenv()
from tqdm import tqdm


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
                try:
                    group_str, key = key.split('/')
                    orientation = group_str
                    group_str += '/'
                    label = info_df.loc[key][column]
                    sex = info_df.loc[key]['sex']
                except:
                    self.logger.warning(f'Key {key} not found in info dataframe. Skipping.')
                    continue

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
                          'sex': sex,
                          'orientation': orientation}
                yield sample

        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]
        sample = {'data': np.transpose(ds['data'][:][np.newaxis, ...].astype(np.float32), (0, -1, 1, 2)),
                  'label': ds['label'],
                  'key': ds['key'],
                  'orientation': ds['orientation']}
        
        # data augmentation
        # data tensor format B x C X H X W
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)
        
        if self.meta:
            sample['position'] = ds['sex']
        return sample