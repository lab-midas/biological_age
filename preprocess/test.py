import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import tqdm
import numpy as np
import glob
import h5py
from pathlib import Path


input_dir = Path('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/processed/seg_ori')
output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/abdominal_MRI/seg')
seg_files = os.listdir('/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/processed/seg_ori')

for seg_file in seg_files:
    if input_dir.joinpath(seg_file).is_dir() and not output_dir.joinpath(seg_file).exists():
        output_dir.joinpath(seg_file).symlink_to(input_dir.joinpath(seg_file), target_is_directory=True)


    print('done')
#output_dir = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/'
csv_input = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv'
csv_input_2 = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/ukb675384.csv'
#csv_output = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_all.csv'


#df = pd.read_csv(csv_input, usecols=['eid', '21003-2.0', '31-0.0', '21002-0.0', '50-0.0'])
df_1 = pd.read_csv(csv_input, usecols=['eid', '21003-2.0', '21003-1.0', '21003-0.0', '21022-0.0'])
df_2 = pd.read_csv(csv_input_2, usecols=['eid', '20201-2.0', '20201-3.0'])
df = pd.merge(df_2, df_1, how='inner', on='eid')
#df = df.rename(columns={'eid': 'key', '21003-2.0': 'age', '31-0.0': 'sex', '21002-0.0': 'weight', '50-0.0': 'height'})
#df.to_csv(csv_output)
print('done')
data = pd.read_csv('/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_all.csv')
#data = pd.read_csv('/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv')
#data.to_feather('/home/raecker1/nako_ukb_age/ukb677731.feather')
print('done')


def copy_files(file_list, source_dir, destination_dir):
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name + '_2_sa.nii.gz')
        shutil.copy(source_path, destination_dir)
        print(f"Copied {file_name} to {destination_dir}")

def copy_directories(directory_list, source_dir, destination_dir):
    for directory_name in directory_list:
        source_path = os.path.join(source_dir, directory_name + '_2')
        destination_path = os.path.join(destination_dir, directory_name + '_2')
        shutil.copytree(source_path, destination_path)
        print(f"Copied directory {directory_name} to {destination_dir}")

if __name__ == '__main__':
    
    files = ['1114380', '1218879', '1220336', '1304612', '1312175', '1327808', '1556503', 
             '1603018', '1650653', '1687423', '1688572', '1749298', '1778633', '1851970', 
             '1916887', '2113661', '2124386', '2164909', '2198789', '2241745']
    data_path = '/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_heart_preprocessed/'

    for file in files:
        fhandle = h5py.File(os.path.join(data_path, file + '_2_sa.h5'), 'r')
        data = fhandle['image']
        data = data[file +'_2_sa']
        data = np.array(data)
        plt.imshow(data[:,:,4,20], cmap='gray')
        plt.savefig('dummy.png')
        print('done')

    def load_data():
        for key in fhandle.keys():
            group_str = 'name' + '/' 

            for idx in [2, 3]:
                keyh5 = key + '_' + str(idx)
                #if Path(self.datapath).joinpath(keyh5 + '_sa.h5').exists():
                if f'{group_str}{keyh5}' in fhandle:
                    break
                else:
                    keyh5 = ''

            #fhandle = h5py.File(self.datapath.joinpath(keyh5 + '_sa.h5'), 'r')

            data = fhandle[f'{group_str}{keyh5}']
            data = np.squeeze(data[:, :, np.random.randint(np.shape(data)[2], size=1), :])  # take a random slice -> data: X x Y x Time
            sample = {'data': data,
                    'orientation': None}
            return sample
    
    """file_list = ['1066822','1129602','1195996','1244324','1263418','1331107','1482798','1566033',
                 '1568698','1629373','1723876','1815812','1891839','1912173','2111994','2167470','2214428','2234488']
    source_dir = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/sa_heart/raw'
    destination_dir = '/home/raecker1/test/'
    copy_files(file_list, source_dir, destination_dir)
    source_dir = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/sa_heart/processed/seg/'
    copy_directories(file_list, source_dir, destination_dir)
    

    input_dir = '/mnt/qdata/share/rakuest1/data/UKB/raw/la_heart/raw/'
    nifti_files = [f for f in glob.glob(input_dir + '*.nii.gz')]
    a = list(nifti_files)
    for nifti_file in a:
        img = nib.load(nifti_file)
        img_data = img.get_fdata().astype(np.float32)
        print('done')
        b = img_data[..., 0, 25]
        plt.imshow(b, cmap='gray')
        plt.show(block=True)
    """