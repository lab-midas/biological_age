import tempfile
import re
import shutil
import time
import subprocess
import argparse
import tempfile
from zipfile import ZipFile
import SimpleITK as sitk
import nibabel as nib
import os
import h5py
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def write_keys(input_dir, output_dir, output_file, verbose=False):
    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    if os.path.exists(output_dir.joinpath('keys', 'all.dat')):
        print('keys already exists')
        #return pickle.load(open(output_dir.joinpath('keys', 'all.dat'), 'r'))
        return [l.strip() for l in output_dir.joinpath('keys', 'all.dat').open().readlines()]

    # Get list of all nifti files in input_dir
    nifti_files = [f for f in input_dir.glob('*.nii.gz')]

    keys = []
    for nifti_file in tqdm(nifti_files):
        keyh5 = nifti_file.stem.split('.')[0]
        key = keyh5.split('_')[0]
        keys.append(key)

    keys = list(set(keys))  # remove duplicates of "*_2" and "*_3", revisits of patients?
    with open(output_dir.joinpath('keys', 'all.dat'), 'w') as f:  # interim_8000 was written and read as binary: "wb" / "rb" (above)
        #pickle.dump(keys, f)
        for key in keys:
            f.write(key + '\n')

    return keys

def convert_nifti_h5(input_dir, output_dir, output_file, verbose=False):
    """
    Converts all nifti files in input_dir to h5 files in output_dir
    """

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    # Get list of all nifti files in input_dir
    nifti_files = [f for f in input_dir.glob('*.nii.gz')]

    # Create list of all h5 files in output_dir
    h5_file = output_dir.joinpath(output_file)

    if os.path.exists(h5_file):
        print('{} already exists'.format(h5_file))
        #return pickle.load(open(output_dir.joinpath('keys', 'all.dat'), 'r'))
        return [l.strip() for l in output_dir.joinpath('keys', 'all.dat').open().readlines()]

    keys = []
    hf = h5py.File(h5_file, 'w')
    grp_image = hf.create_group('image')
    grp_affine = hf.create_group('affine')
    for nifti_file in tqdm(nifti_files):
        img = nib.load(nifti_file)
        img_data = img.get_fdata().astype(np.float32)
        affine = img.affine.astype(np.float16)
        keyh5 = nifti_file.stem.split('.')[0]

        # Write to h5 file
        grp_image.create_dataset(keyh5, data=img_data)
        grp_affine.create_dataset(keyh5, data=affine)

        key = keyh5.split('_')[0]
        keys.append(key)
        img = None
        img_data = None
        affine = None
    hf.close()

    """
        #with h5py.File(h5_file, 'w') as hf:
            grp_image = hf.create_group('image')
            grp_affine = hf.create_group('affine')
            for nifti_file in tqdm(nifti_files):
                img = nib.load(nifti_file)
                img_data = img.get_fdata().astype(np.float32)
                affine = img.affine.astype(np.float16)
                keyh5 = nifti_file.stem.split('.')[0]
    
                # Write to h5 file
                grp_image.create_dataset(keyh5, data=img_data)
                grp_affine.create_dataset(keyh5, data=affine)
    
                key = keyh5.split('_')[0]
                keys.append(key)
    """

    keys = list(set(keys))  # remove duplicates of "*_2" and "*_3", revisits of patients?
    with open(output_dir.joinpath('keys', 'all.dat'), 'w') as f:   # interim_8000 was written and read as binary: "wb" / "rb" (above)
        #pickle.dump(keys, f)
        for key in keys:
            f.write(key + '\n')

    return keys


def create_csv(keys, csv_input, csv_output, verbose=False):
    # sex: 0=female, 1=male
    csv_in = pd.read_csv(csv_input, low_memory=False) # nrows=100000, cut aways a few corrupted lines at the end, header=0, names=['eid', '21022-0.0', '31-0.0', '21002-0.0', '50-0.0'])  # age, sex, weight, height
    df_sel = csv_in[['eid', '21003-2.0', '31-0.0', '21002-0.0', '50-0.0']]
    # find header
    #csv_in_info = pd.read_csv(csv_input, nrows=10)
    #cols = [col for col in csv_in_info.columns if '12144' in col]

    df = df_sel.rename(columns={'eid': 'key', '21003-2.0': 'age', '31-0.0': 'sex', '21002-0.0': 'weight', '50-0.0': 'height'})
    keys_int = [int(k) for k in keys]
    df = df[df['key'].isin(keys_int)]  # filter out only patients with imaging data
    df.to_csv(Path(csv_output))

def create_keys(keys, output_dir, n_folds=5):
    # 80% / 20 % split for train / test
    train_set, test_set = train_test_split(keys, test_size=0.2, random_state=42)
    train_folds = []
    test_folds = []
    """for train_index, test_index in KFold(n_splits=n_folds).split(keys):
        train_folds.append([keys[idx] for idx in train_index])
        test_folds.append([keys[idx] for idx in test_index])"""

    with open(output_dir.joinpath('keys', 'train_brain_mainly_healthy.dat'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(output_dir.joinpath('keys', 'test_brain_mainly_healthy.dat'), 'w') as f:
        for item in test_set:
            f.write("%s\n" % item)

    """for i in range(n_folds):
        with open(output_dir.joinpath('keys', 'train{}.dat'.format(i)), 'w') as f:
            for item in train_folds[i]:
                f.write("%s\n" % item)
        with open(output_dir.joinpath('keys', 'test{}.dat'.format(i)), 'w') as f:
            for item in test_folds[i]:
                f.write("%s\n" % item)
"""
def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for UK Biobank T1 brain MRI data.\n' \
                                                 'CSV creation\n' \
                                                 'Nifti to HDF5 conversion\n'\
                                                 'Key creation for train, test, val')
    parser.add_argument('input_dir', help='Input directory of all nifti files (*.nii.gz)')
    parser.add_argument('output_dir', help='Output directory for all files', default='/mnt/qdata/share/rakuest1/data/UKB/interim/')
    parser.add_argument('--output_file', help='Output h5 file to store processed files.', default='ukb_brain_preprocessed.h5')
    parser.add_argument('--csv_input', help='Input CSV file', default='/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv')
    parser.add_argument('--csv_output', help='Output CSV file', default='ukb_brain.csv')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Create output directory if it does not exist
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.joinpath('keys').mkdir(exist_ok=True)

    #keys = convert_nifti_h5(input_dir, output_dir, args.output_file, args.verbose)
    #keys = write_keys(input_dir, output_dir, args.output_file, args.verbose)
    #create_csv(keys, args.csv_input, output_dir.joinpath(args.csv_output), args.verbose)
    keys = pd.read_csv('/mnt/qdata/share/raecker1/ukbdata_70k/interim/keys/ukb_keys_mainly_healthy_brain.csv', header=None)
    keys = keys[0].to_list()
    #create_keys(keys, output_dir, n_folds=5)
    print('done')


if __name__ == '__main__':
    # python3 ukbbrain.py /mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain/processed /mnt/qdata/share/rakuest1/data/UKB/interim/
    main()

