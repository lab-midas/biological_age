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
from functools import reduce
from sklearn.model_selection import train_test_split, KFold


def get_keys(input_dir, output_dir):
    # Get list of all nifti files in input_dir
    search_dirs = ['abdominal_MRI', 'sa_heart', 't1_brain']

    if os.path.exists(output_dir.joinpath('keys', 'all.dat')):
        keys_all_out = [l.strip() for l in output_dir.joinpath('keys', 'all.dat').open().readlines()]
    else:
        # abdominal MRI
        search_path = Path(input_dir).joinpath('abdominal_MRI', 'raw')
        keys_abdominal = [f for f in os.listdir(search_path) if os.path.isdir(Path(search_path).joinpath(f))]

        # sa heart
        search_path = Path(input_dir).joinpath('sa_heart', 'raw')
        patients = [f for f in search_path.glob('*.nii.gz')]
        keys_saheart = [p.stem.split('.')[0].split('_')[0] for p in patients]

        # t1 brain
        search_path = Path(input_dir).joinpath('t1_brain', 'raw')
        patients = [f for f in search_path.glob('*.nii.gz')]
        keys_t1brain = [p.stem.split('.')[0].split('_')[0] for p in patients]

        # fundus
        keys_fundus_left = [l.strip() for l in Path(input_dir).joinpath('fundus', 'raw', 'keys_LeftFundus_all.dat').open().readlines()]
        keys_fundus_right = [l.strip() for l in Path(input_dir).joinpath('fundus', 'raw', 'keys_RightFundus_all.dat').open().readlines()]
        keys_fundus = list(set(keys_fundus_left + keys_fundus_right))
        # load the actually converted fundus images (written succesfully to H5)
        keys_fundus_left_h5 = [l.strip() for l in Path(output_dir).joinpath('keys', 'leftFundus_h5.dat').open().readlines()]
        keys_fundus_right_h5 = [l.strip() for l in Path(output_dir).joinpath('keys', 'rightFundus_h5.dat').open().readlines()]

        keys_all = [keys_abdominal, keys_saheart, keys_t1brain, keys_fundus]
        #keys_un = set.intersection(*map(set, keys_all))
        keys_un = list(reduce(np.intersect1d, keys_all))

        # unique keys imaging
        keys_imaging = [keys_abdominal, keys_saheart, keys_t1brain]
        keys_imaging_un = list(set.intersection(*map(set, keys_imaging)))

        # unique keys train/val
        keys_imaging_train = list(set(keys_imaging_un) - set(keys_un))
        keys_test = keys_un
        #keys_fundus_train = ['left/' + k for k in list(set(keys_fundus_left) ^ set(keys_un))] + ['right/' + k for k in list(set(keys_fundus_right) ^ set(keys_un))]
        keys_fundus_train = ['left/' + k for k in list(set(keys_fundus_left_h5) - set(keys_un))] + ['right/' + k for k in list(set(keys_fundus_right_h5) - set(keys_un))]
        keys_fundus_test = ['left/' + k for k in list(np.intersect1d(keys_fundus_left_h5, keys_un))] + ['right/' + k for k in list(np.intersect1d(keys_fundus_right_h5, keys_un))]
        keys_all_out = list(set(keys_imaging_un + keys_fundus))
        Path(output_dir).joinpath('keys').mkdir(exist_ok=True)
        with open(output_dir.joinpath('keys', 'imaging_abdomen.dat'), 'w') as f:
            for key in keys_abdominal:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'imaging_heart.dat'), 'w') as f:
            for key in keys_saheart:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'imaging_brain.dat'), 'w') as f:
            for key in keys_t1brain:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'fundus.dat'), 'w') as f:
            for key in keys_fundus:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'test.dat'), 'w') as f:
            for key in keys_test:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'train_imaging.dat'), 'w') as f:
            for key in keys_imaging_train:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'train_fundus.dat'), 'w') as f:
            for key in keys_fundus_train:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'test_fundus.dat'), 'w') as f:
            for key in keys_fundus_test:
                f.write(key + '\n')

        with open(output_dir.joinpath('keys', 'all.dat'), 'w') as f:
            for key in keys_all_out:
                f.write(key + '\n')

    return keys_all_out


def create_csv(keys, csv_input, csv_output, verbose=False):
    # sex: 0=female, 1=male
    csv_in = pd.read_csv(csv_input, low_memory=False) # nrows=100000, cut aways a few corrupted lines at the end, header=0, names=['eid', '21022-0.0', '31-0.0', '21002-0.0', '50-0.0'])  # age, sex, weight, height
    df_sel = csv_in[['eid', '21022-0.0', '31-0.0', '21002-0.0', '50-0.0']]
    # find header
    #csv_in_info = pd.read_csv(csv_input, nrows=10)
    #cols = [col for col in csv_in_info.columns if '12144' in col]

    df = df_sel.rename(columns={'eid': 'key', '21022-0.0': 'age', '31-0.0': 'sex', '21002-0.0': 'weight', '50-0.0': 'height'})
    keys_int = [int(k) for k in keys]
    df = df[df['key'].isin(keys_int)]  # filter out only patients with imaging data
    df.to_csv(Path(csv_output))


def create_keys(keys, output_dir, n_folds=5):
    # 80% / 20 % split for train / validation
    train_set, val_set = train_test_split(keys, test_size=0.2, random_state=42)
    train_folds = []
    val_folds = []
    for train_index, val_index in KFold(n_splits=n_folds).split(keys):
        train_folds.append([keys[idx] for idx in train_index])
        val_folds.append([keys[idx] for idx in val_index])

    with open(output_dir.joinpath('keys', 'train.dat'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(output_dir.joinpath('keys', 'validation.dat'), 'w') as f:
        for item in val_set:
            f.write("%s\n" % item)

    for i in range(n_folds):
        with open(output_dir.joinpath('keys', 'train{}.dat'.format(i)), 'w') as f:
            for item in train_folds[i]:
                f.write("%s\n" % item)
        with open(output_dir.joinpath('keys', 'val{}.dat'.format(i)), 'w') as f:
            for item in val_folds[i]:
                f.write("%s\n" % item)


def check_missing_keys(h5file, csv_file, key_file):
    keys_key_file = [l.strip() for l in Path(key_file).open().readlines()]
    keys_h5 = [key for key in h5py.File(h5file, 'r')['image'].keys()]
    keys_h5_prune = [k.split('_')[0] for k in keys_h5]
    keys_csv = pd.read_csv(csv_file, index_col=0, dtype={'key': 'string', 'age': np.float32})
    key_missing_h5 = [k for k in keys_key_file if k not in keys_h5_prune]
    key_missing_csv = []
    for key in keys_key_file:
        try:
            a = keys_csv.loc[key]
        except:
            key_missing_csv.append(key)

    keys_key_file_pruned = [k for k in keys_key_file if k not in key_missing_csv]
    with open(Path(key_file), 'w') as f:
        for key in keys_key_file_pruned:
            f.write(key + '\n')

    print('missing keys in h5: {}'.format(len(key_missing_h5)))
    print('missing keys in csv: {}'.format(len(key_missing_csv)))

    tmp = pd.read_csv('/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_brain.csv')
    for key in key_missing_csv:
        try:
            a = tmp.loc[key_missing_h5]
            keys_csv.append(a)
        except:
            print('key {} not found in brain csv'.format(key))



def main():
    # preprocessing of UKB data
    # Prerequisites:
    # 1. Raw data conversion
    #    Brain/Abdomen/Heart: DICOM -> Nifty conversion: dcm2nii.py
    #    Fundus: PNG -> H5 conversion: ukbfundus.py::convert_png_h5
    # 2. Nifty/PNG preprocessing with respective scripts: ukb*.py


    # python3 ukbpreprocess.py /mnt/qdata/share/rakuest1/data/UKB/raw /mnt/qdata/share/rakuest1/data/UKB/interim/
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for UK Biobank data.\n' \
                                                 'CSV creation\n' \
                                                 'Key creation for train, test, val')
    parser.add_argument('input_dir', help='Root input directory of all raw files')
    parser.add_argument('output_dir', help='Output directory for all files', default='/mnt/qdata/share/rakuest1/data/UKB/interim/')
    parser.add_argument('--csv_input', help='Input CSV file', default='/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/ukb51137.csv')
    parser.add_argument('--csv_output', help='Output CSV file', default='ukb_all.csv')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Create output directory if it does not exist
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.joinpath('keys').mkdir(exist_ok=True)
    keys = get_keys(input_dir, output_dir)
    create_csv(keys, args.csv_input, output_dir.joinpath(args.csv_output), args.verbose)
    # create_keys(keys, output_dir, n_folds=5)  # no cross-validation needed for now


if __name__ == '__main__':
    main()
