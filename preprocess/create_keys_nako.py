import os
from pathlib import Path
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.model_selection import train_test_split


def create_keys(keys, output_dir, fname, n_folds=5):
    # 80% / 20 % split for train / test
    train_set, test_set = train_test_split(keys, test_size=0.2, random_state=42)
    train_folds = []
    test_folds = []
    """for train_index, test_index in KFold(n_splits=n_folds).split(keys):
        train_folds.append([keys[idx] for idx in train_index])
        test_folds.append([keys[idx] for idx in test_index])"""

    with open(output_dir.joinpath('keys', f'train_{fname}.dat'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(output_dir.joinpath('keys', f'test_{fname}.dat'), 'w') as f:
        for item in test_set:
            f.write("%s\n" % item)


def create_nako_all(csv_input, csv_output):
    df = pd.read_csv(csv_input, usecols=['key', 'age', 'sex'])
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    #df = df.set_index('key')
    #info_df = pd.read_csv(csv_output, index_col=0, usecols=[1,2,3,4,5], dtype={'key': 'string', 'age': np.float32})
    df['key'] = df['key'].astype(str)
    df['age'] = df['age'].astype(int)
    df['sex'] = df['sex'].astype(str)
    df.to_csv(csv_output, columns=['key', 'age', 'sex'], index=True)


def get_imaging_data(h5_path, output_dir, organ):
    fhandle = h5py.File(h5_path, 'r')
    if organ == 'brain' or organ == 'heart':
        group = fhandle['image']
    else:
        group = fhandle['water']
    keys = group.keys()
    print('done')
    with open(output_dir.joinpath('keys', f'{organ}_imaging.dat'), 'w') as f:
        for key in keys:
            f.write("%s\n" % key[:6])


def get_images_wo_age_label(organ, csv_input, output_dir):
    df = pd.read_csv(csv_input)
    df_no_age_label = df[df['age'].isnull()]
    df_no_age_label.to_csv(output_dir.joinpath(f'images_without_age_label_{organ}.csv'), index=False)


def adjust_imaging_and_meta_data(organ, key_file, key_file_out, output_dir):
    # imaging data
    data_df = pd.DataFrame({'key': [l.strip() for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)
    print(len(data_df))
    # meta data
    csv_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None, names=['key'], dtype=str)
    print(len(csv_df))
    # only keep keys with image and meta data
    data_df['key_prefix'] = data_df['key'].str[:6]
    csv_df['key_prefix'] = csv_df['key'].str[:6]
    df_merged = pd.merge(data_df, csv_df, on='key_prefix', how='inner', suffixes=('', '_csv'))
    #print(df_merged.head())
    df_merged = df_merged[['key']]
    print(len(df_merged))
    # exclude images without age label
    df_diff_nako_csvs = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)
    print(len(df_diff_nako_csvs))
    df_filtered = df_merged[~df_merged['key'].isin(set(df_diff_nako_csvs['key']))]
    print(len(df_filtered))
    # save keys
    df_filtered.to_csv(os.path.join(output_dir, 'keys', key_file_out), index=None, header=None)


def create_train_test(key_file, output_dir, out_name):
    keys = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None)
    keys = keys[0].to_list()
    create_keys(keys, output_dir, out_name, n_folds=1)


def get_full_test_set_keys(organ, output_dir, out_name):
    train_set = output_dir.joinpath('keys', f'train_{out_name}.dat')

    img_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)                                                 # get all image keys
    img_wo_age = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)     # get all image keys without age label
    train_df = pd.DataFrame({'key': [l.strip() for l in Path(train_set).open().readlines()]})                                           # get all keys in train set

    df_filtered = img_df[~img_df['key'].isin(set(img_wo_age['key']))]       # filter out images without age label
    df_out = df_filtered[~df_filtered['key'].isin(set(train_df['key']))]    # filter out images in train set
    test_key_list = df_out['key'].to_list()

    with open(output_dir.joinpath('keys', f'full_test_{out_name}.dat'), 'w') as f:
        for key in test_key_list:
            f.write("%s\n" % key)


if __name__ == '__main__':
    #organs = ['brain', 'heart', 'kidneys', 'liver', 'spleen', 'pancreas']
    organs = ['kidneys', 'liver', 'spleen', 'pancreas']
    """h5_paths = [
        'nako_brain_preprocessed.h5', 
        'nako_heart_preprocessed.h5', 
        'nako_lkd_preprocessed.h5', 
        'nako_liv_preprocessed.h5', 
        'nako_spl_preprocessed.h5', 
        'nako_pnc_preprocessed.h5'
    ]"""
    h5_paths = [
        'nako_lkd_preprocessed.h5', 
        'nako_liv_preprocessed.h5', 
        'nako_spl_preprocessed.h5', 
        'nako_pnc_preprocessed.h5'
    ]
    for i, organ in enumerate(organs):
        key_file = f'nako_keys_mainly_healthy_{organ}_full.csv'
        out_name = f'{organ}_mainly_healthy'
        key_file_out = f'nako_keys_mainly_healthy_{organ}.csv'

        output_dir = Path('/mnt/qdata/share/raeckev1/nako_30k/interim')
        h5_path = output_dir.joinpath(h5_paths[i])
        #csv_input = '/home/raeckev1/nako_ukb_age/preprocess/NAKO_706_patient_data_abdominal.csv'
        #csv_output = '/mnt/qdata/share/raeckev1/nako_30k/interim/nako_all.csv'
        #get_imaging_data(h5_path, output_dir, organ)
        #get_images_wo_age_label(organ, csv_output, output_dir)
        #adjust_imaging_and_meta_data(organ, key_file, key_file_out, output_dir)
        #create_train_test(key_file_out, output_dir, out_name)
        get_full_test_set_keys(organ, output_dir, out_name)
    #create_nako_all(csv_input, csv_output)
