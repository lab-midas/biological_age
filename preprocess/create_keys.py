import os
from pathlib import Path
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

    """for i in range(n_folds):
        with open(output_dir.joinpath('keys', 'train{}.dat'.format(i)), 'w') as f:
            for item in train_folds[i]:
                f.write("%s\n" % item)
        with open(output_dir.joinpath('keys', 'test{}.dat'.format(i)), 'w') as f:
            for item in test_folds[i]:
                f.write("%s\n" % item)
"""

def create_ukb_all():
    #output_dir = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/'
    csv_input = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv'
    csv_input_2 = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/ukb675384.csv'
    csv_output = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_all.csv'


    """df_1 = pd.read_csv(csv_input, usecols=['eid', '21003-2.0', '31-0.0', '21002-0.0', '50-0.0'])
    #df_1 = pd.read_csv(csv_input, usecols=['eid', '21003-2.0', '21003-1.0', '21003-0.0', '21022-0.0'])
    df_2 = pd.read_csv(csv_input_2, usecols=['eid', '20201-2.0', '20201-3.0', '20209-2.0', '20209-3.0', '20252-2.0', '20252-3.0'])
    df = pd.merge(df_2, df_1, how='inner', on='eid')
    df = df.rename(columns={'eid': 'key', '21003-2.0': 'age', '31-0.0': 'sex', '21002-0.0': 'weight', '50-0.0': 'height'})"""
    #df = df.set_index('key')
    info_df = pd.read_csv(csv_output, index_col=0, usecols=[1,2,3,4,5], dtype={'key': 'string', 'age': np.float32})
    print('done')
    #df.to_csv(csv_output, columns=['key', 'age', 'sex', 'weight', 'height'])

def adjust_imaging_and_meta_data():
    key_file = 'ukb_keys_mainly_healthy_heart_full.csv'
    key_file_out = 'ukb_keys_mainly_healthy_heart.csv'

    output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim/')
    data_df = pd.DataFrame({'key': [l.split('_')[0] for l in output_dir.joinpath('keys', 'heart_imaging.dat').open().readlines()]}, dtype=str)
    print(len(data_df))
    csv_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None, names=['key'], dtype=str)
    print(len(csv_df))
    df_merged = pd.merge(data_df, csv_df, on='key', how='inner')
    print(len(df_merged))
    df_diff_ukb_csvs = pd.read_csv(os.path.join(output_dir, 'images_without_age_label.csv'), header=None, names=['key'], dtype=str)
    print(len(df_diff_ukb_csvs))
    df_filtered = df_merged[~df_merged['key'].isin(set(df_diff_ukb_csvs['key']))]
    print(len(df_filtered))
    df_filtered.to_csv(os.path.join(output_dir, 'keys', key_file_out), index=None, header=None)

if __name__ == '__main__':
    key_file = 'ukb_keys_mainly_healthy_heart.csv'
    out_name = 'heart_mainly_healthy'

    output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim')
    keys = pd.read_csv(f'/mnt/qdata/share/raecker1/ukbdata_70k/interim/keys/{key_file}', header=None)
    keys = keys[0].to_list()
    create_keys(keys, output_dir, out_name, n_folds=1)
    #adjust_imaging_and_meta_data()