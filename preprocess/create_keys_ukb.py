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
    key_file = 'ukb_keys_healthy_heart_full.csv'
    key_file_out = 'ukb_keys_healthy_heart.csv'
    #image_path = '/mnt/qdata/share/raecker1/ukbdata_70k/t1_brain/raw'
    image_path = '/mnt/qdata/share/raecker1/ukbdata_70k/sa_heart/processed/seg'
    organ = 'heart'

    output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim/')
    data_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)
    #img_list = os.listdir(image_path)
    #data_df = pd.DataFrame({'key': [l.split('_')[0] for l in img_list]})
    print(len(data_df))
    csv_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None, names=['key'], dtype=str)
    print(len(csv_df))
    df_merged = pd.merge(data_df, csv_df, on='key', how='inner')
    print(len(df_merged))
    df_diff_ukb_csvs = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)
    print(len(df_diff_ukb_csvs))
    df_filtered = df_merged[~df_merged['key'].isin(set(df_diff_ukb_csvs['key']))]
    print(len(df_filtered))
    df_filtered.to_csv(os.path.join(output_dir, 'keys', key_file_out), index=None, header=None)


def create_train_test():
    key_file = 'ukb_keys_healthy_heart.csv'
    out_name = 'heart_healthy'

    output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim')
    keys = pd.read_csv(f'/mnt/qdata/share/raecker1/ukbdata_70k/interim/keys/{key_file}', header=None)
    keys = keys[0].to_list()
    create_keys(keys, output_dir, out_name, n_folds=1)

def get_full_test_set_keys():
    organ = 'heart'
    output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim')
    #image_path = '/mnt/qdata/share/raecker1/ukbdata_70k/t1_brain/raw'
    #image_path = '/mnt/qdata/share/raecker1/ukbdata_70k/sa_heart/processed/seg'
    train_set = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/keys/train_heart_mainly_healthy.dat'
    fname = 'heart_mainly_healthy'


    img_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)                                                 # get all image keys
    img_wo_age = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)     # get all image keys without age label
    train_df = pd.DataFrame({'key': [l.strip() for l in Path(train_set).open().readlines()]})                                           # get all keys in train set

    df_filtered = img_df[~img_df['key'].isin(set(img_wo_age['key']))]       # filter out images without age label
    df_out = df_filtered[~df_filtered['key'].isin(set(train_df['key']))]    # filter out images in train set
    test_key_list = df_out['key'].to_list()

    with open(output_dir.joinpath('keys', f'full_test_{fname}.dat'), 'w') as f:
        for key in test_key_list:
            f.write("%s\n" % key)


def create_csv_images_wo_age_label():
    organ_key = '20252'
    organ = 'brain'
    csv_input = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv'
    csv_input_2 = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/ukb675384.csv'
    df_1 = pd.read_csv(csv_input, usecols=['eid', '21003-2.0'])
    df_2 = pd.read_csv(csv_input_2, usecols=['eid', f'{organ_key}-2.0', f'{organ_key}-3.0'])

    df_in_2_notin_1 = df_2[~df_2['eid'].isin(df_1['eid'])]
    print(len(df_in_2_notin_1))

    df_merged = pd.merge(df_1, df_2, on='eid', how='inner')
    df_no_age_label = df_merged[(df_merged[f'{organ_key}-2.0'].notnull() | df_merged[f'{organ_key}-3.0'].notnull()) & df_merged['21003-2.0'].isnull()]
    print(len(df_no_age_label))
    df_exclude = pd.concat([df_no_age_label['eid'], df_in_2_notin_1['eid']]).drop_duplicates()
    print(len(df_exclude))
    df_exclude.to_csv(f'/mnt/qdata/share/raecker1/ukbdata_70k/interim/images_without_age_label_{organ}.csv', index=False)


if __name__ == '__main__':
    key_file = 'ukb_keys_mainly_healthy_heart.csv'
    out_name = 'heart_mainly_healthy'

    output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim')
    keys = pd.read_csv(f'/mnt/qdata/share/raecker1/ukbdata_70k/interim/keys/{key_file}', header=None)
    keys = keys[0].to_list()
    create_keys(keys, output_dir, out_name, n_folds=1)
    #adjust_imaging_and_meta_data()