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


def get_imaging_data(h5_path, output_dir, organ):
    print(f'(1) get keys from imaging data in h5 file for {organ}')
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
    print(f'(2) get keys with image data but without age label for {organ}')
    df = pd.read_csv(csv_input)
    df_no_age_label = df[df['age'].isnull()]
    df_no_age_label.to_csv(output_dir.joinpath(f'images_without_age_label_{organ}.csv'), index=False)


def adjust_imaging_and_meta_data(organ, key_file, key_file_out, output_dir):
    print(f'(3) get keys after adjusting for missing data or labels for {organ}')
    # imaging data
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


def create_train_test(key_file, output_dir, out_name):
    print(f'(4) split in train and test set for {organ}')
    keys = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None)
    keys = keys[0].to_list()
    keys = keys[:int(len(keys) * 0.85)]
    create_keys(keys, output_dir, out_name, n_folds=1)


def get_full_test_set_keys(organ, output_dir, out_name):
    print(f'(5) get full test set including test set of training and unhealthy participants for {organ}')
    train_set = output_dir.joinpath('keys', f'train_{out_name}.dat')

    img_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)                                                 # get all image keys
    img_wo_age = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)     # get all image keys without age label
    train_df = pd.DataFrame({'key': [l.strip() for l in Path(train_set).open().readlines()]})                                           # get all keys in train set

    df_filtered = img_df[~img_df['key'].isin(set(img_wo_age['key']))]       # filter out images without age label
    df_out = df_filtered[~df_filtered['key'].isin(set(train_df['key']))]    # filter out images in train set
    test_key_list = df_out['key'].to_list()
    #test_key_list = test_key_list[int(len(test_key_list) * 0.96):]

    with open(output_dir.joinpath('keys', f'full_test_{out_name}.dat'), 'w') as f:
        for key in test_key_list:
            f.write("%s\n" % key)

def get_gradcam_keys(organ, output_dir, key_file_out, out_name):
    print(f'(6) get gradcam test set including 50 samples of the healthy test set and 50 subjects of unhealthy subjects {organ}')
    test_set = output_dir.joinpath('keys', f'test_{out_name}.dat')

    img_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)                                                 # get all image keys
    img_wo_age = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)     # get all image keys without age label
    test_df = pd.DataFrame({'key': [l.strip() for l in Path(test_set).open().readlines()]}) # get all keys in test set
    healthy_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file_out), header=None, names=['key'], dtype=str)
    healthy_samples = test_df.sample(n=50, random_state=42)                     # randomly select 50 subjects from test set

    df_filtered = img_df[~img_df['key'].isin(set(img_wo_age['key']))]       # filter out images without age label
    df_out = df_filtered[~df_filtered['key'].isin(set(healthy_df['key']))]    # unhealthy subjects
    unhealthy_samples = df_out.sample(n=50, random_state=42)                 # randomly select 50 subjects from unhealthy set
    combined_samples = pd.concat([healthy_samples, unhealthy_samples])            # combine the healthy and unhealthy samples
    
    key_list = combined_samples['key'].to_list()

    with open(output_dir.joinpath('keys', f'gradcam_{out_name}.dat'), 'w') as f:
        for key in key_list:
            f.write("%s\n" % key)


if __name__ == '__main__':
    denbi = True
    #organs = ['brain', 'heart', 'kidneys', 'liver', 'spleen', 'pancreas']
    organs = ['liver']
    """h5_paths = [
        'ukb_brain_preprocessed.h5', 
        'ukb_heart_preprocessed.h5', 
        'ukb_lkd_preprocessed.h5', 
        'ukb_liv_preprocessed.h5', 
        'ukb_spl_preprocessed.h5', 
        'ukb_pnc_preprocessed.h5'
    ]"""
    h5_paths = ['ukb_liv_preprocessed.h5']
    
    #csv_input = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_META/30k/NAKO-707_export_baseline.csv'
    if denbi:
        csv_output = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/ukb_all.csv'
    else:     
        csv_output = '/mnt/qdata/rawdata/UKBIOBANK/ukb_70k/interim/ukb_all.csv'
    #create_ukb_all(csv_input, csv_output)
    for i, organ in enumerate(organs):
        key_file = f'ukb_keys_mainly_healthy_{organ}_full.csv'
        out_name = f'{organ}_mainly_healthy'
        key_file_out = f'ukb_keys_mainly_healthy_{organ}.csv'

        if denbi:
            output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim')  # denbi
        else:
            output_dir = Path('/mnt/qdata/rawdata/UKBIOBANK/ukb_70k/interim')  # clinic

        #h5_path = output_dir.joinpath(h5_paths[i])

        #get_imaging_data(h5_path, output_dir, organ)
        #get_images_wo_age_label(organ, csv_output, output_dir)
        #adjust_imaging_and_meta_data(organ, key_file, key_file_out, output_dir)
        create_train_test(key_file_out, output_dir, out_name)
        get_full_test_set_keys(organ, output_dir, out_name)
        #get_gradcam_keys(organ, output_dir, key_file_out, out_name)

