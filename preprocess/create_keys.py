import os
from pathlib import Path
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split


def create_keys(keys, output_dir, fname, n_folds=5):
    # 80% / 20 % split for train / test

    train_set, test_set = train_test_split(keys, test_size=0.2, random_state=42)

    with open(output_dir.joinpath('keys', f'train_{fname}.dat'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(output_dir.joinpath('keys', f'test_{fname}.dat'), 'w') as f:
        for item in test_set:
            f.write("%s\n" % item)


def create_csv_all(csv_input, output_dir, csv_output, cohort):
    print(f'(0) create {csv_output}')

    if cohort == 'nako':
        df = pd.read_csv(csv_input, usecols=['ID', 'basis_age', 'basis_sex'], delimiter=';')
        df.rename(columns={'ID': 'key', 'basis_age': 'age', 'basis_sex': 'sex'}, inplace=True)

        df['sex'] = df['sex'].map({2: 0, 1: 1})   # before: F:2, M:1 now: F:0, M:1 -> consistent to UKB

        df['key'] = df['key'].astype(str)
        df['age'] = df['age'].astype('Int64')
        df['sex'] = df['sex'].astype(int)

        df.to_csv(output_dir.joinpath(csv_output), columns=['key', 'age', 'sex'], index=True)
    
    elif cohort == 'ukb':
        df = pd.read_csv(csv_input, usecols=['eid', '21003-2.0', '21003-0.0', '31-0.0'])
        df.rename(columns={'eid': 'key', '21003-2.0': 'age', '21003-0.0': 'fundus_age', '31-0.0': 'sex'}, inplace=True)

        df_fundus = df.drop(columns=['age'])
        df_fundus = df_fundus.dropna(subset=['fundus_age'])
        df_fundus.rename(columns={'fundus_age': 'age'}, inplace=True)

        df_fundus['key'] = df_fundus['key'].astype(str)
        df_fundus['age'] = df_fundus['age'].astype(int)
        df_fundus['sex'] = df_fundus['sex'].astype(int)

        df.to_csv(output_dir.joinpath(csv_output.replace('.csv', '_fundus.csv')), columns=['key', 'age', 'sex'], index=True)

        df = df.drop(columns=['fundus_age'])
        df = df.dropna(subset=['age'])

        df['key'] = df['key'].astype(str)
        df['age'] = df['age'].astype(int)
        df['sex'] = df['sex'].astype(int)

        df.to_csv(output_dir.joinpath(csv_output), columns=['key', 'age', 'sex'], index=True)
    else:
        raise ValueError('Cohort not recognized. Please choose nako or ukb.')
    

def get_imaging_data(h5_path, output_dir, organ, cohort):
    print(f'(1) get keys from imaging data in h5 file for {organ}')

    fhandle = h5py.File(h5_path, 'r')
    if organ == 'brain' or organ == 'heart':
        group = fhandle['image']
    elif 'fundus' in organ:
        if 'left' in organ:
            group = fhandle['left']
        else:
            group = fhandle['right']
    else:
        if cohort == 'nako':
            group = fhandle['water']
        elif cohort == 'ukb':
            group = fhandle['wat']
        else:
            raise ValueError('Cohort not recognized. Please choose nako or ukb.')
    keys = group.keys()

    with open(output_dir.joinpath('keys', f'{organ}_imaging.dat'), 'w') as f:
        for key in keys:
            f.write("%s\n" % key[:6])


def get_images_wo_age_label(organ, csv_input, output_dir):
    print(f'(2) get keys with image data but without age label for {organ}')

    df = pd.read_csv(output_dir.joinpath(csv_input))
    df_no_age_label = df[df['age'].isnull()]
    df_no_age_label.to_csv(output_dir.joinpath(f'images_without_age_label_{organ}.csv'), index=False)


def adjust_imaging_and_meta_data(organ, key_file, key_file_out, output_dir, cohort):
    print(f'(3) get keys after adjusting for missing data or labels for {organ}')

    if cohort == 'nako':
        # only keep keys with image and meta data
        data_df = pd.DataFrame({'key': [l.strip() for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)        # imaging data
        csv_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None, names=['key'], dtype=str)        # meta data
        data_df['key_prefix'] = data_df['key'].str[:6]
        csv_df['key_prefix'] = csv_df['key'].str[:6]

        df_merged = pd.merge(data_df, csv_df, on='key_prefix', how='inner', suffixes=('', '_csv'))
        df_merged = df_merged[['key']]

        # exclude images without age label
        df_diff_nako_csvs = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)
        df_filtered = df_merged[~df_merged['key'].isin(set(df_diff_nako_csvs['key']))]

        # save keys
        print(f'Number of final keys for {organ}: {len(df_filtered)}')
        df_filtered.to_csv(os.path.join(output_dir, 'keys', key_file_out), index=None, header=None)
    elif cohort == 'ukb':
        # imaging data
        data_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)
        csv_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None, names=['key'], dtype=str)

        df_merged = pd.merge(data_df, csv_df, on='key', how='inner')
        df_diff_ukb_csvs = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)
        df_filtered = df_merged[~df_merged['key'].isin(set(df_diff_ukb_csvs['key']))]

        print(f'Number of final keys for {organ}: {len(df_filtered)}')
        df_filtered.to_csv(os.path.join(output_dir, 'keys', key_file_out), index=None, header=None)
    else:
        raise ValueError('Cohort not recognized. Please choose nako or ukb.')


def create_train_test(key_file, output_dir, out_name):
    print(f'(4) split in train and test set for {organ}')

    keys = pd.read_csv(os.path.join(output_dir, 'keys', key_file), header=None)
    keys = keys[0].to_list()
    create_keys(keys, output_dir, out_name, n_folds=1)


def get_full_test_set_keys(organ, output_dir, out_name):
    print(f'(5) get full test set including test set of training and unhealthy participants for {organ}')

    train_set = output_dir.joinpath('keys', f'train_{out_name}.dat')

    img_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)    # get all image keys
    img_wo_age = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)     # get all image keys without age label
    train_df = pd.DataFrame({'key': [l.strip() for l in Path(train_set).open().readlines()]})                                           # get all keys in train set

    df_filtered = img_df[~img_df['key'].isin(set(img_wo_age['key']))]       # filter out images without age label
    df_out = df_filtered[~df_filtered['key'].isin(set(train_df['key']))]    # filter out images in train set
    test_key_list = df_out['key'].to_list()

    with open(output_dir.joinpath('keys', f'full_test_{out_name}.dat'), 'w') as f:
        for key in test_key_list:
            f.write("%s\n" % key)


def get_gradcam_keys(organ, output_dir, key_file_out, out_name):
    print(f'(6) get gradcam test set including 50 samples of the healthy test set and 50 subjects of unhealthy subjects {organ}')
    
    test_set = output_dir.joinpath('keys', f'test_{out_name}.dat')

    # get all image keys
    img_df = pd.DataFrame({'key': [l.strip().split('_')[0] for l in output_dir.joinpath('keys', f'{organ}_imaging.dat').open().readlines()]}, dtype=str)

    # get all image keys without age label
    img_wo_age = pd.read_csv(os.path.join(output_dir, f'images_without_age_label_{organ}.csv'), header=None, names=['key'], dtype=str)   

    # get all keys in test set
    test_df = pd.DataFrame({'key': [l.strip() for l in Path(test_set).open().readlines()]})

    # randomly select 50 subjects from test set
    healthy_df = pd.read_csv(os.path.join(output_dir, 'keys', key_file_out), header=None, names=['key'], dtype=str)
    healthy_samples = test_df.sample(n=50, random_state=42)                     

    # randomly select 50 subjects from diseased set
    df_filtered = img_df[~img_df['key'].isin(set(img_wo_age['key']))]      
    df_out = df_filtered[~df_filtered['key'].isin(set(healthy_df['key']))]
    unhealthy_samples = df_out.sample(n=50, random_state=42)

    combined_samples = pd.concat([healthy_samples, unhealthy_samples])            
    key_list = combined_samples['key'].to_list()

    with open(output_dir.joinpath('keys', f'gradcam_{out_name}.dat'), 'w') as f:
        for key in key_list:
            f.write("%s\n" % key)


if __name__ == '__main__':
    denbi = True
    cohort = 'ukb'  # nako or ukb
    organs = ['brain', 'heart', 'kidney', 'liver', 'spleen', 'pancreas']

    h5_paths = {
        'brain': f'{cohort}_brain_preprocessed.h5',
        'heart': f'{cohort}_heart_preprocessed.h5',
        'kidney': f'{cohort}_lkd_preprocessed.h5',
        'liver': f'{cohort}_liv_preprocessed.h5',
        'spleen': f'{cohort}_spl_preprocessed.h5',
        'pancreas': f'{cohort}_pnc_preprocessed.h5'
    }
    
    if cohort == 'nako':
        csv_input = '/mnt/qdata/rawdata/NAKO_706/NAKO_706_META/30k/NAKO-707_export_baseline.csv'
        csv_output = 'nako_all.csv'
        output_dir = Path('/mnt/qdata/share/raeckev1/nako_30k/interim')
    elif cohort == 'ukb':
        if denbi:
            csv_input = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv'
            csv_output = 'ukb_all.csv'
            output_dir = Path('/mnt/qdata/share/raecker1/ukbdata_70k/interim')
        else:
            csv_input = ''
            csv_output = 'ukb_all.csv'
            output_dir = Path('/mnt/qdata/rawdata/UKBIOBANK/ukb_70k/interim')
    else:
        raise ValueError('Cohort not recognized. Please choose nako or ukb.')
    
    create_csv_all(csv_input, output_dir, csv_output, cohort)
    for i, organ in enumerate(organs):
        key_file = f'{cohort}_keys_mainly_healthy_{organ}_full.csv'
        out_name = f'{organ}_mainly_healthy'
        key_file_out = f'{cohort}_keys_mainly_healthy_{organ}.csv'
        h5_path = output_dir.joinpath(h5_paths[organ])
        get_imaging_data(h5_path, output_dir, organ, cohort)
        get_images_wo_age_label(organ, csv_output, output_dir)
        adjust_imaging_and_meta_data(organ, key_file, key_file_out, output_dir, cohort)
        create_train_test(key_file_out, output_dir, out_name)
        get_full_test_set_keys(organ, output_dir, out_name)
        get_gradcam_keys(organ, output_dir, key_file_out, out_name)

