import pandas as pd
import os

def merge_dfs(dfs, on=None, how='inner', suffixes=None):
    """Merge multiple dataframes into one.

    Args:
        dfs (list): List of dataframes.
        on (list, optional): Columns to merge on. Defaults to None.
        how (str, optional): How to merge. Defaults to 'outer'.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    df = dfs[0].sort_values(by=on)
    for i in range(1, len(dfs)):
        if 'fundus' in suffixes[i] or 'kidney' in suffixes[i]:
            for group in ['left', 'right']:
                df = df.merge(dfs[i][dfs[i]['orientation'].str.contains(group)], on=on, how=how, suffixes=('', '_' + group + '_' + suffixes[i]))
        else:
            df = df.merge(dfs[i], on=on, how=how, suffixes=('', '_' + suffixes[i]))
    df.rename(columns={'pred': 'pred_' + suffixes[0], 'sigma': 'sigma_' + suffixes[0], 'sex': 'sex_' + suffixes[0]},
              inplace=True)
    return df


def main(result_path='/mnt/qdata/share/rakuest1/data/UKB/interim/results'):
    # load data
    files = ['ukb_t1brain_volume', 'ukb_t1brain_volume_meta', 'ukb_fundus', 'ukb_fundus_meta', 'ukb_heart_volume', 'ukb_heart_volume_meta', 'ukb_liver_volume', 'ukb_liver_volume_meta', 'ukb_kidney_volume', 'ukb_kidney_volume_meta', 'ukb_pancreas_volume', 'ukb_pancreas_volume_meta', 'ukb_spleen_volume', 'ukb_spleen_volume_meta']
    jobs = ['brain', 'fundus', 'heart', 'liver', 'kidney', 'pancreas', 'spleen']
    cols = ['key', 'sex_brain_meta', 'age'] #, 'pred_brain', 'pred_brain_meta', 'pred_fundus', 'pred_fundus_meta', 'pred_heart', 'pred_heart_meta', 'pred_liver', 'pred_liver_meta', 'pred_kidney', 'pred_kidney_meta', 'pred_pancreas', 'pred_pancreas_meta', 'pred_spleen', 'pred_spleen_meta']
    suffixes = []
    for job in jobs:
        if job in ['fundus', 'kidney']:
            cols.append('pred_left_' + job)
            cols.append('sigma_left_' + job)
            cols.append('pred_left_' + job + '_meta')
            cols.append('sigma_left_' + job + '_meta')
            cols.append('pred_right_' + job)
            cols.append('sigma_right_' + job)
            cols.append('pred_right_' + job + '_meta')
            cols.append('sigma_right_' + job + '_meta')
        else:
            cols.append('pred_' + job)
            cols.append('sigma_' + job)
            cols.append('pred_' + job + '_meta')
            cols.append('sigma_' + job + '_meta')
        suffixes.append(job)
        suffixes.append(job + '_meta')
    dfs = []
    for file in files:
        dfval = pd.read_csv(os.path.join(result_path, file + '_val.csv'))
        dftrain = pd.read_csv(os.path.join(result_path, file + '_train.csv'))
        dfs.append(pd.concat([dfval, dftrain]))

    # merge data
    df = merge_dfs(dfs, on=['key'], how='inner', suffixes=suffixes)
    df = df[cols]
    df.rename(columns={'sex_brain_meta': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'ukb_all_results.csv'), index=False)

def sort(in_path='/mnt/qdata/share/rakuest1/data/UKB/interim/results_bak', out_path='/mnt/qdata/share/rakuest1/data/UKB/interim/results'):
    files = ['ukb_t1brain_volume', 'ukb_t1brain_volume_meta',
             'ukb_heart_volume', 'ukb_heart_volume_meta',
             'ukb_liver_volume', 'ukb_liver_volume_meta',
             'ukb_pancreas_volume', 'ukb_pancreas_volume_meta',
             'ukb_spleen_volume', 'ukb_spleen_volume_meta']

    for file in files:
        for group in ['_val', '_train']:
            df = pd.read_csv(os.path.join(in_path, file + group + '.csv'))
            df = df.sort_values(by=['key'])
            df.to_csv(os.path.join(out_path, file + group + '.csv'), index=False)

if __name__ == '__main__':
    #sort()
    main()