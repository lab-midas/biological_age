import pandas as pd
import os

def merge_dfs(dfs, on=None, how='inner', suffixes=None, howfundus=False):
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
        if howfundus:
            howc = 'outer' if 'fundus' in suffixes[i] else how
        else:
            howc = how
        # df = df.merge(dfs[i].sort_values(by=on), on=on, how=howc, suffixes=suffixes)
        if 'fundus' in suffixes[i] or 'kidney' in suffixes[i]:
            for group in ['left', 'right']:
                df = df.merge(dfs[i][dfs[i]['orientation'].str.contains(group)], on=on, how=howc, suffixes=('', '_' + group + '_' + suffixes[i]))
        else:
            df = df.merge(dfs[i], on=on, how=howc, suffixes=('', '_' + suffixes[i]))
    df.rename(columns={'pred': 'pred_' + suffixes[0], 'sigma': 'sigma_' + suffixes[0], 'sex': 'sex_' + suffixes[0]},
              inplace=True)
    return df


def combine_results_ukb(result_path='C:/Users/Veronika Ecker/Documents/nako_ukb_age/Data/results/ukb_age_healthy_train_set_70k'):
    # load data
    files = ['ukb_t1brain_volume', 'ukb_fundus','ukb_heart_volume', 'ukb_liver_volume', 'ukb_kidney_volume', 'ukb_pancreas_volume', 'ukb_spleen_volume']
    jobs = ['brain', 'fundus', 'heart', 'liver', 'kidney', 'pancreas', 'spleen']
    #files = ['ukb_t1brain_volume','ukb_heart_volume']
    #jobs = ['brain', 'heart']
    #files = ['ukb_t1brain_volume']
    #jobs = ['brain']
    cols = ['key', 'sex_brain', 'age'] #, 'pred_brain', 'pred_brain_meta', 'pred_fundus', 'pred_fundus_meta', 'pred_heart', 'pred_heart_meta', 'pred_liver', 'pred_liver_meta', 'pred_kidney', 'pred_kidney_meta', 'pred_pancreas', 'pred_pancreas_meta', 'pred_spleen', 'pred_spleen_meta']
    suffixes = []
    for job in jobs:
        if job in ['fundus', 'kidney']:
            cols.append('pred_left_' + job)
            cols.append('sigma_left_' + job)
#            cols.append('pred_left_' + job + '_meta')
#            cols.append('sigma_left_' + job + '_meta')
            cols.append('pred_right_' + job)
            cols.append('sigma_right_' + job)
#            cols.append('pred_right_' + job + '_meta')
#            cols.append('sigma_right_' + job + '_meta')
        else:
            cols.append('pred_' + job)
            cols.append('sigma_' + job)
#            cols.append('pred_' + job + '_meta')
#            cols.append('sigma_' + job + '_meta')
        suffixes.append(job)
#        suffixes.append(job + '_meta')
    dfs = []
    for file in files:
        dfval = pd.read_csv(os.path.join(result_path, file + '_val.csv'))
        dftrain = pd.read_csv(os.path.join(result_path, file + '_train.csv'))
        dfs.append(pd.concat([dfval, dftrain]))

    # merge data (inner join), overlap between MRI and fundus
    df = merge_dfs(dfs, on=['key'], how='inner', suffixes=suffixes)
    df = df[cols]
    df.rename(columns={'sex_brain': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'ukb_all_results_only_overlap.csv'), index=False)

    # merge data (outer join), all data of MRI and fundus
    df = merge_dfs(dfs, on=['key'], how='outer', suffixes=suffixes)
    df = df[cols]
    df.rename(columns={'sex_brain': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'ukb_all_results.csv'), index=False)

    # merge data (outer join only for fundus), all MRI data + overlap with fundus
    df = merge_dfs(dfs, on=['key'], how='inner', suffixes=suffixes, howfundus=True)
    df = df[cols]
    df.rename(columns={'sex_brain': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'ukb_all_results_mri.csv'), index=False)


def combine_results_nako(result_path):
    # load data
    files = ['nako_t1brain_volume', 'nako_heart_volume', 'nako_liver_volume', 'nako_kidney_volume', 'nako_pancreas_volume', 'nako_spleen_volume']
    jobs = ['brain', 'heart', 'liver', 'kidney', 'pancreas', 'spleen']
    #files = ['nako_t1brain_volume', 'nako_heart_volume', 'nako_liver_volume', 'nako_kidney_volume', 'nako_pancreas_volume', 'nako_spleen_volume']
    #jobs = ['brain', 'heart', 'liver', 'kidney', 'pancreas', 'spleen']

    #files = ['nako_liver_volume', 'nako_kidney_volume', 'nako_pancreas_volume', 'nako_spleen_volume']
    #jobs = ['liver', 'kidney', 'pancreas', 'spleen']
    cols = ['key', 'sex_brain', 'age'] # ToDo: change back to brain, 'pred_brain', 'pred_brain_meta', 'pred_fundus', 'pred_fundus_meta', 'pred_heart', 'pred_heart_meta', 'pred_liver', 'pred_liver_meta', 'pred_kidney', 'pred_kidney_meta', 'pred_pancreas', 'pred_pancreas_meta', 'pred_spleen', 'pred_spleen_meta']
    suffixes = []
    for job in jobs:
        if job in ['fundus', 'kidney']:
            cols.append('pred_left_' + job)
            cols.append('sigma_left_' + job)
#            cols.append('pred_left_' + job + '_meta')
#            cols.append('sigma_left_' + job + '_meta')
            cols.append('pred_right_' + job)
            cols.append('sigma_right_' + job)
#            cols.append('pred_right_' + job + '_meta')
#            cols.append('sigma_right_' + job + '_meta')
        else:
            cols.append('pred_' + job)
            cols.append('sigma_' + job)
#            cols.append('pred_' + job + '_meta')
#            cols.append('sigma_' + job + '_meta')
        suffixes.append(job)
#        suffixes.append(job + '_meta')
    dfs = []
    for file in files:
        dfval = pd.read_csv(os.path.join(result_path, file + '_val.csv'))
        dftrain = pd.read_csv(os.path.join(result_path, file + '_train.csv'))
        dfs.append(pd.concat([dfval, dftrain]))

    # merge data (inner join), overlap between MRI and fundus
    df = merge_dfs(dfs, on=['key'], how='inner', suffixes=suffixes)
    df = df[cols]
    df.rename(columns={'sex_brain': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'nako_all_results_only_overlap.csv'), index=False)

    # merge data (outer join), all data of MRI and fundus
    df = merge_dfs(dfs, on=['key'], how='outer', suffixes=suffixes)
    df = df[cols]
    df.rename(columns={'sex_brain': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'nako_all_results.csv'), index=False)

    # merge data (outer join only for fundus), all MRI data + overlap with fundus
    df = merge_dfs(dfs, on=['key'], how='inner', suffixes=suffixes, howfundus=True)
    df = df[cols]
    df.rename(columns={'sex_brain': 'sex'}, inplace=True)
    df.to_csv(os.path.join(result_path, 'nako_all_results_mri.csv'), index=False)


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


def merge_predage_ukb_data(pred_age_csv, data_csv, result_path):
    df_results = pd.read_csv(pred_age_csv)
    df_data = pd.read_csv(data_csv, usecols=['eid', '3894-0.0', '3894-1.0', '3894-2.0', '3894-3.0', '40007-0.0', '40007-1.0'])
    df_data = df_data.rename(columns={'eid': 'key'})
    df_merged = pd.merge(df_data, df_results, how='left', on='key')
    #df_merged.to_csv(os.path.join(result_path, 'ukb_all_results_full_data_50k_trueage.csv'), index=False)
    df_merged.to_feather(os.path.join(result_path, 'ukb_all_results_partial_data_70k_healthy.feather'))



if __name__ == '__main__':
    #main()
    #pred_age_csv = 'C:/Users/Veronika Ecker/Documents/nako_ukb_age/Data/results/ukb_age_healthy_train_set_70k/ukb_all_results.csv'
    #data_csv = 'C:/Users/Veronika Ecker/Documents/nako_ukb_age/ukb675384.csv'
    #result_path = 'C:/Users/Veronika Ecker/Documents/nako_ukb_age/Data/results'
    #merge_predage_data(pred_age_csv, data_csv, result_path)
    #sort()
    #combine_results_ukb()
    #result_path = 'C:/Users/Veronika Ecker/Documents/nako_ukb_age/Data/results/ukb_age_healthy_train_set_70k'
    #result_path = 'C:/Users/Veronika Ecker/Documents/nako_ukb_age/Data/results/nako/nako_age_mainly_healthy_trainset_30k_masked'
    result_path = '/mnt/qdata/share/raecker1/ukbdata_70k/results/ukb_age_mainly_healthy_trainset_70k_masked_best_model'
    #combine_results_nako(result_path)
    combine_results_ukb(result_path)