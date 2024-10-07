import pandas as pd
import csv
import pyarrow
import seaborn as sns


def filter_keys(organ):
    df = pd.read_csv('/mnt/qdata/rawdata/NAKO_706/NAKO_706_META/30k/NAKO-707_export_baseline.csv', delimiter=';')

    if organ == 'heart':
        diseases_list = ['d_an_ca', 'd_an_met_1', 'd_an_met_2', 'd_an_cv_1', 'd_an_cv_2', 'd_an_cv_3', 'd_an_cv_4', 'd_an_cv_5', 'd_an_cv_6', 'd_an_lung_1', 'd_an_lung_2']
    elif organ == 'brain':
        diseases_list = ['d_an_ca', 'd_an_met_1', 'a_apo_yearsfirst', 'd_an_cv_5', 'd_an_lung_1', 'd_an_lung_2']
    elif organ == 'kidneys':
        diseases_list = ['d_an_ca', 'd_an_met_1', 'd_an_neph', 'd_an_cv_5', 'd_an_lung_1', 'd_an_lung_2']
    elif organ == 'liver':
        diseases_list = ['d_an_ca', 'd_an_met_1', 'd_an_ge_5', 'd_an_inf_4', 'd_an_inf_5', 'd_an_cv_5', 'd_an_lung_1', 'd_an_lung_2']
    elif organ == 'spleen':
        diseases_list = ['d_an_ca', 'd_an_met_1', 'd_an_cv_5', 'd_an_lung_1', 'd_an_lung_2']
    elif organ == 'pancreas':
        diseases_list = ['d_an_ca', 'd_an_met_1', 'd_an_cv_5', 'd_an_lung_1', 'd_an_lung_2']
    else:
        raise ValueError(f"Unknown organ: {organ}")

    df_filt = df
    print(len(df_filt))
    for dis in diseases_list:
        if dis[0] == 'd':   # disease yes (1), no (2): if first char is d, filter all 2 (No Disease)
            #print(df[dis].unique())
            df_filt = df_filt[df_filt[dis] == 2]
        elif dis[0] == 'a': # years since disease: if first char is a, filter all 7777 (missing data because no disease)
            df_filt = df_filt[df_filt[dis] == 7777]
        print(len(df_filt))
    print(len(df_filt))

    with open(f'/mnt/qdata/share/raeckev1/nako_30k/interim/keys/nako_keys_mainly_healthy_{organ}.csv', "w", newline="") as csv_file:
        cwriter = csv.writer(csv_file, delimiter=',')
        for key in list(df_filt['ID']):
            cwriter.writerow([str(key)[:6]])
        ax = sns.histplot(data=df_filt, x='basis_age', kde=True, binwidth=1)
        ax.figure.savefig(f'/home/raeckev1/nako_ukb_age/results/age_dist_{organ}_wo_dis.png', bbox_inches="tight")
        subject_count = len(df_filt.index)
        print(f'No: {subject_count}')
    


if __name__ == '__main__':
    organs = ['brain', 'heart', 'kidneys', 'liver', 'spleen', 'pancreas']
    for organ in organs:
        filter_keys(organ)
