import pandas as pd
import csv

organ_key_list = ['20209']
icd_list = ['41270', '41271', '41202', '41203', '41204', '41205', '41201']
# cardiac
diseases_list = ['I20', 'I21', 'I25', 'I2510', 'I42', 'I46', 'I48', 'I49', 'I509', 'I65', 'I739', 'I10', 'I11', 'I71', 'E11', 'E78', 'E8881', 'E731', 'E739']
# brain
#diseases_list = ['G30', 'C71', 'G40', 'F', 'G20', 'I639', 'G45', 'G12', 'R270', 'G51', 'G03', 'G610', 'S06', 'S07', 'S08', 'G04', 'S141', 'I672', 'I679', 'E11', 'E78', 'E8881', 'E731', 'E739']
# abdominal
# kidney
#diseases_list = ['N17', 'N18', 'N19', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'Q61', 'M32', 'N08', 'N02', 'E7204', 'N200', 'N11', 'C64', 'E11', 'E78', 'E8881', 'E731', 'E739']
# liver
#diseases_list = ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'R160', 'C22', 'E11', 'E78', 'E8881', 'E731', 'E739']
# spleen
#diseases_list = ['R161', 'D73', 'S36', 'C261', 'E11', 'E78', 'E8881', 'E731', 'E739']
# pancreas
#diseases_list = ['K86', 'K85', 'C25', 'E10', 'E11', 'E78', 'E8881', 'E731', 'E739']


def get_full_cols(keys, id_list):
    list_cols = []
    for ids in id_list:
        for col in keys:
            if col.startswith(ids):
                list_cols.append(col)
    return list_cols


def filter_healthy_keys(df, icd_cols_list):
    """
    Filters the keys of subjects with no ICD-Codes and imaging data of organ.
    """
    # only subjects with no ICD-Codes
    df_filt = df[df[icd_cols_list].isna().all(axis=1)]
    print(len(df_filt))
    return df_filt


def filter_mainly_healthy_keys(df, icd_cols_list, diseases_list):
    """
    Filters the keys of subjects without specific ICD-Codes.
    """
    def starts_with_any(string, prefixes):
        return any(string.startswith(prefix) for prefix in prefixes)
    
    mask = df[icd_cols_list].applymap(lambda x: starts_with_any(str(x), diseases_list))
    df_filt = df[~mask.any(axis=1)]
    print(len(df_filt))
    
    # get mainly healthy subjects
    df_filt_mainly_healthy = df_filt[~df_filt[icd_cols_list].isin(diseases_list).any(axis=1)]
    print(len(df_filt_mainly_healthy))
    
    return df_filt_mainly_healthy
    


if __name__ == '__main__':
    img_csv_path = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/ukb675384.csv'
    gen_csv_path = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv'
    save_path = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb_keys_prescan_brain.csv'

    df_first = pd.read_csv(img_csv_path, nrows=1)
    chunk_list = []
    keys = df_first.keys()
    icd_cols_list = get_full_cols(keys, icd_list)
    organ_cols_list = get_full_cols(keys, organ_key_list)

    df_first = pd.read_csv(img_csv_path, nrows=1)
    keys = df_first.keys()
    organ_cols_list = get_full_cols(keys, organ_key_list)

    df_icd = pd.read_csv(gen_csv_path, usecols=['eid']+icd_cols_list)
    df_img = pd.read_csv(img_csv_path, usecols=['eid']+organ_cols_list)

    df_cols = pd.merge(df_icd, df_img, on='eid', how='inner')
    #df_cols = pd.read_csv('C:/Users/Veronika Ecker/Documents/nako_ukb_age/ukb675384.csv', usecols=['eid']+icd_cols_list+organ_cols_list) 
        

    #keys = df_chunk.keys()
    # only subjects with imaging data of organ
    df_filt = df_cols[df_cols[organ_cols_list].notna().any(axis=1)]
    print(len(df_filt))

    df_filt = filter_mainly_healthy_keys(df_filt, icd_cols_list, diseases_list)

    """with open(save_path, "w", newline="") as csv_file:
        cwriter = csv.writer(csv_file, delimiter=',')
        for key in list(df_filt['eid']):
            cwriter.writerow([key])"""
        #ax = sns.histplot(data=df_out, x='21022-0.0', kde=True, binwidth=1)
        #ax.figure.savefig('C:/Users/Veronika Ecker/Documents/nako_ukb_age/results_true_age/age_dist_pancreas_wodis.png', bbox_inches="tight")
        #subject_count = len(df_filt.index)
        #print(f'No: {subject_count}')
    