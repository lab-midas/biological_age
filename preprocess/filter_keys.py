import pandas as pd
from pathlib import Path
from functools import reduce


# Define organ keys and associated disease codes
ORGAN_KEY_LIST = {}
DISEASES_LIST = {}

ORGAN_KEY_LIST['brain'] = ['20252']
DISEASES_LIST['brain'] = ['G30', 'C71', 'G40', 'F', 'G20', 'I639', 'G45', 'G12', 'R270', 'G51', 'G03', 'G610', 'S06', 'S07', 'S08', 'G04', 'S141', 'I672', 'I679', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['heart'] = ['20209']
DISEASES_LIST['heart'] = ['I20', 'I21', 'I25', 'I42', 'I46', 'I48', 'I49', 'I509', 'I65', 'I739', 'I10', 'I11', 'I71', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['kidney'] = ['20201']
DISEASES_LIST['kidney'] = ['N17', 'N18', 'N19', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15', 'Q61', 'M32', 'N08', 'N02', 'E7204', 'N200', 'N11', 'C64', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['liver'] = ['20201']
DISEASES_LIST['liver'] = ['K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'R160', 'C22', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['spleen'] = ['20201']
DISEASES_LIST['spleen'] = ['R161', 'D73', 'S36', 'C261', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['pancreas'] = ['20201']
DISEASES_LIST['pancreas'] = ['K86', 'K85', 'C25', 'E10', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['right_fundus'] = ['21016']
DISEASES_LIST['right_fundus'] = ['H34', 'H360', 'H40', 'H46', 'H35', 'C692', 'E11', 'E78', 'E8881', 'E731', 'E739']

ORGAN_KEY_LIST['left_fundus'] = ['21015']
DISEASES_LIST['left_fundus'] = ['H34', 'H360', 'H40', 'H46', 'H35', 'C692', 'E11', 'E78', 'E8881', 'E731', 'E739']

SPECIFIC_DISEASES = ['G30']


def get_healthy_keys_ukb(organs, csv_file, output_dir, output_file):
    """Get keys of healthy participants, i.e. those without diseases in DISEASES_LIST for each organ.
    Args:
        organs (list): List of organs to check for diseases.
        csv_file (str): Path to the CSV file containing participant data.
        output_dir (str): Directory to save the output files.
    """

    def match_icd(row, organ):
        for col in icd_cols:
            val = str(row[col])
            for icd in DISEASES_LIST[organ]:
                if val.startswith(icd):
                    return True
        return False
    
    output_dir = Path(output_dir)
    df_first = pd.read_csv(csv_file, nrows=1)
    icd_cols = [col for col in df_first.columns if col.startswith('41270-') or col.startswith('41280-') or col.startswith('21003-')]
    df = pd.read_csv(csv_file, usecols=['eid', '52-0.0', '34-0.0'] + icd_cols)

    for organ in organs:
        df[f"disease_found_{organ}"] = df.apply(lambda row: match_icd(row, organ), axis=1)
        print(f"Number of rows with disease found for {organ}: {df[f'disease_found_{organ}'].sum()}")
        healthy_eids = df.loc[~df[f"disease_found_{organ}"], 'eid']
        healthy_eids.to_csv(output_dir.joinpath(output_file.replace('<organ>', organ)), index=False, header=False)


def get_disease_status(organs, output_file, csv_file, disease_list, type='specific'):
    """Get disease status for each participant in the CSV file.
    Args:
        organs (list): List of organs to check for diseases.
        output_dir (str): Directory to save the output files.
        output_file (str): Name of the output file.
        csv_file (str): Path to the CSV file containing participant meta data.
        disease_list (list of str): List of diseases (ICD10-Codes) to check for.
        type (str): 'specific' to check for specific diseases, 'all' to check for all diseases in DISEASES_LIST.

    Saves dataframe with additional columns:
        - has_diseases: boolean indicating if any diseases were found
        - earliest_disease_age: age at earliest disease onset
        - current_age: current age from column 21003-2.0
        - disease_before_current_age: boolean indicating if disease age < current age
    """

    def check_diseases_and_compare_ages(df, organ, disease_list):
        """
        Check if any diseases from disease_list are found in ICD columns (41270*).
        If found, collect corresponding dates from date columns (41280*), calculate 
        age at disease onset using birth year (34-0.0) and birth month (52-0.0), and 
        compare with current age in column 21003-2.0.
        """
        
        icd_cols_41270 = [col for col in df.columns if col.startswith('41270')]
        date_cols_41280 = [col for col in df.columns if col.startswith('41280')]
        birth_year_col = '34-0.0'
        birth_month_col = '52-0.0'
        
        icd_to_date_mapping = {}
        
        # Map 41270 to 41280 columns
        for icd_col in icd_cols_41270:
            suffix = icd_col.replace('41270', '')
            corresponding_date_col = '41280' + suffix
            if corresponding_date_col in date_cols_41280:
                icd_to_date_mapping[icd_col] = corresponding_date_col
        
        print(f"Found {len(icd_to_date_mapping)} ICD-date column pairs")
        
        # Function to calculate age from disease date and birth info
        def calculate_age_at_disease(disease_date_str, birth_year, birth_month):
            """Calculate age at disease onset"""        
            try:
                if isinstance(disease_date_str, str):
                    disease_date = pd.to_datetime(disease_date_str)
                else:
                    disease_date = pd.to_datetime(str(disease_date_str))
                
                disease_year = disease_date.year
                disease_month = disease_date.month
                
                # Age at disease onset
                age_at_disease = disease_year - int(birth_year)
                
                # Adjust for birth month if available
                if not pd.isna(birth_month) and disease_month < int(birth_month):
                    age_at_disease -= 1
                    
                return age_at_disease
            except:
                return None
        
        def process_row(row, organ):
            found_diseases = []
            disease_ages = []
            
            # Get birth info from specific columns
            birth_year = row.get(birth_year_col, None)
            birth_month = row.get(birth_month_col, None)
            
            for icd_col, date_col in icd_to_date_mapping.items():
                icd_value = str(row[icd_col]) if pd.notna(row[icd_col]) else ""
                
                # Check if any disease from disease_list is found
                for disease in disease_list:
                    if icd_value.startswith(disease):
                        found_diseases.append(disease)

                        # Calculate age at disease onset if date is available
                        if pd.notna(row[date_col]):
                            age_at_disease = calculate_age_at_disease(row[date_col], birth_year, birth_month)
                            if age_at_disease is not None:
                                disease_ages.append(age_at_disease)
                        break
            
            # Process results
            has_diseases = len(found_diseases) > 0
            earliest_disease_age = min(disease_ages) if disease_ages else None
            if organ == 'fundus':
                current_age = row.get('21003-0.0', None)
            else:
                current_age = row.get('21003-2.0', None)
            
            # Compare ages if both are available
            disease_before_current_age = None
            disease_after_current_age = None
            if earliest_disease_age is not None and current_age is not None:
                disease_before_current_age = earliest_disease_age < current_age
                disease_after_current_age = earliest_disease_age > current_age

            return pd.Series({
                'has_diseases': has_diseases,
                'earliest_disease_age': earliest_disease_age,
                'current_age': current_age,
                'disease_before_current_age': disease_before_current_age,
                'disease_after_current_age': disease_after_current_age,
                'found_diseases_count': len(found_diseases),
                'disease_ages_count': len(disease_ages)
            })
        
        # Apply the function to each row
        print("Processing rows...")
        result_df = df.apply(lambda row: process_row(row, organ), axis=1)
        
        # Combine with original dataframe
        df_with_results = pd.concat([df, result_df], axis=1)
        
        total_rows = len(df_with_results)
        has_diseases_count = df_with_results['has_diseases'].sum()
        has_current_age = df_with_results['current_age'].notna().sum()
        disease_before_current_age_count = df_with_results['disease_before_current_age'].sum() if df_with_results['disease_before_current_age'].notna().any() else 0
        
        print(f"\nSummary:")
        print(f"Total rows: {total_rows}")
        print(f"Rows with diseases found: {has_diseases_count}")
        print(f"Rows with current age (21003-2.0): {has_current_age}")
        print(f"Rows where disease occurred before current age: {disease_before_current_age_count}")
        
        return df_with_results
    
    df_first = pd.read_csv(csv_file, nrows=1)
    icd_cols = [col for col in df_first.columns if col.startswith('41270-') or col.startswith('41280-') or col.startswith('21003-')]
    df = pd.read_csv(csv_file, usecols=['eid', '52-0.0', '34-0.0'] + icd_cols)

    if type == 'specific':
        disease_list = SPECIFIC_DISEASES
        for disease in disease_list:
            df_with_disease_analysis = check_diseases_and_compare_ages(df, None, [disease])
            df_with_disease_analysis.to_csv(output_file.replace('<>', disease), index=False)
            print(f"Processed disease: {disease}, saved to {output_file.replace('<>', disease)}")
    else:
        for organ in organs:
            disease_list = DISEASES_LIST[organ]
            df_with_disease_analysis = check_diseases_and_compare_ages(df, organ, disease_list)
            df_with_disease_analysis.to_csv(output_file.replace('<>', organ), index=False)
            print(f"Processed organ: {organ}, saved to '{output_file.replace('<>', organ)}'")


def get_disease_status_combined(organs, csv_file, input_file, output_file):
    # Read meta data
    df_meta = pd.read_csv(csv_file, usecols=["eid", "31-0.0", "21003-2.0"])
    df_meta = df_meta.rename(columns={"31-0.0": "sex", "21003-2.0": "age"})

    # Prepare list to collect per-organ dataframes
    organ_dfs = []

    for organ in organs:
        df_disease = pd.read_csv(input_file.replace('<>', organ), usecols=["eid", "has_diseases", "disease_before_current_age", "disease_after_current_age"])
        if organ == 'kidney':
            for orientation in ['left_kidney', 'right_kidney']:
                df_disease_temp = df_disease.rename(columns={
                    "has_diseases": f"has_{orientation}_diseases",
                    "disease_before_current_age": f"{orientation}_disease_before_current_age",
                    "disease_after_current_age": f"{orientation}_disease_after_current_age"
                })
                # Convert boolean-like columns to int (1/0)
                organ_dfs.append(df_disease_temp)
        else:
            df_disease = df_disease.rename(columns={
                "has_diseases": f"has_{organ}_diseases",
                "disease_before_current_age": f"{organ}_disease_before_current_age",
                "disease_after_current_age": f"{organ}_disease_after_current_age"
            })
            # Convert boolean-like columns to int (1/0)
            organ_dfs.append(df_disease)

    # Merge all organ dfs on 'eid'
    df_merged = reduce(lambda left, right: pd.merge(left, right, on="eid", how="outer"), organ_dfs)

    # Merge with meta data
    df_final = pd.merge(df_merged, df_meta, on="eid", how="left")

    # Reorder columns: eid, age, sex, then all organ columns
    cols = ["eid", "age", "sex"] + [col for col in df_final.columns if col not in ["eid", "age", "sex"]]
    df_final = df_final[cols]

    # Save to CSV
    df_final.to_csv(output_file, index=False)


def get_healthy_keys_nako(organ, csv_file, output_file, output_dir):
    for organ in organs:
        df = pd.read_csv(csv_file, delimiter=';')

        if organ == 'heart':
            diseases_list = ['d_an_ca', 'd_an_met_1', 'd_an_met_2', 'd_an_cv_1', 'd_an_cv_2', 'd_an_cv_3', 'd_an_cv_4', 'd_an_cv_5', 'd_an_cv_6', 'd_an_lung_1', 'd_an_lung_2']
        elif organ == 'brain':
            diseases_list = ['d_an_ca', 'd_an_met_1', 'a_apo_yearsfirst', 'd_an_cv_5', 'd_an_lung_1', 'd_an_lung_2']
        elif organ == 'kidney':
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
                df_filt = df_filt[df_filt[dis] == 2]
            elif dis[0] == 'a': # years since disease: if first char is a, filter all 7777 (missing data because no disease)
                df_filt = df_filt[df_filt[dis] == 7777]
            print(len(df_filt))
        print(len(df_filt))

        output_dir = Path(output_dir)

        df_filt['ID'] = df_filt['ID'].astype(str).str[:6]

        subject_count = len(df_filt.index)
        print(f'No: {subject_count}')

        # Save to CSV
        df_filt['ID'].to_csv(output_dir.joinpath(output_file.replace('<organ>', organ)), index=False, header=False)


if __name__ == '__main__':
    organs = ['brain', 'heart', 'kidney', 'liver', 'spleen', 'pancreas', 'left_fundus', 'right_fundus']
    cohort = 'ukb'  # nako or ukb

    if cohort == 'ukb':

        csv_file = '/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv'
        output_dir = '/mnt/qdata/share/raecker1/ukbdata_70k/interim/keys'

        mainly_healthy_file = f"ukb_keys_mainly_healthy_<organ>_full.csv"
        get_healthy_keys_ukb(organs, csv_file, output_dir, mainly_healthy_file)

        disease_file = f'ukb677731_disease_status_<>.csv'
        get_disease_status(organs, disease_file, csv_file, disease_list=None, type='all')

        diseases_all_organs = "ukb_all_organs_disease_status.csv"
        get_disease_status_combined(organs, csv_file, disease_file, diseases_all_organs)
    else:
        mainly_healthy_file = f'nako_keys_mainly_healthy_<organ>.csv'
        output_dir = '/mnt/qdata/share/raeckev1/nako_30k/interim/keys'
        get_healthy_keys_nako(organs)
        
