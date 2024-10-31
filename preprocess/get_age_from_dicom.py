import re
import sys
import tempfile
from pathlib import Path
import argparse
import zipfile
import pydicom
import numpy as np
from zipfile import ZipFile
from datetime import datetime


def unzip(zip_file, out_dir):
    with ZipFile(str(zip_file), 'r') as zipObj:
        zipObj.extractall(str(out_dir))


def get_age_from_dicom_files(zip_file, out_dir):
    f = Path(zip_file)
    output_dir = Path(out_dir)

    # create temp directory
    tmp = tempfile.TemporaryDirectory()
    # get subject id
    subj_id = re.match('.*([0-9]{6}).*', f.name).group(1)

    print('unzipping: ', f)

    # unzip to temp directory
    try:
        unzip(f, tmp.name)
    except:
        print(f'zip error {subj_id}', file=sys.stderr)
        return
    
    # Read the header of the first DICOM file in the directory
    dcm_dir = next(Path(tmp.name).glob('*'))
    dcm_dir = next(dcm_dir.glob('*'))
    first_dicom_file = next(dcm_dir.glob('*'))
    try:
        dicom_data = pydicom.dcmread(first_dicom_file)
        patient_age = dicom_data.PatientAge
        scanning_date = dicom_data.AcquisitionDate
        birth_date = dicom_data.PatientBirthDate
        print(f'Subject ID: {subj_id}, Age: {patient_age}, Scanning Date: {scanning_date}, Birth Date: {birth_date}')
        # Convert dates from string to datetime objects

        birth_date = datetime.strptime(birth_date, '%Y%m%d')
        scanning_date = datetime.strptime(scanning_date, '%Y%m%d')
        # Extract the age in years from the patient_age string
        age_years_from_dicom = int(patient_age[:-1])

        # Calculate the difference in years
        age_years = (scanning_date - birth_date).days // 365
        # Extract sex from DICOM data
        patient_sex = dicom_data.PatientSex

        # Prepare the CSV file path
        csv_file_path = output_dir / 'dicom_patient_data.csv'

        # Write the data to the CSV file
        with open(csv_file_path, 'a') as csv_file:
            csv_file.write(f'{subj_id},{patient_sex},{age_years_from_dicom}\n')

        print(f'Subject ID: {subj_id}, Age in Years: {age_years}')
    except Exception as e:
        print(f'Error reading DICOM file {first_dicom_file}: {e}', file=sys.stderr)
    # Clean up the temporary directory
    tmp.cleanup()


if __name__ == '__main__':
    """
    python dcm2nii.py '/path/to/zip_dir' '/path/to/output_dir' (:dixon) (-v) (-cores C) (-csv S)
    
    # UKBiobank
    # brain
    python dcm2nii.py /mnt/qdata/rawdata/UKBIOBANK/ukbdata/brain/t1/Dicom /mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain --cores 8
    # heart
    python dcm2nii.py /mnt/qdata/share/rafruem1/ukb/MRI/raw/ShortAxisHeart /mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart --cores 8
    """

    #parser = argparse.ArgumentParser(description='Convert dicom directories into nifti files.')
    #parser.add_argument('zip_dir', help='Path to directory with zipped files.')
    #parser.add_argument('out_dir', help='Output directory to store niftis.')

    #args = parser.parse_args()

    #zip_dir = Path(args.zip_dir)
    #out_dir = Path(args.out_dir)
    zip_dir = Path('/mnt/qdata/rawdata/NAKO_706/NAKO_706_MRT/3D_GRE_TRA_W_COMPOSED')
    out_dir = Path('/home/raecker1/nako_ukb_age/')

    file_list = list(zip_dir.glob('*.zip'))

    for file in file_list:
        get_age_from_dicom_files(file, out_dir)