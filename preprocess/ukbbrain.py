import tempfile
import re
import shutil
import time
import subprocess
import argparse
import tempfile
from zipfile import ZipFile
import SimpleITK as sitk
import nibabel as nib
import os
import h5py
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def write_keys(key_path, keys):

    with open(key_path, 'w') as f:
        for key in keys:
            f.write(key + '\n')
    print(f'Keys written to {key_path}')


def convert_nifti_h5(input_dir, output_dir, key_path, output_file, verbose=False):
    """
    Converts all nifti files in input_dir to h5 files in output_dir
    Args:
        input_dir (Path): input directory with nifti files
        output_dir (Path): output directory to store h5 file
        key_path (Path): output dat file to store keys
        output_file (str): output h5 file name
        verbose (bool): print progress
    """

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    # Get list of all nifti files in input_dir
    nifti_files = [f for f in input_dir.glob('*.nii.gz')]

    # Create list of all h5 files in output_dir
    h5_path = output_dir.joinpath(output_file)

    if os.path.exists(h5_path):
        print('{} already exists'.format(h5_path))
        if os.path.exists(key_path):
            print(f'Image keys already exist in {key_path}.')
            return
        else:
            print(f'Image keys do not exist in {key_path}. Creating keys file.')
            keys = []

            hf = h5py.File(h5_path, 'r')
            for key in hf['image'].keys():
                keys.append(key.split('_')[0])
            hf.close()

            keys = list(set(keys))
            write_keys(key_path, keys)

            return

    keys = []
    hf = h5py.File(h5_path, 'w')
    grp_image = hf.create_group('image')
    grp_affine = hf.create_group('affine')

    for nifti_file in tqdm(nifti_files):
        img = nib.load(nifti_file)
        img_data = img.get_fdata().astype(np.float32)
        affine = img.affine.astype(np.float16)
        keyh5 = nifti_file.stem.split('.')[0]

        # Write to h5 file
        grp_image.create_dataset(keyh5, data=img_data)
        grp_affine.create_dataset(keyh5, data=affine)

        # Store keys
        key = keyh5.split('_')[0]
        keys.append(key)
        
        img = None
        img_data = None
        affine = None
    hf.close()

    print(f'Wrote {len(nifti_files)} nifti files to {h5_path}.')
    
    keys = list(set(keys))
    write_keys(key_path, keys)


def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for UK Biobank T1 brain MRI data.\n' \
                                                 'Nifti to HDF5 conversion\n')
    parser.add_argument('input_dir', help='Input directory of all nifti files (*.nii.gz)', default='/mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/t1_brain/raw/')
    parser.add_argument('output_dir', help='Output directory for all files', default='/mnt/qdata/share/raecker1/ukbdata_70k/interim/')
    parser.add_argument('--h5_file', help='Output h5 file to store processed files.', default='ukb_brain_preprocessed.h5')
    parser.add_argument('--key_file', help='Output csv file to store keys.', default='brain_imaging.dat')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    key_path = output_dir.joinpath('keys', args.key_file)

    # Create output directory if it does not exist
    output_dir.joinpath('keys').mkdir(exist_ok=True)

    # Convert nifti files to h5 file
    convert_nifti_h5(input_dir, output_dir, key_path, args.h5_file, args.verbose)


if __name__ == '__main__':
    main()

