# -*- coding: utf-8 -*-
"""Dicom to nifti conversion for NAKO zipped imaging studies.
Dicom to nifti conversion for NAKO zipped imaging studies.
Example:
    Example usage::
        $ python dcm2nii -h
        $ python dcm2nii /srcdir /destdir --dixon -v --id
Todo:
    * ...
"""

import multiprocessing
import os
import sys
import time
import re
import tempfile
import pydicom as dicom
from pathlib import Path
import sys
import dicom2nifti
import dicom2nifti.settings
import SimpleITK as sitk
import shutil
import argparse
from zipfile import ZipFile
from joblib import Parallel, delayed
from walkdir import filtered_walk, file_paths
from collections.abc import Sequence
from collections import OrderedDict
import csv
import dateutil.parser
import pandas as pd
from biobank_utils import *


def unzip(zip_file, output_dir):
    """Unzips a zip file to a temporary dicom directory in the outputdir

    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): destination
    """

    with ZipFile(str(zip_file), 'r') as zipObj:
        zipObj.extractall(str(output_dir))


def get_dcm_names(dicom_dir):
    """Returns the path/names of all DICOM files in a folder as strings

    Args:
        dicom_dir (str/Path) : dicom directory

    Returns:
        list with dicom files
    """

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

    return dicom_names


def conv_dicom_nii(dicom_dir, nifti_dir):
    """Convert dicom directory to nifti file.
    Replaces DICOM files in a specified directory by .nii.gz files;
    initial DICOM files are deleted.

    Args:
        dicom_dir (str/Path): dicom directory
    """
    dicom2nifti.settings.disable_validate_slice_increment()
    dicom2nifti.convert_directory(str(dicom_dir), str(nifti_dir))
    dicom_files = get_dcm_names(dicom_dir)
    for f in dicom_files:
        os.remove(f)


def sort_dcm_dir(dicom_dir):
    """Seperate dixon contrasts.

    Separates dixon contrasts stored into four different folders named
    'fat','water','in','opp', deletes original folder.

    Args:
        dicom_dir (str/Path): directory with dcm files
    """

    dicom_dir = Path(dicom_dir)
    dicom_files = get_dcm_names(dicom_dir)
    contrasts = ['fat', 'water', 'in', 'opp']

    for contrast in contrasts:
        dicom_dir.joinpath(contrast).mkdir()

    # get echo times
    echo_times = set(())
    for f in dicom_files:
        new_echo_time = dicom.read_file(f).EchoTime
        echo_times.add(new_echo_time)

    # copy files to different folders
    for f in dicom_files:
        dicomfile = dicom.read_file(f)
        # print(str(file))
        print(dicomfile[0x00511019].value)
        if 'DIXF' in dicomfile[0x00511019].value:  # fat
            shutil.copy(f, str(dicom_dir.joinpath(contrasts[0])))
        elif 'DIXW' in dicomfile[0x00511019].value:  # water
            shutil.copy(f, str(dicom_dir.joinpath(contrasts[1])))
        elif dicomfile.EchoTime == max(echo_times):  # in
            shutil.copy(f, str(dicom_dir.joinpath(contrasts[2])))
        else:  # op
            shutil.copy(f, str(dicom_dir.joinpath(contrasts[3])))

    #shutil.rmtree(dicom_dir)

def get_tags_in_files(dicom_path, only_first=True, tag_file_path=''):
    """
    get_tags_in_files iterates over a directory, finds dicom files with
    a .dcm extension, and finds all unique dicom tag instances. it then
    writes the tags out as a csv file.
    Args:
        dicom_path (str): Path to scan for dicom files.
        tag_file_path (str): Path and file name for the output csv file.
    Returns:
        dict: A dictionary containing the tags loaded.
    """
    # create the output directory
    if not tag_file_path:
        if not os.path.exists(os.path.dirname(tag_file_path)):
            os.makedirs(os.path.dirname(tag_file_path))

    # get the tags
    tags_in_files = {}
    dicom_file_paths = file_paths(filtered_walk(dicom_path, included_files=["*.dcm", "*.ima"]))
    if only_first:
        dicom_file_paths = [dicom_file_paths[0]]

    for dicom_file_path in dicom_file_paths:
        dicom_file = pydicom.read_file(dicom_file_path)
        for item in dicom_file:
            if item.keyword not in tags_in_files:
                group = "0x%04x" % item.tag.group
                element = "0x%04x" % item.tag.element
                tags_in_files[item.keyword] = group, element, item.keyword, item.name

    # sort the tags
    tags_in_files = OrderedDict(
        sorted(tags_in_files.items(), key=(lambda k: (k[1][0], k[1][1])))
    )

    # write out the file
    if not tag_file_path:
        with open(tag_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["group", "element", "keyword", "name"])
            for item in tags_in_files:
                writer.writerow(tags_in_files[item])

    return tags_in_files


def directory_to_csv(dicom_path, csv_file_path, tags_in_files, tags_to_exclude, only_first=True):
    """
    directory_to_csv iterates over a directory, finds dicom files with
    a .dcm extension and then creates a spreadsheet containing all of
    the tag values for the tags in the csv for every dicom file
    Args:
        dicom_path (str): Path to scan for dicom files.
        csv_file_path (str): Path and file name for the output csv file.
        tags_in_files (dict): Dictionary containing tags to include in the csv
        tags_to_exclude (dict): Dictionary containing tags to exclude in the csv
    Returns:
        None
    """
    tags_in_files = tags_in_files.copy()  # copy because we're going to modify
    for tag_to_exclude in tags_to_exclude:
        if tag_to_exclude in tags_in_files:
            del tags_in_files[tag_to_exclude]

    # sort by group and then element number
    tags_in_files = OrderedDict(
        sorted(tags_in_files.items(), key=(lambda k: (k[1][0], k[1][1])))
    )
    dicom_file_paths = file_paths(filtered_walk(dicom_path, included_files=["*.dcm", "*.ima"]))
    if only_first:
        dicom_file_paths = [dicom_file_paths[0]]

    with open(csv_file_path, "w") as f:
        writer = csv.writer(f)

        # write the headers
        header_row = list(tags_in_files.keys())
        header_row.append("FilePath")
        writer.writerow(header_row)

        # write the rows
        for dicom_file_path in dicom_file_paths:
            dicom_file = pydicom.read_file(dicom_file_path)

            row_vals = []
            for keyword in tags_in_files:
                tag_val = dicom_file.get(keyword)

                if tag_val is None:
                    tag_val = ""
                else:
                    if isinstance(tag_val, Sequence) and not isinstance(
                        tag_val, (str, bytes, bytearray)
                    ):
                        tag_val = "^".join([str(x) for x in tag_val])
                    elif not isinstance(tag_val, str):
                        tag_val = str(tag_val)

                    tag_val = (
                        tag_val.replace(",", "^").replace("\n", "").replace("\r", "")
                    )
                row_vals.append(tag_val)

            row_vals.append(dicom_file_path)
            writer.writerow(row_vals)

def dcm2nii_zipped(zip_file, output_dir,
                   add_id=False,
                   single_dir=False,
                   csv='',
                   verbose=False):
    """Covert single sequence zip to nifti.

    Converts zipped NAKO DICOM data stored in a sequence folder
    (e.g.'/mnt/data/rawdata/NAKO_195/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')
    to .nii files in a defined output folder
    (such as '/mnt/data/rawdata/NAKO_195_nii/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')

    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): output directory for nifti files
        add_id (bool): add subject id (parsed from zip filename) as praefix
        single_dir (bool): save nifti files in a single directory (no subdirs)
        verbose (bool): activate prints
    """

    f = Path(zip_file)
    output_dir = Path(output_dir)

    # create temp directory
    tmp = tempfile.TemporaryDirectory()
    # get subject id
    subj_id = re.match('.*([0-9]{6}).*', f.name).group(1)

    if verbose:
        print('unzipping: ', f)

    # unzip to temp directory
    try:
        unzip(f, tmp.name)
    except:
        print(f'zip error {subj_id}', file=sys.stderr)
        return

    if verbose:
        print('converting ... ')

    # create folder with subject id
    dest_dir = output_dir.joinpath(subj_id)
    dest_dir.mkdir(exist_ok=True)

    try:
        # convert dicom files in tmpdir to nii file
        dcm_dir = next(Path(tmp.name).glob('*'))
        dcm_dir = next(dcm_dir.glob('*'))
        conv_dicom_nii(dcm_dir, dest_dir)

        # rename nifti file
        nii_path = next(dest_dir.glob('*.nii.gz'))

        if single_dir:
            # use id praefix, if all files are saved in one directory
            subj_str = (subj_id + '_')
            shutil.move(nii_path, output_dir.joinpath(subj_str + nii_path.name))
            shutil.rmtree(dest_dir)
        else:
            # if add_id = True use subj_id as filename praefix
            subj_str = (subj_id + '_') if add_id else ''
            shutil.move(nii_path, dest_dir.joinpath(subj_str + nii_path.name))

        if len(csv) > 0:
            tags_in_files = get_tags_in_files(dcm_dir)
            tags_to_exclude = {"PixelData": ["0x7fe0", "0x0010", "Pixel Data", "PixelData"]}
            directory_to_csv(dcm_dir, os.path.join(csv, subj_str + nii_path.name), tags_in_files, tags_to_exclude)

    except:
        print(f'conversion error {subj_id}', file=sys.stderr)
        shutil.rmtree(dest_dir)

    finally:
        # delete tmp directory
        tmp.cleanup()


def dcm2nii_zipped_dixon(zip_file, output_dir,
                         add_id=False,
                         single_dir=False,
                         csv='',
                         verbose=False):
    """Covert dixon sequence zip (with four contrasts) to nifti.
    Converts zipped NAKO DICOM data stored in a sequence folder
    (e.g.'/mnt/data/rawdata/NAKO_195/NAKO-195_MRT-Dateien/3D_GRE_TRA_W_COMPOSED')
    to .nii files in a defined output folder
    (such as '/mnt/data/rawdata/NAKO_195_nii/NAKO-195_MRT-Dateien/3D_GRE_TRA_W_COMPOSED').

    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): output directory for nifti files
        add_id (bool): add subject id (parsed from zip filename) as praefix
        single_dir (bool): save nifti files in a single directory (no subdirs)
        verbose (bool): activate prints
    """

    f = Path(zip_file)
    output_dir = Path(output_dir)

    # create temp directory
    tmp = tempfile.TemporaryDirectory()
    # get subject id
    subj_id = re.match('.*([0-9]{6}).*', f.name).group(1)

    if verbose:
        print('unzipping: ', f)

    # unzip to temp directory
    try:
        unzip(f, tmp.name)
    except:
        print(f'zip error {subj_id}', file=sys.stderr)
        return

    # create folder with subject id, if single_dir = False
    dest_dir = output_dir.joinpath(subj_id)
    dest_dir.mkdir(exist_ok=True)

    contrasts = ['fat', 'water', 'in', 'opp']

    if verbose:
        print('converting ...')

    try:
        # sort dcm directory
        dcm_dir = next(Path(tmp.name).glob('*'))
        sort_dcm_dir(next(dcm_dir.glob('*')))

        for contrast in contrasts:
            # create subfolder foreach contrast, if single_dir = False
            contrast_dest_dir = dest_dir.joinpath(contrast)
            contrast_dest_dir.mkdir(exist_ok=True)
            dixon_dir = dcm_dir.joinpath(contrast)
            conv_dicom_nii(dixon_dir, contrast_dest_dir)

            # rename nifti file
            nii_path = next(contrast_dest_dir.glob('*.nii.gz'))

            if single_dir:
                # use id praefix, if all files are saved in one directory
                subj_str = (subj_id + '_')
                shutil.move(nii_path, output_dir.joinpath(f'{subj_str}{contrast}.nii.gz'))
            else:
                # if add_id = True use subj_id as filename praefix
                subj_str = (subj_id + '_') if add_id else ''
                shutil.move(nii_path, contrast_dest_dir.joinpath(f'{subj_str}{contrast}.nii.gz'))

            if len(csv) > 0:
                tags_in_files = get_tags_in_files(dcm_dir)
                tags_to_exclude = {"PixelData": ["0x7fe0", "0x0010", "Pixel Data", "PixelData"]}
                directory_to_csv(dcm_dir, os.path.join(csv, f'{subj_str}{contrast}'), tags_in_files, tags_to_exclude)

        if single_dir:
            shutil.rmtree(dest_dir)

    except:
        print(f'conversion error {subj_id}', file=sys.stderr)
        shutil.rmtree(dest_dir)

    finally:
        # delete tmp directory
        tmp.cleanup()
        return

def nii_zipped_brain(zip_file, output_dir,
                   add_id=False,
                   single_dir=False,
                   csv='',
                   verbose=False):
    """Convert zipped sequences to nifti.

    Converts zipped NAKO DICOM data stored in a sequence folder
    (e.g.'/mnt/data/rawdata/NAKO_195/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')
    to .nii files in a defined output folder
    (such as '/mnt/data/rawdata/NAKO_195_nii/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')

    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): output directory for nifti files
        add_id (bool): add subject id (parsed from zip filename) as praefix
        single_dir (bool): save nifti files in a single directory (no subdirs)
        verbose (bool): activate prints
    """

    f = Path(zip_file)
    output_dir = Path(output_dir)

    # create temp directory
    tmp = tempfile.TemporaryDirectory()
    # get subject id
    subj_id = re.match('.*([0-9]{7}).*', f.name).group(1)
    run_id = f.name.split('_')[2]

    if verbose:
        print('unzipping: ', f)

    # unzip to temp directory
    try:
        unzip(f, tmp.name)
    except:
        print(f'zip error {subj_id}', file=sys.stderr)
        return

    if verbose:
        print('converting ... ')

    # create folder with subject id
    #dest_dir = output_dir.joinpath(subj_id)
    #dest_dir.mkdir(exist_ok=True)
    output_dir.joinpath('raw').mkdir(exist_ok=True)
    output_dir.joinpath('n4_flirt').mkdir(exist_ok=True)
    output_dir.joinpath('n4_flirt_robex_fcm').mkdir(exist_ok=True)
    output_dir.joinpath('processed').mkdir(exist_ok=True)

    '''
    import matplotlib.pyplot as plt
    import nibabel as nib

    def plot_data(file):
        a = nib.load(os.path.join(tmp.name, 'T1', file))
        plt.imshow(a.get_fdata()[:, :, 100])
        plt.title(file)
        plt.show()
    plot_data('T1_brain_mask.nii.gz')
    plot_data('T1_unbiased_brain.nii.gz')
    plot_data('T1.nii.gz')
    plot_data('T1_brain.nii.gz')
    plot_data('T1_orig_defaced.nii.gz')
    plot_data('T1_brain_to_MNI.nii.gz')
    '''

    try:
        # get corresponding nifti
        shutil.move(os.path.join(tmp.name, 'T1', 'T1.nii.gz'), output_dir.joinpath('raw', subj_id + '_' + run_id + '_T1.nii.gz'))
        shutil.move(os.path.join(tmp.name, 'T1', 'transforms', 'T1_to_MNI_linear.mat'), output_dir.joinpath('n4_flirt', subj_id + '_' + run_id + '_mni_linear.mat'))
        shutil.move(os.path.join(tmp.name, 'T1', 'transforms', 'T1_to_MNI_warp_coef.nii.gz'), output_dir.joinpath('n4_flirt', subj_id + '_' + run_id + '_warp_coef.nii.gz'))
        shutil.move(os.path.join(tmp.name, 'T1', 'T1_fast', 'T1_brain_bias.nii.gz'), output_dir.joinpath('n4_flirt_robex_fcm', subj_id + '_' + run_id + '_bias.nii.gz'))
        shutil.move(os.path.join(tmp.name, 'T1', 'T1_fast', 'T1_brain_seg.nii.gz'), output_dir.joinpath('n4_flirt_robex_fcm', subj_id + '_' + run_id + '_seg.nii.gz'))
        shutil.move(os.path.join(tmp.name, 'T1', 'T1_brain_mask.nii.gz'), output_dir.joinpath('n4_flirt_robex_fcm', subj_id + '_' + run_id + '_T1_brain_mask.nii.gz'))
        shutil.move(os.path.join(tmp.name, 'T1', 'T1_brain_to_MNI.nii.gz'), output_dir.joinpath('processed', subj_id + '_' + run_id + '.nii.gz'))

        #if len(csv) > 0:
            #tags_in_files = get_tags_in_files(dcm_dir)
            #tags_to_exclude = {"PixelData": ["0x7fe0", "0x0010", "Pixel Data", "PixelData"]}
            #directory_to_csv(dcm_dir, os.path.join(csv, subj_str + nii_path.name), tags_in_files, tags_to_exclude)

    except:
        print(f'conversion error {subj_id}', file=sys.stderr)
        #shutil.rmtree(dest_dir)

    finally:
        # delete tmp directory
        tmp.cleanup()

def nii_zipped_sa_heart(zip_file, output_dir,
                   add_id=False,
                   single_dir=False,
                   csv='',
                   verbose=False):
    """Convert zipped sequences to nifti.

    Converts zipped NAKO DICOM data stored in a sequence folder
    (e.g.'/mnt/data/rawdata/NAKO_195/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')
    to .nii files in a defined output folder
    (such as '/mnt/data/rawdata/NAKO_195_nii/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')

    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): output directory for nifti files
        add_id (bool): add subject id (parsed from zip filename) as praefix
        single_dir (bool): save nifti files in a single directory (no subdirs)
        verbose (bool): activate prints
    """

    f = Path(zip_file)
    output_dir = Path(output_dir)

    # create temp directory
    tmp = tempfile.TemporaryDirectory()
    # get subject id
    subj_id = re.match('.*([0-9]{7}).*', f.name).group(1)
    run_id = f.name.split('_')[2]

    if output_dir.joinpath('raw', subj_id + '_' + run_id + '_sa.nii.gz').exists():
        print(f'{subj_id} already converted', file=sys.stdout)
        return

    if verbose:
        print('unzipping: ', f)

    # unzip to temp directory
    try:
        unzip(f, tmp.name)
    except:
        print(f'zip error {subj_id}', file=sys.stderr)
        return

    if verbose:
        print('converting ... ')

    # create folder with subject id
    #dest_dir = output_dir.joinpath(subj_id)
    #dest_dir.mkdir(exist_ok=True)
    output_dir.joinpath('raw').mkdir(exist_ok=True)
    output_dir.joinpath('processed').mkdir(exist_ok=True)

    dicom_dir = tmp.name

    if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
        os.system('cp {0} {1}'.format(os.path.join(dicom_dir, 'manifest.cvs'),
                                      os.path.join(dicom_dir, 'manifest.csv')))
    process_manifest(os.path.join(dicom_dir, 'manifest.csv'),
                     os.path.join(dicom_dir, 'manifest2.csv'))
    df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), error_bad_lines=False)

    # Patient ID and acquisition date
    pid = df2.at[0, 'patientid']
    date = dateutil.parser.parse(df2.at[0, 'date'][:11]).date().isoformat()

    # Organise the dicom files
    # Group the files into subdirectories for each imaging series
    for series_name, series_df in df2.groupby('series discription'):
        series_dir = os.path.join(dicom_dir, series_name)
        if not os.path.exists(series_dir):
            os.mkdir(series_dir)
        series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
        os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

    # Convert dicom files and annotations into nifti images
    dset = Biobank_Dataset(dicom_dir)
    dset.read_dicom_images()
    Path(tmp.name).joinpath('nii').mkdir(exist_ok=True)
    dset.convert_dicom_to_nifti(os.path.join(tmp.name, 'nii'))

    try:
        # get corresponding nifti
        shutil.move(os.path.join(tmp.name, 'nii', 'sa.nii.gz'), output_dir.joinpath('raw', subj_id + '_' + run_id + '_sa.nii.gz'))

    except:
        print(f'conversion error {subj_id}', file=sys.stderr)

    finally:
        # delete tmp directory
        tmp.cleanup()


def nii_zipped_la_heart(zip_file, output_dir,
                   add_id=False,
                   single_dir=False,
                   csv='',
                   verbose=False):
    """Convert zipped sequences to nifti.

    Converts zipped NAKO DICOM data stored in a sequence folder
    (e.g.'/mnt/data/rawdata/NAKO_195/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')
    to .nii files in a defined output folder
    (such as '/mnt/data/rawdata/NAKO_195_nii/NAKO-195_MRT-Dateien/3D_GRE_TRA_W')

    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): output directory for nifti files
        add_id (bool): add subject id (parsed from zip filename) as praefix
        single_dir (bool): save nifti files in a single directory (no subdirs)
        verbose (bool): activate prints
    """

    f = Path(zip_file)
    output_dir = Path(output_dir)

    # create temp directory
    tmp = tempfile.TemporaryDirectory()
    # get subject id
    subj_id = re.match('.*([0-9]{7}).*', f.name).group(1)
    run_id = f.name.split('_')[2]

    if output_dir.joinpath('raw', subj_id + '_' + run_id + '_la_4ch.nii.gz').exists():
        print(f'{subj_id} already converted', file=sys.stdout)
        return

    if verbose:
        print('unzipping: ', f)

    # unzip to temp directory
    try:
        unzip(f, tmp.name)
    except:
        print(f'zip error {subj_id}', file=sys.stderr)
        return

    if verbose:
        print('converting ... ')

    # create folder with subject id
    #dest_dir = output_dir.joinpath(subj_id)
    #dest_dir.mkdir(exist_ok=True)
    output_dir.joinpath('raw').mkdir(exist_ok=True)
    output_dir.joinpath('processed').mkdir(exist_ok=True)

    dicom_dir = tmp.name

    if os.path.exists(os.path.join(dicom_dir, 'manifest.cvs')):
        os.system('cp {0} {1}'.format(os.path.join(dicom_dir, 'manifest.cvs'),
                                      os.path.join(dicom_dir, 'manifest.csv')))
    process_manifest(os.path.join(dicom_dir, 'manifest.csv'),
                     os.path.join(dicom_dir, 'manifest2.csv'))
    df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), error_bad_lines=False)

    # Patient ID and acquisition date
    pid = df2.at[0, 'patientid']
    date = dateutil.parser.parse(df2.at[0, 'date'][:11]).date().isoformat()

    # Organise the dicom files
    # Group the files into subdirectories for each imaging series
    for series_name, series_df in df2.groupby('series discription'):
        series_dir = os.path.join(dicom_dir, series_name)
        if not os.path.exists(series_dir):
            os.mkdir(series_dir)
        series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
        os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

    # Convert dicom files and annotations into nifti images
    dset = Biobank_Dataset(dicom_dir)
    dset.read_dicom_images()
    Path(tmp.name).joinpath('nii').mkdir(exist_ok=True)
    dset.convert_dicom_to_nifti(os.path.join(tmp.name, 'nii'))

    try:
        # get corresponding nifti
        shutil.move(os.path.join(tmp.name, 'nii', 'la_2ch.nii.gz'), output_dir.joinpath('raw', subj_id + '_' + run_id + '_la_2ch.nii.gz'))
    except:
        print(f'conversion error la_2ch_{subj_id}', file=sys.stderr)

    try:
        shutil.move(os.path.join(tmp.name, 'nii', 'la_3ch.nii.gz'), output_dir.joinpath('raw', subj_id + '_' + run_id + '_la_3ch.nii.gz'))
    except:
        print(f'conversion error la_3ch_{subj_id}', file=sys.stderr)

    try:
        shutil.move(os.path.join(tmp.name, 'nii', 'la_4ch.nii.gz'), output_dir.joinpath('raw', subj_id + '_' + run_id + '_la_4ch.nii.gz'))
    except:
        print(f'conversion error la_4ch_{subj_id}', file=sys.stderr)

    finally:
        # delete tmp directory
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
    num_cores = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description='Convert dicom directories into nifti files.')
    parser.add_argument('zip_dir', help='Path to directory with zipped files.')
    parser.add_argument('out_dir', help='Output directory to store niftis.')
    parser.add_argument('--dixon', action='store_true',
                        help='Dicom directories includes different dixon contrasts.')
    parser.add_argument('--nako', action='store_true',
                        help='NAKO database')
    parser.add_argument('--study', type=str, help='Study type to be converted (UK Biobank)', default='brain')
    parser.add_argument('--id', action='store_true')
    parser.add_argument('--cores', type=int, choices=range(1, num_cores + 1))
    parser.add_argument('--csv', type=str, help='Dump DICOM header as CSV file to specified path (str)', default='')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--singledir', action='store_true',
                        help='Store all nifti files in one directory (no sub-dirs).')
    args = parser.parse_args()

    zip_dir = Path(args.zip_dir)
    out_dir = Path(args.out_dir)


    def process_file(f):
        if args.nako:
            if args.dixon:
                dcm2nii_zipped_dixon(f, out_dir, args.id,
                                     args.singledir, args.csv, args.verbose)
            else:
                dcm2nii_zipped(f, out_dir, args.id,
                               args.singledir, args.csv, args.verbose)
        else:
            if args.study == 'brain':
                nii_zipped_brain(f, out_dir, args.id, args.singledir, args.csv, args.verbose)
            elif args.study == 'sa_heart':
                nii_zipped_sa_heart(f, out_dir, args.id, args.singledir, args.csv, args.verbose)
            elif args.study == 'la_heart':
                nii_zipped_la_heart(f, out_dir, args.id, args.singledir, args.csv, args.verbose)


    # single process version
    # t = time.time()
    # for f in zip_dir.glob('*.zip'):
    #    process_file(f)
    # elapsed_time = time.time() - t

    file_list = list(zip_dir.glob('*.zip'))

    # multiprocessing
    num_cores = 10
    if args.cores:
        num_cores = args.cores
    print(f'using {num_cores} CPU cores')

    t = time.time()
    results = Parallel(n_jobs=num_cores)(
        delayed(process_file)(f) for f in file_list)
    elapsed_time = time.time() - t

    print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


    '''
    - CSV file already available
    - Age at recruitment at Data-Field 21022: 
    csv = pd.read_csv('/mnt/qdata/rawdata/UKBIOBANK/ukbdata/ukb46167.csv', nrows=50)
    ages = csv.loc[:, csv.columns.str.contains('21022')]
    
    '''
