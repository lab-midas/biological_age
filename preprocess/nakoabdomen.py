import tempfile
import re
import shutil
import time
import subprocess
import argparse
import tempfile
from zipfile import ZipFile
import signal
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import os
import h5py
import dicom2nifti
import csv
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from skimage.measure import regionprops
from scipy.ndimage import label
from nakoheart import crop
import ast
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from glob import glob


def unzip(zip_file, output_dir):
    """Unzips a zip file to a temporary dicom directory in the outputdir
    Args:
        zip_file (str/Path): zip file to extract
        output_dir (str/Path): destination
    """
    with ZipFile(str(zip_file), 'r') as zipObj:
        zipObj.extractall(str(output_dir))

def write_keys(input_dir, output_dir, output_file, verbose=False):
    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    if os.path.exists(output_dir.joinpath('keys', 'all.dat')):
        print('keys already exists')
        #return pickle.load(open(output_dir.joinpath('keys', 'all.dat'), 'r'))
        return [l.strip() for l in output_dir.joinpath('keys', 'all.dat').open().readlines()]

    # Get list of all nifti files in input_dir
    nifti_files = [f for f in input_dir.glob('*.nii.gz')]

    keys = []
    for nifti_file in tqdm(nifti_files):
        keyh5 = nifti_file.stem.split('.')[0]
        key = keyh5.split('_')[0]
        keys.append(key)

    keys = list(set(keys))  # remove duplicates of "*_2" and "*_3", revisits of patients?
    with open(output_dir.joinpath('keys', 'all.dat'), 'w') as f:  # interim_8000 was written and read as binary: "wb" / "rb" (above)
        #pickle.dump(keys, f)
        for key in keys:
            f.write(key + '\n')

    return keys


def rewrite_keys(input_dir, output_dir, output_file, save_path, verbose=False):
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]
    # patsseg = list(bounding_boxes_rem['pat'].values)
    pats = list(bounding_boxes_rem['pat'].values)
    pats = [str(k) for k in pats]

    keys_imaging_abdomen_train = [l.strip() for l in Path(output_dir).joinpath('keys', 'train_imaging_abdomen.dat').open().readlines()]
    keys_imaging_abdomen_test = [l.strip() for l in Path(output_dir).joinpath('keys', 'test_abdomen.dat').open().readlines()]

    keys_imaging_abdomen_train = [l for l in keys_imaging_abdomen_train if l in pats]
    keys_imaging_abdomen_test = [l for l in keys_imaging_abdomen_test if l in pats]

    with open(output_dir.joinpath('keys', 'train_imaging_abdomen.dat'), 'w') as f:
        for key in keys_imaging_abdomen_train:
            f.write(key + '\n')

    with open(output_dir.joinpath('keys', 'test_abdomen.dat'), 'w') as f:
        for key in keys_imaging_abdomen_test:
            f.write(key + '\n')

def inspect_boxes_segmentation(input_dir, output_dir, output_file, save_path, verbose=False):
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen.csv'))

    failed_keys = bounding_boxes.loc[(bounding_boxes['liv'] == '[-1, -1, -1, -1, -1, -1]') |
                                     (bounding_boxes['spl'] == '[-1, -1, -1, -1, -1, -1]') |
                                     (bounding_boxes['rkd'] == '[-1, -1, -1, -1, -1, -1]') |
                                     (bounding_boxes['lkd'] == '[-1, -1, -1, -1, -1, -1]') |
                                     (bounding_boxes['pnc'] == '[-1, -1, -1, -1, -1, -1]')]

    failed_keys.to_csv(Path(save_path).joinpath('failed_segmentations.csv'))
    keys_failed = list(failed_keys['pat'].values)
    keys_failed = [str(k) for k in keys_failed]
    keys_imaging_train = [l.strip() for l in Path(output_dir).joinpath('keys', 'train_imaging.dat').open().readlines()]
    keys_imaging_test = [l.strip() for l in Path(output_dir).joinpath('keys', 'test.dat').open().readlines()]

    keys_imaging_abdomen_train = [l for l in keys_imaging_train if l not in keys_failed]
    keys_imaging_abdomen_test = [l for l in keys_imaging_test if l not in keys_failed]


    if len(keys_imaging_abdomen_train) != len(keys_imaging_train):
        with open(output_dir.joinpath('keys', 'train_imaging_abdomen.dat'), 'w') as f:
            for key in keys_imaging_abdomen_train:
                f.write(key + '\n')

    if len(keys_imaging_abdomen_test) != len(keys_imaging_test):
        with open(output_dir.joinpath('keys', 'test_abdomen.dat'), 'w') as f:
            for key in keys_imaging_abdomen_test:
                f.write(key + '\n')

    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                     (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                     (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                     (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                     (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]

    organs = ['liv', 'spl', 'rkd', 'lkd', 'pnc']
    fig, axs = plt.subplots(len(organs), 3, figsize=(15, 20))
    for idxo, organ in enumerate(organs):
        data = list(bounding_boxes_rem[organ].values)
        data = np.asarray([list(ast.literal_eval(l)) for l in data])
        crop_shapes = data[:, 3:6] - data[:, 0:3]
        min_shape = np.min(crop_shapes, axis=0)
        max_shape = np.max(crop_shapes, axis=0)
        per_shape = np.percentile(crop_shapes, 98, axis=0)
        median_shape = np.median(crop_shapes, axis=0)
        print(f'{organ}:')
        print(f'min: {min_shape}')
        print(f'max: {max_shape}')
        print(f'per: {per_shape}')
        print(f'med: {median_shape}')
        axs[idxo, 0].hist(crop_shapes[:, 0], bins=50)
        axs[idxo, 0].set_title(organ + '_x')
        axs[idxo, 1].hist(crop_shapes[:, 1], bins=50)
        axs[idxo, 1].set_title(organ + '_y')
        axs[idxo, 2].hist(crop_shapes[:, 2], bins=50)
        axs[idxo, 2].set_title(organ + '_z')

    plt.show()


def convert_nifti_h5_masked_parallel(input_dir, output_dir, output_file, save_path, parallelize, verbose=False):
    """
    Converts all nifti files in input_dir to h5 files in output_dir
    """

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen_largest_component_parallel.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]

    pats = list(bounding_boxes_rem['pat'].values)
    pats = [str(k) for k in pats]
    pats = ['123585']

    sel_shape = {'liv': [185, 150, 70], 'spl': [80, 80, 50], 'rkd': [60, 60, 50], 'lkd': [60, 60, 50], 'pnc': [120, 70, 70]}     #nako
    contrasts = ['fat', 'in', 'opp', 'water']

    keys = []
    files = ['nako_liv_preprocessed_masked.h5', 'nako_spl_preprocessed_masked.h5', 'nako_rkd_preprocessed_masked.h5', 'nako_lkd_preprocessed_masked.h5', 'nako_pnc_preprocessed_masked.h5']

    for idx, file in enumerate(files):
        output_dir.joinpath(Path(file).stem).mkdir(exist_ok=True)   # create nako_xxx_preprocessed dir

    def process_file(pat, parallelize):
        for contrast in contrasts:
            # check existence in file
            exists_all = []
            for iclass in np.arange(1, 6):
                if iclass == 1: class_name = 'liv'
                elif iclass == 2: class_name = 'spl'
                elif iclass == 3: class_name = 'rkd'
                elif iclass == 4: class_name = 'lkd'
                elif iclass == 5: class_name = 'pnc'
                else: raise ValueError('Class not recognized')
                if output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5').exists(): # checks if files (4 contrasts) for pat exist in ukb_xxx_preprocessed dir
                    exists_all.append(True)
                else:
                    exists_all.append(False)
            
            if np.all(exists_all):
                #print(files[iclass-1] + ': ' + contrast + '_' + pat + '.h5 already exists')
                continue            
            try:
                if not os.path.exists(input_dir.joinpath(pat)):
                    fname = pat + '_30'
                else:
                    fname = pat
                if os.path.isdir(input_dir.joinpath(fname, contrast)):
                    nifti_file = input_dir.joinpath(fname, contrast, contrast + '.nii.gz')
                else:
                    nifti_file = input_dir.joinpath(fname, contrast + '.nii.gz')
                img = nib.load(nifti_file)
                img_data = img.get_fdata().astype(np.float32)
                affine = img.affine.astype(np.float16)
                if not os.path.exists(save_path.joinpath(pat)):
                    mname = pat + '_30'
                else:
                    mname = pat
                mask = nib.load(save_path.joinpath(mname, 'prd.nii.gz'))
                mask_data = mask.get_fdata().astype(np.float32)
            except Exception as exc:
                print(exc)
                print('Error in pat:' + str(pat))
                return

            for iclass in np.arange(1, 6):
                if iclass == 1: class_name = 'liv'
                elif iclass == 2: class_name = 'spl'
                elif iclass == 3: class_name = 'rkd'
                elif iclass == 4: class_name = 'lkd'
                elif iclass == 5: class_name = 'pnc'
                else: raise ValueError('Class not recognized')

                img_mask = np.where(mask_data == iclass, 1, 0)

                sel_shape_curr = sel_shape[class_name]
                box = bounding_boxes.loc[bounding_boxes['pat'] == int(pat)][class_name].values
                box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
                center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
                center = [int(x) for x in center]
                
                img_crop = crop(img_data, sel_shape_curr, center)   # crop image around segmentation mask
                mask_crop = crop(img_mask, sel_shape_curr, center)
                img_crop = img_crop * mask_crop

                file_curr = output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5')
                with h5py.File(file_curr, 'w') as hfcurr:
                    hfcurr.create_dataset('image', data=img_crop)   # create contrast_pat.h5 file in ukb_xxx_preprocessed dir

                if iclass == 1 and contrast == 'fat':
                    file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, 'affine_' + pat + '.h5')
                    with h5py.File(file_curr, 'w') as hfcurr:
                        hfcurr.create_dataset('affine', data=affine)

    num_cores = 1
    print(f'using {num_cores} CPU cores')

    t = time.time()
    _ = Parallel(n_jobs=num_cores)(
        delayed(process_file)(f, parallelize) for f in tqdm(pats))
    elapsed_time = time.time() - t

    print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    return keys


def convert_nifti_h5_masked(input_dir, output_dir, output_file, save_path, parallelize, verbose=False):
    """
    Converts all nifti files in input_dir to h5 files in output_dir
    """

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    # Get list of all nifti files in input_dir
    #pats = input_dir.glob('*')
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen_largest_component_parallel.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]

    pats = list(bounding_boxes_rem['pat'].values)
    pats = [str(k) for k in pats]

    #sel_shape = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50], 'pnc': [80, 50, 50]}    #ukb
    sel_shape = {'liv': [185, 150, 70], 'spl': [80, 80, 50], 'rkd': [60, 60, 50], 'lkd': [60, 60, 50], 'pnc': [120, 70, 70]}     #nako

    contrasts = ['fat', 'in', 'opp', 'water']

    keys = []
    files = ['nako_liv_preprocessed_masked.h5', 'nako_spl_preprocessed_masked.h5', 'nako_rkd_preprocessed_masked.h5', 'nako_lkd_preprocessed_masked.h5', 'nako_pnc_preprocessed_masked.h5']
    modes = ['w', 'w', 'w', 'w', 'w']
    for idx, file in enumerate(files):
        if output_dir.joinpath(file).exists():
            Path(output_dir.joinpath(file)).unlink()    # remove existing ukb_xxx_preprocessed.h5 file
            #modes[idx] = 'a'
        output_dir.joinpath(Path(file).stem).mkdir(exist_ok=True)   # create ukb_xxx_preprocessed dir

    hfs = {'liv': h5py.File(output_dir.joinpath('nako_liv_preprocessed_masked.h5'), modes[0]),
           'spl': h5py.File(output_dir.joinpath('nako_spl_preprocessed_masked.h5'), modes[1]),
           'rkd': h5py.File(output_dir.joinpath('nako_rkd_preprocessed_masked.h5'), modes[2]),
           'lkd': h5py.File(output_dir.joinpath('nako_lkd_preprocessed_masked.h5'), modes[3]),
           'pnc': h5py.File(output_dir.joinpath('nako_pnc_preprocessed_masked.h5'), modes[4])}

    #for pat in tqdm(pats):
    def process_file(pat, parallelize):
        #if pat.name not in patsseg:
        #    continue
        for contrast in contrasts:
            # check existence in file
            exists_all = []
            for iclass in np.arange(1, 6):
                if iclass == 1:
                    class_name = 'liv'
                elif iclass == 2:
                    class_name = 'spl'
                elif iclass == 3:
                    class_name = 'rkd'
                elif iclass == 4:
                    class_name = 'lkd'
                elif iclass == 5:
                    class_name = 'pnc'
                else:
                    raise ValueError('Class not recognized')
                #if modes[iclass-1] == 'a' and contrast + '/' + pat in hfs[class_name]:
                if output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5').exists(): # checks if files (4 contrasts) for pat exist in ukb_xxx_preprocessed dir
                    exists_all.append(True)
                else:
                    exists_all.append(False)
            
            if np.all(exists_all):
                doload = False
            else:
                doload = True

            if doload:  # load nifti files only if not yet in ukb_xxx_preprocessed dir
                if not os.path.exists(input_dir.joinpath(pat)):
                    fname = pat + '_30'
                else:
                    fname = pat
                if os.path.isdir(input_dir.joinpath(fname, contrast)):
                    nifti_file = input_dir.joinpath(fname, contrast, contrast + '.nii.gz')
                else:
                    nifti_file = input_dir.joinpath(fname, contrast + '.nii.gz')
                img = nib.load(nifti_file)
                img_data = img.get_fdata().astype(np.float32)
                affine = img.affine.astype(np.float16)
                keyh5 = nifti_file.stem.split('.')[0]
                if not os.path.exists(save_path.joinpath(pat)):
                    mname = pat + '_30'
                else:
                    mname = pat
                mask = nib.load(save_path.joinpath(mname, 'prd.nii.gz'))
                mask_data = mask.get_fdata().astype(np.float32)


            for iclass in np.arange(1, 6):
                #if exists_all[iclass-1]:
                #    continue
                if iclass == 1:
                    class_name = 'liv'
                elif iclass == 2:
                    class_name = 'spl'
                elif iclass == 3:
                    class_name = 'rkd'
                elif iclass == 4:
                    class_name = 'lkd'
                elif iclass == 5:
                    class_name = 'pnc'
                else:
                    raise ValueError('Class not recognized')
                if exists_all[iclass-1]:     
                    # link pats in ukb_xxx_preprocessed.h5 if files for pat already exist in ukb_xxx_preprocessed dir
                    file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, contrast + '_' + pat + '.h5')
                    if not parallelize:
                        hfs[class_name][contrast + '/' + pat] = h5py.ExternalLink(file_curr, "/image")
                    if iclass == 1 and contrast == 'fat':
                        file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, 'affine_' + pat + '.h5')
                        if not parallelize:
                            hfs[class_name]['affine/' + pat] = h5py.ExternalLink(file_curr, "/affine")
                    continue

                img_mask = np.where(mask_data == iclass, 1, 0)

                sel_shape_curr = sel_shape[class_name]

                box = bounding_boxes.loc[bounding_boxes['pat'] == int(pat)][class_name].values
                box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
                center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
                center = [int(x) for x in center]

                # list(box[0:2] + np.floor(np.asarry(sel_shape[0:2]) / 2)) + list(np.floor(np.asarry(box[5] + box[2])/2)) +

                # crop image around segmentation mask
                img_crop = crop(img_data, sel_shape_curr, center)
                mask_crop = crop(img_mask, sel_shape_curr, center)
                img_crop = img_crop * mask_crop

                #save_path_curr = Path(save_path).joinpath(key + keyh5.split('_')[1])
                #Path(save_path_curr).mkdir(exist_ok=True)

                # Write to h5 file
                #hfs[class_name].create_dataset(contrast + '/' + pat, data=img_crop)
                #if iclass == 1 and contrast == 'fat':
                #    hfs[class_name].create_dataset('affine/' + pat, data=affine)
                file_curr = output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5')
                with h5py.File(file_curr, 'w') as hfcurr:
                    hfcurr.create_dataset('image', data=img_crop)   # create contrast_pat.h5 file in ukb_xxx_preprocessed dir
                if not parallelize:
                    hfs[class_name][contrast + '/' + pat] = h5py.ExternalLink(file_curr, "/image")   # link contrast_pat.h5 file to ukb_xxx_preprocessed.h5

                if iclass == 1 and contrast == 'fat':
                    file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, 'affine_' + pat + '.h5')
                    with h5py.File(file_curr, 'w') as hfcurr:
                        hfcurr.create_dataset('affine', data=affine)
                    if not parallelize:
                        hfs[class_name]['affine/' + pat] = h5py.ExternalLink(file_curr, "/affine")

    if parallelize:
        num_cores = 24
        print(f'using {num_cores} CPU cores')

        t = time.time()
        _ = Parallel(n_jobs=num_cores)(
            delayed(process_file)(f, parallelize) for f in tqdm(pats))
        elapsed_time = time.time() - t

        print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

        # now link the files to the main hdf5 file
        for pat in tqdm(pats):
            process_file(pat, False)

    else:
        for pat in tqdm(pats):
            process_file(pat, parallelize)

    for hf in hfs:
        hf.close()

    return keys


def convert_nifti_h5(input_dir, output_dir, output_file, save_path, parallelize, verbose=False):
    """
    Converts all nifti files in input_dir to h5 files in output_dir
    """

    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    # Get list of all nifti files in input_dir
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen_largest_component_parallel.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]
    #patsseg = list(bounding_boxes_rem['pat'].values)
    pats = list(bounding_boxes_rem['pat'].values)
    pats = [str(k) for k in pats]
    #pats = [pat[:6] + '_' + pat[6:] if len(pat) == 8 else pat for pat in pats]

    # Create list of all h5 files in output_dir
    #h5_file = output_dir.joinpath(output_file)

    #sel_shape = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50], 'pnc': [80, 50, 50]}    #ukb
    sel_shape = {'liv': [185, 150, 70], 'spl': [80, 80, 50], 'rkd': [60, 60, 50], 'lkd': [60, 60, 50], 'pnc': [120, 70, 70]}     #nako
    contrasts = ['fat', 'in', 'opp', 'water']

    keys = []
    files = ['nako_liv_preprocessed.h5', 'nako_spl_preprocessed.h5', 'nako_rkd_preprocessed.h5', 'nako_lkd_preprocessed.h5', 'nako_pnc_preprocessed.h5']
    modes = ['w', 'w', 'w', 'w', 'w']
    for idx, file in enumerate(files):
        if output_dir.joinpath(file).exists():
            Path(output_dir.joinpath(file)).unlink()
            #modes[idx] = 'a'
        output_dir.joinpath(Path(file).stem).mkdir(exist_ok=True)

    hfs = {'liv': h5py.File(output_dir.joinpath('nako_liv_preprocessed.h5'), modes[0]),
           'spl': h5py.File(output_dir.joinpath('nako_spl_preprocessed.h5'), modes[1]),
           'rkd': h5py.File(output_dir.joinpath('nako_rkd_preprocessed.h5'), modes[2]),
           'lkd': h5py.File(output_dir.joinpath('nako_lkd_preprocessed.h5'), modes[3]),
           'pnc': h5py.File(output_dir.joinpath('nako_pnc_preprocessed.h5'), modes[4])}
    #hfs = {'liv': h5py.File(output_dir.joinpath('nako_liv_preprocessed.h5'), modes[0])}

    def process_file(pat, parallelize):
        #print(pat)
        for contrast in contrasts:
            # check existence in file
            exists_all = []
            for iclass in np.arange(1, 6):
            #for iclass in [1]:          # only liver TODO: change back  
                if iclass == 1:
                    class_name = 'liv'
                elif iclass == 2:
                    class_name = 'spl'
                elif iclass == 3:
                    class_name = 'rkd'
                elif iclass == 4:
                    class_name = 'lkd'
                elif iclass == 5:
                    class_name = 'pnc'
                else:
                    raise ValueError('Class not recognized')
                #if modes[iclass-1] == 'a' and contrast + '/' + pat in hfs[class_name]:
                if output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5').exists():
                    exists_all.append(True)
                else:
                    exists_all.append(False)
            if np.all(exists_all):
                doload = False
            else:
                doload = True

            if doload:
                if not os.path.exists(input_dir.joinpath(pat)):
                    fname = pat + '_30'
                else:
                    fname = pat
                if os.path.isdir(input_dir.joinpath(fname, contrast)):
                    nifti_file = input_dir.joinpath(fname, contrast, contrast + '.nii.gz')
                else:
                    nifti_file = input_dir.joinpath(fname, contrast + '.nii.gz')
                #print(f'loading {nifti_file}')
                img = nib.load(nifti_file)
                img_data = img.get_fdata().astype(np.float32)
                affine = img.affine.astype(np.float16)
                keyh5 = nifti_file.stem.split('.')[0]

            for iclass in np.arange(1, 6):
            #for iclass in [1]:          # only liver TODO: change back
                #if exists_all[iclass-1]:
                #    continue
                if iclass == 1:
                    class_name = 'liv'
                elif iclass == 2:
                    class_name = 'spl'
                elif iclass == 3:
                    class_name = 'rkd'
                elif iclass == 4:
                    class_name = 'lkd'
                elif iclass == 5:
                    class_name = 'pnc'
                else:
                    raise ValueError('Class not recognized')
                if exists_all[iclass-1]:
                    # only link them
                    file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, contrast + '_' + pat + '.h5')
                    if not parallelize:
                        hfs[class_name][contrast + '/' + pat] = h5py.ExternalLink(file_curr, "/image")
                    if iclass == 1 and contrast == 'fat':
                        file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, 'affine_' + pat + '.h5')
                        if not parallelize:
                            hfs[class_name]['affine/' + pat] = h5py.ExternalLink(file_curr, "/affine")
                    continue

                sel_shape_curr = sel_shape[class_name]

                box = bounding_boxes.loc[bounding_boxes['pat'] == int(pat)][class_name].values
                box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
                center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
                center = [int(x) for x in center]

                # list(box[0:2] + np.floor(np.asarry(sel_shape[0:2]) / 2)) + list(np.floor(np.asarry(box[5] + box[2])/2)) +

                # crop image around segmentation mask
                img_crop = crop(img_data, sel_shape_curr, center)

                #save_path_curr = Path(save_path).joinpath(key + keyh5.split('_')[1])
                #Path(save_path_curr).mkdir(exist_ok=True)

                # Write to h5 file
                #hfs[class_name].create_dataset(contrast + '/' + pat, data=img_crop)
                #if iclass == 1 and contrast == 'fat':
                #    hfs[class_name].create_dataset('affine/' + pat, data=affine)
                file_curr = output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5')
                with h5py.File(file_curr, 'w') as hfcurr:
                    hfcurr.create_dataset('image', data=img_crop)
                if not parallelize:
                    hfs[class_name][contrast + '/' + pat] = h5py.ExternalLink(file_curr, "/image")

                if iclass == 1 and contrast == 'fat':
                    file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, 'affine_' + pat + '.h5')
                    with h5py.File(file_curr, 'w') as hfcurr:
                        hfcurr.create_dataset('affine', data=affine)
                    if not parallelize:
                        hfs[class_name]['affine/' + pat] = h5py.ExternalLink(file_curr, "/affine")

    if parallelize:
        num_cores = 24
        print(f'using {num_cores} CPU cores')

        t = time.time()
        _ = Parallel(n_jobs=num_cores)(
            delayed(process_file)(f, parallelize) for f in tqdm(pats))
        elapsed_time = time.time() - t

        print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

        # now link the files to the main hdf5 file
        for pat in tqdm(pats):
            process_file(pat, False, False)

    else:
        for pat in tqdm(pats):
            process_file(pat, parallelize)

    for hf in hfs:
        hf.close()

    return keys


def combine_kidney_h5(input_dir, output_dir, output_file, save_path, verbose=False):
    # Combine left and right kidney h5 files
    hdf5file_lkd = h5py.File(Path(output_dir).joinpath('nako_lkd_preprocessed_masked.h5'), 'r')
    pats = list(hdf5file_lkd['water'].keys())
    print(pats[:5])

    # grp_image = hdf5file.create_group('image')

    """keys_kidney_train = [l.strip() for l in Path(output_dir).joinpath('keys', 'train_kidneys_mainly_healthy.dat').open().readlines()]
    keys_kidney_train = [k + '/left' for k in keys_kidney_train] + [k + '/right' for k in keys_kidney_train]
    with open(output_dir.joinpath('keys', 'train_kidneys_mainly_healthy_combined.dat'), 'w') as f:
        for key in keys_kidney_train:
            f.write(key + '\n')

    keys_kidney_test = [l.strip() for l in Path(output_dir).joinpath('keys', 'test_kidneys_mainly_healthy.dat').open().readlines()]
    keys_kidney_test = [k + '/left' for k in keys_kidney_test] + [k + '/right' for k in keys_kidney_test]
    with open(output_dir.joinpath('keys', 'test_kidneys_mainly_healthy_combined.dat'), 'w') as f:
        for key in keys_kidney_test:
            f.write(key + '\n')

    keys_kidney_train = [l.strip() for l in Path(output_dir).joinpath('keys', 'full_test_kidneys_mainly_healthy.dat').open().readlines()]
    keys_kidney_train = [k + '/left' for k in keys_kidney_train] + [k + '/right' for k in keys_kidney_train]
    with open(output_dir.joinpath('keys', 'full_test_kidneys_mainly_healthy_combined.dat'), 'w') as f:
        for key in keys_kidney_train:
                f.write(key + '\n')"""

    contrasts = ['fat', 'in', 'opp', 'water']
    hdf5file = h5py.File(Path(output_dir).joinpath('nako_kidney_preprocessed_masked.h5'), 'w')
    for pat in pats:
        for contrast in contrasts:
            #keyh5 = "_".join(pat.split('_')[0:2])
            hdf5file["/" + contrast + "/" + pat + "/left"] = h5py.ExternalLink(Path(output_dir).joinpath('nako_lkd_preprocessed_masked.h5'), "/" + contrast + "/" + pat)
            hdf5file["/" + contrast + "/" + pat + "/right"] = h5py.ExternalLink(Path(output_dir).joinpath('nako_rkd_preprocessed_masked.h5'), "/" + contrast + "/" + pat)
    hdf5file.close()


def signalhandler(signum, frame):
    #print("Error in loading file!")
    raise Exception("Error in loading file!")

def check_files(input_dir, output_dir, output_file, save_path, parallelize, verbose=False):
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]
    # patsseg = list(bounding_boxes_rem['pat'].values)
    pats = list(bounding_boxes_rem['pat'].values)
    pats = [str(k) for k in pats]

    sel_shape = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50],
                 'pnc': [80, 50, 50]}
    contrasts = ['fat', 'in', 'opp', 'water']
    keys = []
    files = ['nako_liv_preprocessed.h5', 'nako_spl_preprocessed.h5', 'nako_rkd_preprocessed.h5', 'nako_lkd_preprocessed.h5',
             'nako_pnc_preprocessed.h5']

    iCnt = 0
    signal.signal(signal.SIGALRM, signalhandler)
    for pat in tqdm(pats):
        #if pat.name not in patsseg:
        #    continue
        for contrast in contrasts:
            # check existence in file
            exists_all = []
            for iclass in np.arange(1, 6):
                if iclass == 1:
                    class_name = 'liv'
                elif iclass == 2:
                    class_name = 'spl'
                elif iclass == 3:
                    class_name = 'rkd'
                elif iclass == 4:
                    class_name = 'lkd'
                elif iclass == 5:
                    class_name = 'pnc'
                else:
                    raise ValueError('Class not recognized')
                #if modes[iclass-1] == 'a' and contrast + '/' + pat in hfs[class_name]:
                if output_dir.joinpath(Path(files[iclass-1]).stem, contrast + '_' + pat + '.h5').exists():
                    exists_all.append(True)
                else:
                    exists_all.append(False)
            if np.all(exists_all):
                continue

            signal.alarm(30)
            try:
                nifti_file = input_dir.joinpath(pat, contrast + '.nii.gz')
                img = nib.load(nifti_file)
                img_data = img.get_fdata().astype(np.float32)
            except Exception as exc:
                print(exc)
                print('Pat[' + str(iCnt) + ']: ' + str(pat))
            signal.alarm(0)
        iCnt += 1

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    l.sort(key=alphanum_key)

def is_stitching_correct(subject_dir):
    sub_exists = os.path.isdir(subject_dir)
    wat_exists = os.path.isfile(os.path.join(subject_dir, 'water.nii.gz'))
    inp_exists = os.path.isfile(os.path.join(subject_dir, 'in.nii.gz'))
    opp_exists = os.path.isfile(os.path.join(subject_dir, 'opp.nii.gz'))
    fat_exists = os.path.isfile(os.path.join(subject_dir, 'fat.nii.gz'))
    return sub_exists and wat_exists and inp_exists and opp_exists and fat_exists

def rename_and_filter_files(subject_dir):
    files = glob.glob(os.path.join(subject_dir, '*.nii.gz'))
    for f in files:
        f_base = os.path.basename(f)
        if f_base == 'T1_water.nii.gz':
            os.rename(f, os.path.join(subject_dir, 'water.nii.gz'))
        elif f_base == 'T1_opp.nii.gz':
            os.rename(f, os.path.join(subject_dir, 'opp.nii.gz'))
        elif f_base == 'T1_in.nii.gz':
            os.rename(f, os.path.join(subject_dir, 'in.nii.gz'))
        elif f_base == 'T1_fat.nii.gz':
            os.rename(f, os.path.join(subject_dir, 'fat.nii.gz'))
    if not is_stitching_correct(subject_dir):
        shutil.rmtree(subject_dir)

def redo_dicom2nifti(input_dir, output_dir, output_file, save_path):
    # get patient by number
    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]
    # patsseg = list(bounding_boxes_rem['pat'].values)
    pats = list(bounding_boxes_rem['pat'].values)
    pats = [str(k) for k in pats]
    # from run of check_files() and
    # cp logfile error_pats.txt
    # sed '1~2d' error_pats.txt | sed 's/^.\{4\}//' | sed 's/.\{10\}$//' > error_pats.dat
    #with open('../experiments/error_pats.dat') as file:  # indexes of patients with errors
    #    lines = file.readlines()
    #    lines = [line.rstrip() for line in lines]
    #pat_idx = [int(p) for p in lines]
    #pat_idx = list(set(pat_idx))  # unique entries
    #pat_sels = [pats[idx] for idx in pat_idx]  # patient index
    # from:
    # cd /mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw
    # find . -type f -newermt '29 june 2022' -print | sed 's/.\{11\}//' | sed 's/^.\{2\}//' > ~/Documents/nako_ukb_age/experiments/redo_pats.dat
    with open('../experiments/redo_pats.dat') as file:  # folder names
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    pat_idx = [int(p) for p in lines]
    pat_idx = list(set(pat_idx))  # unique entries
    pat_sels = [str(p) for p in pat_idx]
    pat_sels = ['1876508']

    for pat_sel in tqdm(pat_sels):
        redo_pat(pat_sel, input_dir, output_dir, output_file, save_path)

def redo_pat(pat_sel, input_dir, output_dir, output_file, save_path):
    # check if broken
    # imgtmp = nib.load(os.path.join(input_dir, pat_sel, 'fat.nii.gz')).get_fdata()
    if os.path.exists(os.path.join(input_dir, pat_sel, 'fat.nii.gz')):
        os.remove(os.path.join(input_dir, pat_sel, 'fat.nii.gz'))
        os.remove(os.path.join(input_dir, pat_sel, 'water.nii.gz'))
        os.remove(os.path.join(input_dir, pat_sel, 'in.nii.gz'))
        os.remove(os.path.join(input_dir, pat_sel, 'opp.nii.gz'))

    f = '/mnt/qdata/share/rafruem1/ukb/MRI/raw/abdominalMri/' + pat_sel + '_20201_2_0.zip'  # patient zip file
    if not os.path.exists(f):
        f = '/mnt/qdata/share/rafruem1/ukb/MRI/raw/abdominalMri/' + pat_sel + '_20201_3_0.zip'  # patient zip file
    tmp = tempfile.TemporaryDirectory()
    unzip(f, tmp.name)

    tmp2 = tempfile.TemporaryDirectory()
    #dest_dir = '/home/rakuest1/tmp/ukbnifti'
    #Path(dest_dir).mkdir(exist_ok=True)

    dicom2nifti.convert_directory(tmp.name, tmp2.name)
    subject_dir = tmp2.name
    nii_files = [name for name in os.listdir(subject_dir)
                 if os.path.isfile(os.path.join(subject_dir, name))]
    sort_nicely(nii_files)
    tool = '/home/rakuest1/Documents/nako_ukb_age/preprocess/stitching/build/stitching/stitching'
    out_fnames = ['in.nii.gz', 'opp.nii.gz', 'fat.nii.gz', 'water.nii.gz', ]
    margin = 3
    if len(nii_files) >= 24:
        for k in range(4):
            output_image = os.path.join(subject_dir, out_fnames[k])
            input_images = os.path.join(subject_dir, nii_files[k])
            for f in range(1, 6):
                input_images += ' ' + os.path.join(subject_dir, nii_files[k + f * 4])
            command = ' '.join((tool, '-a -m', str(margin), '-i', input_images, '-o', output_image))
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output)
    else:
        print('insufficient stations...')

    # moving files
    for k in range(4):
        shutil.copyfile(os.path.join(subject_dir, out_fnames[k]), os.path.join(input_dir, pat_sel, out_fnames[k]))

    tmp.cleanup()
    tmp2.cleanup()
    #tmppath = os.path.join(input_dir, pat_sel, 'fat.nii.gz')
    #print(f'Test it out: imgtmp = nib.load({tmppath}).get_fdata()')

def redo_segmentation(input_dir, output_dir, output_file, save_path):
    with open('../experiments/redo_pats.dat') as file:  # folder names
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    pat_idx = [int(p) for p in lines]
    pat_idx = list(set(pat_idx))  # unique entries
    pat_sels = [str(p) for p in pat_idx]
    out_fnames = ['in.nii.gz', 'opp.nii.gz', 'fat.nii.gz', 'water.nii.gz', ]

    for pat_sel in tqdm(pat_sels):
        Path(output_dir).joinpath(pat_sel).mkdir(exist_ok=True)
        for out_fname in out_fnames:
            Path(output_dir).joinpath(pat_sel, out_fname).symlink_to(Path(input_dir).joinpath(pat_sel, out_fname))

def redo_nifti_to_h5(input_dir, output_dir, output_file, save_path):
    pats = ['1524207']
    iclass = 3

    bounding_boxes = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen.csv'))
    bounding_boxes_rem = bounding_boxes.loc[(bounding_boxes['liv'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['spl'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['rkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['lkd'] != '[-1, -1, -1, -1, -1, -1]') &
                                            (bounding_boxes['pnc'] != '[-1, -1, -1, -1, -1, -1]')]
    sel_shape = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50], 'pnc': [80, 50, 50]}
    contrasts = ['fat', 'in', 'opp', 'water']
    files = ['nako_liv_preprocessed.h5', 'nako_spl_preprocessed.h5', 'nako_rkd_preprocessed.h5', 'nako_lkd_preprocessed.h5',
             'nako_pnc_preprocessed.h5']
    if iclass == 1:
        class_name = 'liv'
    elif iclass == 2:
        class_name = 'spl'
    elif iclass == 3:
        class_name = 'rkd'
    elif iclass == 4:
        class_name = 'lkd'
    elif iclass == 5:
        class_name = 'pnc'
    
    for pat in tqdm(pats):
        for contrast in contrasts:
            nifti_file = input_dir.joinpath(pat, contrast + '.nii.gz')
            img = nib.load(nifti_file)
            img_data = img.get_fdata().astype(np.float32)
            affine = img.affine.astype(np.float16)
            keyh5 = nifti_file.stem.split('.')[0]
    
            sel_shape_curr = sel_shape[class_name]
    
            box = bounding_boxes.loc[bounding_boxes['pat'] == int(pat)][class_name].values
            box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
            center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
            center = [int(x) for x in center]
    
            img_crop = crop(img_data, sel_shape_curr, center)
    
            file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, contrast + '_' + pat + '.h5')
            with h5py.File(file_curr, 'w') as hfcurr:
                hfcurr.create_dataset('image', data=img_crop)
    
            if iclass == 1 and contrast == 'fat':
                file_curr = output_dir.joinpath(Path(files[iclass - 1]).stem, 'affine_' + pat + '.h5')
                with h5py.File(file_curr, 'w') as hfcurr:
                    hfcurr.create_dataset('affine', data=affine)
        
def create_csv(keys, csv_input, csv_output, verbose=False):
    # sex: 0=female, 1=male
    csv_in = pd.read_csv(csv_input, low_memory=False) # nrows=100000, cut aways a few corrupted lines at the end, header=0, names=['eid', '21022-0.0', '31-0.0', '21002-0.0', '50-0.0'])  # age, sex, weight, height
    df_sel = csv_in[['eid', '21022-0.0', '31-0.0', '21002-0.0', '50-0.0']]
    # find header
    #csv_in_info = pd.read_csv(csv_input, nrows=10)
    #cols = [col for col in csv_in_info.columns if '12144' in col]

    df = df_sel.rename(columns={'eid': 'key', '21022-0.0': 'age', '31-0.0': 'sex', '21002-0.0': 'weight', '50-0.0': 'height'})
    keys_int = [int(k) for k in keys]
    df = df[df['key'].isin(keys_int)]  # filter out only patients with imaging data
    df.to_csv(Path(csv_output))

def create_keys(keys, output_dir, n_folds=5):
    # 80% / 20 % split for train / test
    train_set, test_set = train_test_split(keys, test_size=0.2, random_state=42)
    train_folds = []
    test_folds = []
    for train_index, test_index in KFold(n_splits=n_folds).split(keys):
        train_folds.append([keys[idx] for idx in train_index])
        test_folds.append([keys[idx] for idx in test_index])

    with open(output_dir.joinpath('keys', 'train.dat'), 'w') as f:
        for item in train_set:
            f.write("%s\n" % item)

    with open(output_dir.joinpath('keys', 'test.dat'), 'w') as f:
        for item in test_set:
            f.write("%s\n" % item)

    for i in range(n_folds):
        with open(output_dir.joinpath('keys', 'train{}.dat'.format(i)), 'w') as f:
            for item in train_folds[i]:
                f.write("%s\n" % item)
        with open(output_dir.joinpath('keys', 'test{}.dat'.format(i)), 'w') as f:
            for item in test_folds[i]:
                f.write("%s\n" % item)


def get_bounding_boxes(save_path, redo_pat=False, largest_component=True):
    # classes {'0': 'background', '1': 'liv', '2': 'spl', '3': 'rkd', '4': 'lkd', '5': 'pnc'}
    def process_file(pat, save_path, redo_pat, largest_component):
        bounding_boxes = []
        pred = nib.load(Path(save_path).joinpath(pat, 'prd.nii.gz')).get_fdata()

        for iclass in np.arange(1, 6):
            if iclass == 1:
                class_name = 'liv'
            elif iclass == 2:
                class_name = 'spl'
            elif iclass == 3:
                class_name = 'rkd'
            elif iclass == 4:
                class_name = 'lkd'
            elif iclass == 5:
                class_name = 'pnc'
            else:
                raise ValueError('Class not recognized')

            try:
                if largest_component:
                    pred_mask = np.asarray(pred == iclass, dtype=int)
                    labeled_mask, num_features = label(pred_mask)

                    # If no features are found, return the original mask
                    if num_features != 0:                   
                        # Find the largest connected component
                        largest_component = 0
                        largest_size = 0
                        for i in range(1, num_features + 1):
                            component_size = np.sum(labeled_mask == i)
                            if component_size > largest_size:
                                largest_size = component_size
                                largest_component = i
                        
                        # Create a new mask with only the largest connected component
                        pred_mask = (labeled_mask == largest_component).astype(int)
                    bounding_box = regionprops(pred_mask)[0].bbox
                else:
                    bounding_box = regionprops(np.asarray(pred == iclass, dtype=int))[0].bbox
            except:
                bounding_box = [-1, -1, -1, -1, -1, -1]
                print('Error in patient {}'.format(pat))
            bounding_boxes.append(bounding_box)
        #bounding_boxes['pat'].append(pat)
        pat = pat[:6] if len(pat) != 6 else pat 
        with open(Path(save_path).joinpath('bounding_boxes_abdomen_largest_component_parallel.csv'), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[pat] + bounding_boxes])

    if redo_pat:
        with open('../experiments/redo_pats.dat') as file:  # folder names
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        pat_idx = [int(p) for p in lines]
        pat_idx = list(set(pat_idx))  # unique entries
        pat_redo = [str(p) for p in pat_idx]
        bounding_boxes_prev = pd.read_csv(Path(save_path).joinpath('bounding_boxes_abdomen.csv'))

    pat_list = [p.name for p in Path(save_path).iterdir() if
                      p.is_dir() and ('prd.nii.gz' in os.listdir(Path(save_path).joinpath(p)))]
    #pat_list = os.listdir(save_path)    #ToDo change back
    
    with open(Path(save_path).joinpath('bounding_boxes_abdomen_largest_component_parallel.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pat', 'liv', 'spl', 'rkd', 'lkd', 'pnc'])
    # multiprocessing
    num_cores = 24
    print(f'using {num_cores} CPU cores')

    t = time.time()
        
    results = Parallel(n_jobs=num_cores)(delayed(process_file)(pat, save_path, redo_pat, largest_component) for pat in pat_list)    

    #df = pd.DataFrame(bounding_boxes)
    #df.to_csv(Path(save_path).joinpath('bounding_boxes_abdomen_largest_component_parallel.csv'), index=False)

    print('done bounding boxes')


def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for NAKO abdomen MRI data.\n' \
                                                 'CSV creation\n' \
                                                 'Nifti to HDF5 conversion\n'\
                                                 'Key creation for train, test, val')
    parser.add_argument('input_dir', help='Input directory of all nifti files (*.nii.gz)')
    parser.add_argument('output_dir', help='Output directory for all files', default='/mnt/qdata/share/raeckev1/nako_30k/interim/')
    parser.add_argument('--output_file', help='Output h5 file to store processed files.', default='nako_abdomen_preprocessed.h5')
    parser.add_argument('--csv_input', help='Input CSV file', default='/home/raeckev1/nako_ukb_age/preprocess/NAKO_706_patient_data_abdominal.csv')
    parser.add_argument('--csv_output', help='Output CSV file', default='nako_abdomen.csv')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Create output directory if it does not exist
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    save_path = Path('/mnt/qdata/share/raeckev1/nako_30k/abdominal_MRI/seg/')  # for segmentations

    output_dir.joinpath('keys').mkdir(exist_ok=True)

    #update_external_links('/mnt/qdata/share/raecker1/ukb_liv_preprocessed.h5', 'test', 'test_str')
    #keys = convert_nifti_h5(input_dir, output_dir, args.output_file, args.verbose)
    #keys = write_keys(input_dir, output_dir, args.output_file, args.verbose)
    #create_csv(keys, args.csv_input, output_dir.joinpath(args.csv_output), args.verbose)
    #create_keys(keys, output_dir, n_folds=5)
    #get_bounding_boxes(save_path)
    #inspect_boxes_segmentation(input_dir, output_dir, args.output_file, save_path, args.verbose)
    #convert_nifti_h5(input_dir, output_dir, args.output_file, save_path, False, args.verbose)
    #convert_nifti_h5_masked_parallel(input_dir, output_dir, args.output_file, save_path, False, args.verbose)
    combine_kidney_h5(input_dir, output_dir, args.output_file, save_path, args.verbose)
    #check_files(input_dir, output_dir, args.output_file, save_path, False, args.verbose)
    #redo_dicom2nifti(input_dir, output_dir, args.output_file, save_path)
    #redo_segmentation(input_dir, output_dir, args.output_file, save_path)
    #redo_nifti_to_h5(input_dir, output_dir, args.output_file, save_path)
    #rewrite_keys(input_dir, output_dir, args.output_file, save_path, args.verbose)
    #convert_masks_h5(save_path, output_dir, args.output_file, save_path, False, args.verbose)

    #file_path = '/mnt/qdata/share/raecker1/nako_30k/interim/nako_pnc_preprocessed.h5'
    #update_h5_links(file_path)


if __name__ == '__main__':
    # python3 ukbabdomen.py /mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/raw/ /mnt/qdata/share/rakuest1/data/UKB/interim/
    main()
