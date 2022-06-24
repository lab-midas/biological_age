# modified script from b.glocker to convert and stitch single ukbb abdominal scans by sow12
# this version is a modification of sow12's version by t.kart


import sys
import os
import dicom2nifti
import subprocess
import zipfile
import shutil
import glob
from joblib import Parallel, delayed
import time
import re

import pdb

def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def stitch(in_zip_file, out_dir):
    out_fnames = ['T1_in.nii.gz', 'T1_opp.nii.gz', 'T1_fat.nii.gz', 'T1_water.nii.gz',]
    margin = 3

    # assumed the first part of the file name describes the subject ID
    subject_id = os.path.basename(in_zip_file).split('_')[0]
    subject_dir = os.path.join(out_dir, subject_id)
        
    if not os.path.exists(os.path.join(subject_dir, out_fnames[0])):
        # create output dir
        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)
       
        # create dicom subdir 
        dicom_dir = os.path.join(subject_dir, 'dcm')
        if not os.path.exists(dicom_dir):
            os.makedirs(dicom_dir)

        zip_ref = zipfile.ZipFile(in_zip_file, 'r')
        zip_ref.extractall(dicom_dir)
        zip_ref.close()

        dicom2nifti.convert_directory(dicom_dir, subject_dir)
        shutil.rmtree(dicom_dir)

        nii_files = [name for name in os.listdir(subject_dir) 
                if os.path.isfile(os.path.join(subject_dir, name))]
        sort_nicely(nii_files)

        if len(nii_files) >= 24:
            for k in range(4):
                output_image = os.path.join(subject_dir, out_fnames[k])
                input_images = os.path.join(subject_dir, nii_files[k])
                for f in range(1, 6):
                    input_images += ' ' + os.path.join(subject_dir, nii_files[k+f*4])

                command = ' '.join((tool, '-a -m', str(margin), '-i', input_images, '-o', output_image))
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print(output)
        else:
            print('insufficient stations...')

        for file in nii_files:
            os.remove(os.path.join(subject_dir, file))


if __name__ == '__main__':
    # ===============================================================================
    # If you change this part, it should work
    tool = '/home/rakuest1/Documents/nako_ukb_age/preprocess/stitching/build/stitching/stitching'
    # How to run
    # In your terminal, just do:
    # python stitching_script.py /directory_to_your_originally_downloaded_wholebody_scan/ /your_output_directory/
    # ===============================================================================
    zip_dir = sys.argv[1]
    out_dir = sys.argv[2]
    zip_files = glob.glob(os.path.join(zip_dir, '*_20201_2_0.zip'))
    #os.environ['LD_LIBRARY_PATH'] += '/usr/lib/x86_64-linux-gnu/'
    print(f'{len(zip_files)} subjects are found in the directory. Stitching starting...')
    #for file in zip_files:
    #    stitch(file, out_dir)

    def stitch_wrapper(file):
        stitch(file, out_dir)

    # multiprocessing
    num_cores = 10
    #if args.cores:
    #    num_cores = args.cores
    print(f'using {num_cores} CPU cores')

    t = time.time()
    results = Parallel(n_jobs=num_cores)(
        delayed(stitch_wrapper)(f) for f in zip_files)
    elapsed_time = time.time() - t

    print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')





