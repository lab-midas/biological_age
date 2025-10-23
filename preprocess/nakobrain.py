import os
import h5py
from tqdm import tqdm
import numpy as np
import time
import subprocess
import argparse
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
import intensity_normalization
from nipype.interfaces import fsl


def write_keys(key_path, keys):

    with open(key_path, 'w') as f:
        for key in keys:
            f.write(key + '\n')
    print(f'Keys written to {key_path}')


def convert_nifti_h5(input_dir, h5_dir, key_path, output_file, verbose=False):
    """
    Converts all nifti files in input_dir to h5 files in output_dir
    Args:
        input_dir (Path): input directory with nifti files
        h5_dir (Path): output directory to store h5 file
        key_path (Path): output dat file to store keys
        output_file (str): output h5 file name
        verbose (bool): print progress
    """

    # Create output directory if it does not exist
    h5_dir.mkdir(exist_ok=True)

    # Get list of all nifti files in input_dir
    nifti_files = [f for f in input_dir.glob('*')]

    # Create list of all h5 files in output_dir
    h5_path = h5_dir.joinpath(output_file)

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
        file_key = nifti_file.stem

        # Load image
        img = nib.load(nifti_file.joinpath(nifti_file, 'n4_flirt_robex_fcm', file_key + '_n4_flirt_fcmnorm.nii.gz'))
        img_data = img.get_fdata().astype(np.float32)
        # Load mask and apply to image
        mask = nib.load(nifti_file.joinpath(nifti_file, 'n4_flirt_robex_fcm', file_key + '_n4_flirt_robexmask.nii.gz'))
        mask_data = mask.get_fdata().astype(np.float32)

        img_data = np.multiply(img_data, mask_data)

        affine = img.affine.astype(np.float16)
        keyh5 = file_key.split('_')[0]

        # Write to h5 file
        try:
            grp_image.create_dataset(keyh5, data=img_data)
            grp_affine.create_dataset(keyh5, data=affine)
        except ValueError:
            print(f'key {keyh5} not unique')
            continue

        key = keyh5.split('_')[0]
        keys.append(key)

        img = None
        img_data = None
        affine = None
    hf.close()

    print(f'Wrote {len(keys)} nifti files to {h5_path}.')
    
    keys = list(set(keys))
    write_keys(key_path, keys)


def n4_bias_field_correction(input_file,
                             output_file,
                             mask_file=None,
                             number_iterations=[50, 50, 50, 50]):
    """ N4 bias field correction for brain mri.

    Additional information:
    https://simpleitk.readthedocs.io/en/latest/Examples/N4BiasFieldCorrection/Documentation.html
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855/
    Args:
        input_file (str/Path): nii input file
        output_file (str/Path): nii output file
        mask_file (str/Path, optional): path to store nii mask. Defaults to None.
        number_iterations (list, optional): Iterations per level. Defaults to [50,50,50].
    """

    input_img = sitk.ReadImage(str(input_file))
    mask_image = sitk.OtsuThreshold(input_img, 0, 1, 200)
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(number_iterations)
    output_img = corrector.Execute(input_img, mask_image)
    sitk.WriteImage(output_img, str(output_file))
    if mask_file:
        sitk.WriteImage(mask_image, str(mask_file))


def flirt_registration(input_file,
                       output_file,
                       matrix_file,
                       ref_file,
                       verbose=False):
    """Registration to MNI template using FSL-FLIRT

    Ubuntu: Create link for flirt
    sudo ln -s /usr/bin/fsl5.0-flirt /usr/bin/flirt
    Args:
        input_file (str/Path): input file (nii.gz)
        output_file (str/Path): output file (nii.gz)
        matrix_file (str/Path): transformation matrix (.mat)
        ref_file (str/Path): path to MNI-152-1mm reference file (nii.gz)
        verbose (bool): print fsl command
    """
    flt = fsl.FLIRT(bins=256, cost_func='mutualinfo')
    flt.inputs.in_file = str(input_file)
    flt.inputs.reference = str(ref_file)
    flt.inputs.output_type = "NIFTI_GZ"
    if not verbose:
        flt.inputs.verbose = 0
    flt.inputs.out_file = str(output_file)
    flt.inputs.out_matrix_file = str(matrix_file)
    if verbose:
        print(flt.cmdline)
    flt.run()


def robex_skull_stripping(input_file,
                          output_file,
                          mask_file,
                          robex_dir,
                          verbose):
    """Skull stripping using ROBEX

    Create stripped output (nii.gz) and brain mask (nii.gz).
    Publication: http://pages.ucsd.edu/~ztu/publication/TMI11_ROBEX.pdf
    Download and extract ROBEX 1.3 from https://www.nitrc.org/projects/robex
    Args:
        input_file (str/Path): input file (nii.gz)
        output_file (str/Path): output file (nii.gz)
        mask_file (str/Path): output mask file (nii.gz)
        robex_dir (str/Path):  extracted ROBEX archive
        verbose (bool): print stdout from ROBEX
    """
    robex_dir = Path(robex_dir)
    out = subprocess.Popen([str(robex_dir.joinpath('ROBEX')),
                            str(input_file),
                            str(output_file),
                            str(mask_file)],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,
                           cwd=str(robex_dir))
    stdout, stderr = out.communicate()
    if verbose:
        print(stdout)
        print(stderr)
    return


def fcm_normalize(input_file,
                  mask_file,
                  output_file,
                  output_mask):
    """Fuzzy C-means (FCM)-segmentation-based white matter (WM) mean normalization

    Computes normalized image and the wm mask.
    Paper: https://arxiv.org/abs/1812.04652
    Install the following package (git clone + pip install ./)
    https://github.com/jcreinhold/intensity-normalization
    Requirements have to be installed manually!
    (matplotlib, numpy, nibabel, scikit-fuzzy, scikit-learn, scipy, statsmodel)
    Args:
        input_file (str/Path): input file (nii.gz)
        mask_file (str/Path):  brain mask (nii.gz) file (e.g. created by ROBEX)
        output_file (str/Path): path to normalized output (nii.gz)
        output_mask (str/Path): path to wm mask (nii.gz)
    """

    img = nib.load(str(input_file))
    brain_mask = nib.load(str(mask_file))

    wm_mask = intensity_normalization.normalize.fcm.find_tissue_mask(img, brain_mask)
    normalized = intensity_normalization.normalize.fcm.fcm_normalize(img, wm_mask)

    nib.save(wm_mask, str(output_mask))
    nib.save(normalized, str(output_file))

    return output_file, output_mask


def process_brain_t1(input_file,
                     output_dir,
                     reference_file='/mnt/qdata/software/fsl/ref/MNI152_T1_1mm.nii.gz',
                     robex_dir='/mnt/qdata/software/robex',
                     split=0,
                     verbose=False):
    """Preprocessing pipeline for brain T1 MRI data.
    Steps:
    1) N4 bias field correction
    2) FLIRT MNI152 coregistration
    3) ROBEX skull stripping
    4) FCM WM intensity normalization
    Args:
        input_file (str/Path): input file (nii.gz)
        output_dir (str/Path): output directory to store processed files.
        reference_file (str/Path, optional): MNI152-1mm reference .nii.gz file. Defaults to '/mnt/qdata/software/fsl/ref/MNI152_T1_1mm.nii.gz'.
        robex_dir (str/Path, optional): ROBEX installation directory. Defaults to '/mnt/qdata/software/robex'.
        split (int, optional): (<0) process full pipeline, (0) only bias field corrections, or (1) full pipeline except bias field corrections.
        verbose (bool, optional): print progress. Defaults to False.
    """
    
    print(split)
    input_file = Path(input_file)
    reference_file = Path(reference_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # create output directories
    output_dir.joinpath('raw').mkdir(exist_ok=True)
    # output_dir.joinpath('n4').mkdir(exist_ok=True) tmp
    output_dir.joinpath('n4_flirt').mkdir(exist_ok=True)
    # output_dir.joinpath('n4_flirt_robex').mkdir(exist_ok=True)
    output_dir.joinpath('n4_flirt_robex_fcm').mkdir(exist_ok=True)  # and robex mask, delete robex output

    # start timer
    t = time.time()

    # bias field correction
    print('(n4) bias field correction ...')
    n4_file = output_dir.joinpath(
        'n4_flirt', input_file.name.replace('.nii.gz', '_n4.nii.gz'))

    if split <= 0:
        n4_bias_field_correction(input_file, n4_file)

    if split < 0 or split == 1:
        # mni coregistration
        print('(fsl-flirt) mni template registration ...')
        input_file = n4_file
        n4_flirt_file = output_dir.joinpath(
            'n4_flirt', input_file.name.replace('.nii.gz', '_flirt.nii.gz'))
        n4_flirt_matrix = output_dir.joinpath(
            'n4_flirt', input_file.name.replace('.nii.gz', '_flirt.mat'))
        flirt_registration(input_file, n4_flirt_file,
                           n4_flirt_matrix, reference_file, verbose=verbose)

        # robex skull stripping
        print('(robex) skull stripping ...')
        input_file = n4_flirt_file
        n4_flirt_robex_file = output_dir.joinpath('n4_flirt_robex_fcm',
                                                  input_file.name.replace('.nii.gz', '_robex.nii.gz'))
        n4_flirt_robex_mask = output_dir.joinpath('n4_flirt_robex_fcm',
                                                  input_file.name.replace('.nii.gz', '_robexmask.nii.gz'))
        robex_skull_stripping(input_file,
                              n4_flirt_robex_file,
                              n4_flirt_robex_mask,
                              robex_dir,
                              verbose=verbose)
        # delete stripped mri (can be produced later on, using the dilated mask)
        n4_flirt_robex_file.unlink()

        print('(fcm) intensity normalization')
        input_file = n4_flirt_file
        mask_file = n4_flirt_robex_mask
        n4_flirt_fcmwmmask = str(output_dir.joinpath('n4_flirt_robex_fcm',
                                                     input_file.name.replace('.nii.gz', '_fcmwmmask.nii.gz')))
        n4_flirt_fcmnorm = str(output_dir.joinpath('n4_flirt_robex_fcm',
                                                   input_file.name.replace('.nii.gz', '_fcmnorm.nii.gz')))
        fcm_normalize(input_file,
                      mask_file,
                      n4_flirt_fcmnorm,
                      n4_flirt_fcmwmmask)

    # stop timer
    elapsed_time = time.time() - t
    if verbose:
        print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for NAKO T1 brain MRI data.\n' \
                                                 'N4 bias field correction\n' \
                                                 'FLIRT MNI152 coregistration\n' \
                                                 'ROBEX skull stripping\n' \
                                                 'FCM WM intensity normalization\n' \
                                                 'Converts nifti files to h5 file.')
    parser.add_argument('input_dir', help='Input directory with files (T1w brain MRI, .nii.gz)')
    parser.add_argument('output_dir', help='Output directory to store processed files.')
    parser.add_argument('--h5_dir', help='Directory to store h5 file', default='/mnt/qdata/share/raecker1/nako_30k/interim/')
    parser.add_argument('--h5_file', help='Output h5 file to store processed files.', default='nako_brain_preprocessed.h5')
    parser.add_argument('--key_file', help='Output csv file to store keys.', default='brain_imaging.dat')
    parser.add_argument('--reference', help='MNI152-1mm reference .nii.gz file', default='/mnt/qdata/software/fsl/ref/MNI152_T1_1mm.nii.gz')
    parser.add_argument('--robex', help='ROBEX installation directory.', default='/mnt/qdata/software/robex')
    parser.add_argument('--split',
                        help='Split bias field correction (0) from the followings steps (1). Process at once (-1).',
                        type=int,
                        default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    h5_dir = Path(args.h5_dir)
    key_path = h5_dir.joinpath('keys', args.key_file)
    output_file = args.h5_file

    if args.verbose:
        print(Path(args.input_file).name)

    reference_file = args.reference
    robex_dir = args.robex
    split = args.split

    file_list = list(input_dir.glob('*.nii.gz'))

    for file in file_list:
        out_file_dir = output_dir.joinpath(file.stem.split('.')[0])
        process_brain_t1(file,
                        out_file_dir,
                        reference_file=reference_file,
                        robex_dir=robex_dir,
                        split=split,
                        verbose=args.verbose)
    convert_nifti_h5(output_dir, h5_dir, key_path, output_file)


if __name__ == '__main__':
    main()