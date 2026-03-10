import os
from tqdm import tqdm
from pathlib import Path
import h5py
import argparse
import SimpleITK as sitk


def convert_png_h5(input_dir, output_dir, output_file, verbose=False):
    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    # Get list of all png files in input_dir
    png_files = [f for f in input_dir.glob('*.png')]

    # Create list of all h5 files in output_dir
    h5_file = output_dir.joinpath(output_file)

    hf = h5py.File(h5_file, 'w')
    grp_image = hf.create_group('image')
    for png_file in tqdm(png_files):
        keyh5 = png_file.stem.split('.')[0]
        img = sitk.GetArrayFromImage(sitk.ReadImage(png_file.as_posix()))
        # Write to h5 file
        grp_image.create_dataset(keyh5, data=img)

    """keys = list(set(keys))
    with open(output_dir.joinpath('keys_' + Path(output_file).stem.split('.')[0] + '_all.dat'), 'w') as f:
        # pickle.dump(keys, f)
        for key in keys:
            f.write(key + '\n')"""


def combine_h5(input_dir, output_dir, output_file, keys, verbose=False):
    Path(output_dir).mkdir(exist_ok=True)

    h5_left = h5py.File(Path(input_dir).joinpath('LeftFundus.h5'), 'r')
    h5_right = h5py.File(Path(input_dir).joinpath('RightFundus.h5'), 'r')

    h5_file = Path(output_dir).joinpath(output_file)
    hf = h5py.File(h5_file, 'w')
    grp_left = hf.create_group('left')
    grp_right = hf.create_group('right')
    for key in tqdm(keys):
        group_str = key.split('/')[0]
        pat = key.split('/')[1]
        if group_str == 'left':
            if pat + '_21015_0_0' in h5_left['image']:
                loadkey = pat + '_21015_0_0'
                addkey = '_0'
            elif pat + '_21015_0_1' in h5_left['image']:
                loadkey = pat + '_21015_0_1'
                addkey = '_1'
            elif pat + '_21015_1_0' in h5_left['image']:
                loadkey = pat + '_21015_1_0'
                addkey = '_2'
            elif pat + '_21015_1_1' in h5_left['image']:
                loadkey = pat + '_21015_1_1'
                addkey = '_3'
            try:
                grp_left.create_dataset(pat + addkey, data=h5_left['image'][loadkey])
            except:
                print('Patient {} not found in left fundus'.format(pat))
        else:
            if pat + '_21016_0_0' in h5_right['image']:
                loadkey = pat + '_21016_0_0'
                addkey = '_0'
            elif pat + '_21016_0_1' in h5_right['image']:
                loadkey = pat + '_21016_0_1'
                addkey = '_1'
            elif pat + '_21016_1_0' in h5_right['image']:
                loadkey = pat + '_21016_1_0'
                addkey = '_2'
            elif pat + '_21016_1_1' in h5_right['image']:
                loadkey = pat + '_21016_1_1'
                addkey = '_3'
            try:
                grp_right.create_dataset(pat + addkey, data=h5_right['image'][loadkey])
            except:
                print('Patient {} not found in right fundus'.format(pat))

    h5_left.close()
    h5_right.close()
    hf.close()


def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for UK Biobank fundus data.\n' \
                                                 'CSV creation\n' \
                                                 'PNG to HDF5 conversion\n' \
                                                 'Key creation for train, test, val')
    parser.add_argument('--in_left', help='Input directory of all PNG files of the left fundus (*.png)')
    parser.add_argument('--in_right', help='Input directory of all PNG files of the right fundus (*.png)')
    parser.add_argument('--out_dir', help='Output directory for all files',
                        default='/mnt/qdata/share/raecker1/ukbdata_70k/interim/')
    parser.add_argument('--output_file', help='Output h5 file to store processed files.',
                        default='ukb_fundus_preprocessed.h5')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    data_path_left = Path(args.in_left)
    data_path_right = Path(args.in_right)
    save_path = Path(args.out_dir)  # for converted h5  (RGB images)
    convert_png_h5(data_path_left, save_path, 'LeftFundus.h5', args.verbose)
    convert_png_h5(data_path_right, save_path, 'RightFundus.h5', args.verbose)

    """
    if os.path.exists(Path(save_path).joinpath('RightFundus.h5')):
        keys_left = [l.strip() for l in
                       Path(save_path).joinpath('keys_LeftFundus_all.dat').open().readlines()]
        keys_right = [l.strip() for l in
                        Path(save_path).joinpath('keys_RightFundus_all.dat').open().readlines()]
    else:
    """


    # get the test keys
    #keys_test = [l.strip() for l in Path('/mnt/qdata/share/rakuest1/data/UKB/interim/keys/test.dat').open().readlines()]
    #keys = ['left/' + k for k in keys_left] + ['right/' + k for k in keys_right]
    #combine_h5(save_path, args.output_dir, args.output_file, keys, args.verbose)

if __name__ == '__main__':
    main()

