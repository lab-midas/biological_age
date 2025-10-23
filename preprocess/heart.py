import os
import csv
import time

import numpy as np
from tqdm import tqdm
from preprocess.utils.biobank_utils import *
import urllib.request
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import h5py
import math
from preprocess.utils.image_utils import rescale_intensity, crop
from skimage.measure import regionprops
import argparse
from joblib import Parallel, delayed


def segment_heart(data_list, seq_name='sa', seg4=False, save_path='./seg'):
    """
    Segments the heart from a list of nifti files using a trained neural network.

    Args:
        data_list (list): List of nifti files to process (full string path).
        seq_name (str): Sequence name ['sa', 'la_2ch', 'la_4ch'].
        seg4 (bool): Segment all the 4 chambers in long-axis 4 chamber view.
            This seg4 network is trained using 200 subjects from Application 18545.
            By default, for all the other tasks (ventricular segmentation
            on short-axis images and atrial segmentation on long-axis images,
            the networks are trained using 3,975 subjects from Application 2964.
        save_path (str): Path to save segmentation results including the patient name as folder.

    Returns:
        None
    """

    # The URL for downloading demo data
    URL = 'https://www.doc.ic.ac.uk/~wbai/data/ukbb_cardiac/'

    # Download information spreadsheet
    print('Downloading information spreadsheet ...')
    if not os.path.exists('demo_csv'):
        os.makedirs('demo_csv')
    for f in ['demo_csv/blood_pressure_info.csv']:
        urllib.request.urlretrieve(URL + f, f)

    # Download trained models
    print('Downloading trained models ...')
    if not os.path.exists('trained_model'):
        os.makedirs('trained_model')
    for model_name in ['FCN_sa', 'FCN_la_2ch', 'FCN_la_4ch', 'FCN_la_4ch_seg4', 'UNet-LSTM_ao']:
        for f in ['trained_model/{0}.meta'.format(model_name),
                  'trained_model/{0}.index'.format(model_name),
                  'trained_model/{0}.data-00000-of-00001'.format(model_name)]:
            urllib.request.urlretrieve(URL + f, f)

    print('******************************')
    print('  Short-axis image analysis')
    print('******************************')

    # Deploy the segmentation network
    print('Deploying the segmentation network ...')
    model_path = 'trained_model/FCN_sa'

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Load the segmentation network
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # Import the computation graph and restore the variable values
        saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(model_path))
        saver.restore(sess, '{0}'.format(model_path))

        start_time = time.time()
        processed_list = []
        if Path(save_path).joinpath('processed_list.txt').exists():
            try:
                with open(Path(save_path).joinpath('processed_list.txt'), 'rb') as f:
                    processed_list = pickle.load(f)
            except:
                # broken during write
                processed_list = [p.name for p in Path(save_path).iterdir() if p.is_dir() and ('seg_sa_ES.nii.gz' in os.listdir(Path(save_path).joinpath(p)))]
        table_time = []
        bounding_boxes_ED = []
        bounding_boxes_ES = []

        for i,data in enumerate(data_list):
            if '_'.join(Path(data).name.split('.')[0].split('_')[0:2]) in processed_list:
                continue

            image_name = data
            pat_name = '_'.join(os.path.basename(image_name).split('.')[0].split('_')[0:2])
            save_path_pat = Path(save_path).joinpath(pat_name)
            save_path_pat.mkdir(exist_ok=True)
            # Read the image
            print('  Reading {} ...'.format(image_name))
            try: 
                nim = nib.load(image_name)
                image = nim.get_data()
            except Exception as e:
                bounding_boxes_ED.append(None)
                bounding_boxes_ES.append(None)
                print('------------')
                print(f'subject: {pat_name}')
                print('------------')
                print(e)
                continue
            X, Y, Z, T = image.shape
            orig_image = image

            print('  Segmenting full sequence ...')
            start_seg_time = time.time()

            # Intensity rescaling
            image = rescale_intensity(image, (1, 99))

            # Prediction (segmentation)
            pred = np.zeros(image.shape)

            # Pad the image size to be a factor of 16 so that the
            # downsample and upsample procedures in the network will
            # result in the same image size at each resolution level.
            X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
            x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
            x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
            image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')

            # Process each time frame
            for t in range(T):
                # Transpose the shape to NXYC
                image_fr = image[:, :, :, t]
                image_fr = np.transpose(image_fr, axes=(2, 0, 1)).astype(np.float32)
                image_fr = np.expand_dims(image_fr, axis=-1)

                # Evaluate the network
                prob_fr, pred_fr = sess.run(['prob:0', 'pred:0'],
                                            feed_dict={'image:0': image_fr, 'training:0': False})

                # Transpose and crop segmentation to recover the original size
                pred_fr = np.transpose(pred_fr, axes=(1, 2, 0))
                pred_fr = pred_fr[x_pre:x_pre + X, y_pre:y_pre + Y]
                pred[:, :, :, t] = pred_fr

            seg_time = time.time() - start_seg_time
            print('  Segmentation time = {:3f}s'.format(seg_time))
            table_time += [seg_time]
            processed_list += [data]

            # ED frame defaults to be the first time frame.
            # Determine ES frame according to the minimum LV volume.
            k = {}
            k['ED'] = 0
            if seq_name == 'sa' or (seq_name == 'la_4ch' and seg4):
                k['ES'] = np.argmin(np.sum(pred == 1, axis=(0, 1, 2)))
            else:
                k['ES'] = np.argmax(np.sum(pred == 1, axis=(0, 1, 2)))
            print('  ED frame = {:d}, ES frame = {:d}'.format(k['ED'], k['ES']))

            print('Saving bounding box ...')
            try:
                bounding_box_ED = regionprops(np.asarray(pred[:, :, :, k['ED']] > 0, dtype=int))[0].bbox
                bounding_box_ES = regionprops(np.asarray(pred[:, :, :, k['ES']] > 0, dtype=int))[0].bbox
                bounding_boxes_ED.append(bounding_box_ED)
                bounding_boxes_ES.append(bounding_box_ES)
            except:
                bounding_boxes_ED.append((-1, -1, -1, -1, -1, -1))
                bounding_boxes_ES.append((-1, -1, -1, -1, -1, -1))

            print('  Saving segmentation ...')
            nim2 = nib.Nifti1Image(pred, nim.affine)
            nim2.header['pixdim'] = nim.header['pixdim']
            if seq_name == 'la_4ch' and seg4:
                seg_name = '{0}/seg4_{1}.nii.gz'.format(save_path_pat, seq_name)
            else:
                seg_name = '{0}/seg_{1}.nii.gz'.format(save_path_pat, seq_name)
            nib.save(nim2, seg_name)

            for fr in ['ED', 'ES']:
                nib.save(nib.Nifti1Image(orig_image[:, :, :, k[fr]], nim.affine),
                         '{0}/{1}_{2}.nii.gz'.format(save_path_pat, seq_name, fr))
                if seq_name == 'la_4ch' and seg4:
                    seg_name = '{0}/seg4_{1}_{2}.nii.gz'.format(save_path_pat, seq_name, fr)
                else:
                    seg_name = '{0}/seg_{1}_{2}.nii.gz'.format(save_path_pat, seq_name, fr)
                nib.save(nib.Nifti1Image(pred[:, :, :, k[fr]], nim.affine), seg_name)

            # save processed list
            with open(Path(save_path).joinpath('processed_list.txt'), 'wb') as f:
                pickle.dump(processed_list, f)

            mode = 'a' if Path(save_path).joinpath('boundingbox_ES.csv').exists() else 'w'
            with open(Path(save_path).joinpath('boundingbox_ES.csv'), mode) as out:
                csv_out = csv.writer(out)
                csv_out.writerow((pat_name,) + bounding_boxes_ES[i])

            with open(Path(save_path).joinpath('boundingbox_ED.csv'), mode) as out:
                csv_out = csv.writer(out)
                csv_out.writerow((pat_name,) + bounding_boxes_ED[i])

    print('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
    process_time = time.time() - start_time
    print('Including image I/O, CUDA resource allocation, '
          'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
        process_time, 1, process_time ))


def get_bounding_boxes(save_path):
    """
    Extract bounding boxes from segmentation masks.
    Args:
        save_path (str): Path to save bounding boxes files.
    """

    pat_list = [p.name for p in Path(save_path).iterdir() if
                      p.is_dir() and ('seg_sa_ES.nii.gz' in os.listdir(Path(save_path).joinpath(p)))]
    bounding_boxes_ED = []
    bounding_boxes_ES = []
    for pat in tqdm(pat_list):
        pred_ED = nib.load(Path(save_path).joinpath(pat, 'seg_sa_ED.nii.gz')).get_fdata()
        pred_ES = nib.load(Path(save_path).joinpath(pat, 'seg_sa_ES.nii.gz')).get_fdata()
        try:
            bounding_box_ED = regionprops(np.asarray(pred_ED > 0, dtype=int))[0].bbox
            bounding_box_ES = regionprops(np.asarray(pred_ES > 0, dtype=int))[0].bbox
        except:
            bounding_box_ED = [-1, -1, -1, -1, -1, -1]
            bounding_box_ES = [-1, -1, -1, -1, -1, -1]
            print('Error in patient {}'.format(pat))
        bounding_boxes_ED.append(bounding_box_ED)
        bounding_boxes_ES.append(bounding_box_ES)

    with open(Path(save_path).joinpath('boundingbox_ES.csv'), 'w') as out:
        csv_out = csv.writer(out)
        for idx, row in enumerate(bounding_boxes_ES):
            csv_out.writerow([pat_list[idx]] + list(row))

    with open(Path(save_path).joinpath('boundingbox_ED.csv'), 'w') as out:
        csv_out = csv.writer(out)
        for idx, row in enumerate(bounding_boxes_ED):
            csv_out.writerow([pat_list[idx]] + list(row))

    print('done bounding boxes')


def convert_nifti_h5(input_dir, output_dir, output_file, key_path, save_path, cohort, sel_shape, single_file=True, df_redo=None, verbose=False):
    """
    Convert nifti files to h5 files. Crop around segmentation masks.
    Args:
        input_dir (Path): input directory with nifti files
        output_dir (Path): output directory to store h5 file
        output_file (str): output h5 file name
        key_path (Path): output dat file to store keys
        save_path (str): path where segmentation masks are stored
        sel_shape (list of int): desired shape after cropping around segmentation masks
        single_file (bool): store all images in a single h5 file (True) or in separate h5 files (False)
        df_redo (DataFrame): DataFrame with keys of nifti files to redo. If None, process all files.
        verbose (bool): print progress
    """

    def write_keys(key_path, keys):

        with open(key_path, 'w') as f:
            for key in keys:
                f.write(key + '\n')
        print(f'Keys written to {key_path}')

    def process_file(nifti_file):
        file_key = nifti_file.stem.split('.')[0]
        keyh5 = file_key.split('_')[0]

        if not single_file and h5_dir.joinpath(file_key + '_sa.h5').exists():
            if h5py.File(h5_dir.joinpath(file_key + '_sa.h5'), 'r')['image/' + f'{file_key}' + '_sa'].shape[0:2] == sel_shape[0:2]:
                return
        try: 
            img = nib.load(nifti_file)
            img_data = img.get_fdata().astype(np.float32)
            affine = img.affine.astype(np.float16)
        except Exception as e:
            print('------------')
            print(f'subject {nifti_file} failed to load')
            print('------------')
            print(e)
            return

        # find patient
        if cohort == 'ukb':
            if not '_'.join(file_key.split('_')[0:2]) in pat_list:
                print('Patient mask extraction failed: ' + file_key)
                return
            box = bounding_boxes_ED[pat_list.index('_'.join(file_key.split('_')[0:2]))]
        elif cohort == 'nako':
            if not file_key[:-3] in pat_list:
                print('Patient mask extraction failed: ' + file_key)
                return
            box = bounding_boxes_ED[pat_list.index(file_key[:-3])]
        else:
            print('Cohort not recognized. Use "ukb" or "nako".')
            return
        center = list(np.floor((np.asarray(box[0:3], dtype='int') + np.asarray(box[3:6], dtype='int')) / 2)) + [
            np.floor(np.shape(img_data)[3] / 2)]
        center = [int(x) for x in center]

        # crop image around segmentation mask
        img_crop = crop(img_data, sel_shape + [np.shape(img_data)[3]], center)

        # Write to h5 file
        if single_file:
            grp_image.create_dataset(keyh5, data=img_crop)
            grp_affine.create_dataset(keyh5, data=affine)
        else:
            if h5_dir.joinpath(keyh5 + '_sa.h5').exists():
                Path(h5_dir.joinpath(keyh5 + '_sa.h5')).unlink()
            h5file = h5py.File(h5_dir.joinpath(keyh5 + '.h5'), 'w')
            grps_image = h5file.create_group('image')
            grps_affine = h5file.create_group('affine')
            grps_image.create_dataset(keyh5, data=img_crop)
            grps_affine.create_dataset(keyh5, data=affine)
    
    # Create output directory if it does not exist
    output_dir.mkdir(exist_ok=True)

    # Get list of all nifti files in input_dir
    nifti_files = [f for f in input_dir.glob('*.nii.gz')]

    if df_redo is not None:
        nifti_files = [f for f in nifti_files if '_'.join(f.stem.split('.')[0].split('_')[0:2]) in df_redo['keys'].tolist()]
        print('Redoing {} nifti files'.format(len(nifti_files)))

    # Create list of all h5 files in output_dir
    if single_file:
        h5_file = output_dir.joinpath(output_file)
    else:
        h5_dir = output_dir.joinpath(Path(output_file).stem)
        h5_dir.mkdir(exist_ok=True)

    # Save path for segmentation masks
    bounding_boxes_ED = []
    pat_list = []
    with open(Path(save_path).joinpath('boundingbox_ED.csv'), 'r') as out:
        csv_reader = csv.reader(out)
        for row in csv_reader:
            pat_list.append(row[0])
            bounding_boxes_ED.append(row[1:])

    bounding_boxes_ES = []
    with open(Path(save_path).joinpath('boundingbox_ES.csv'), 'r') as out:
        csv_reader = csv.reader(out)
        for row in csv_reader:
            bounding_boxes_ES.append(row[1:])

    print('Extracted cardiac shapes: {}'.format(sel_shape))

    # Process each nifti file sequentially in a single file or in parallel
    if single_file:
        hf = h5py.File(h5_file, 'w')
        grp_image = hf.create_group('image')
        grp_affine = hf.create_group('affine')
        for nifti_file in tqdm(nifti_files):
            process_file(nifti_file)

        # Get keys and write to file
        keys = []

        h5 = h5py.File(h5_file, 'r')
        for key in h5['image'].keys():
            keys.append(key)
        h5.close()

        keys = list(set(keys))
        write_keys(key_path, keys)
    
    else:
        num_cores = 10
        print(f'using {num_cores} CPU cores')

        t = time.time()
        # parallel processing
        results = Parallel(n_jobs=num_cores)(
            delayed(process_file)(f) for f in tqdm(nifti_files))
        
        # Get keys and write to file
        keys = []

        h5 = h5py.File(h5_file, 'r')
        for key in h5['image'].keys():
            keys.append(key)
        h5.close()

        keys = list(set(keys))
        write_keys(key_path, keys)
        
        elapsed_time = time.time() - t
        print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')


def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for UK Biobank SA Heart MRI data.\n' \
                                     'Segmentation of the heart\n' \
                                     'Nifti to HDF5 conversion')
    parser.add_argument('input_dir', help='Input directory of all nifti files (*.nii.gz)')
    parser.add_argument('output_dir', help='Output directory for segmentation masks\n' \
                        'ukb: /mnt/qdata/rawdata/UKBIOBANK/ukbdata_70k/sa_heart/processed/seg\n' \
                        'nako: /mnt/qdata/share/raeckev1/nako_30k/sa_heart/processed/seg',
                        default='/mnt/qdata/share/raeckev1/nako_30k/sa_heart/processed/seg')
    parser.add_argument('h5_dir', help='Output directory for all files\n' \
                        'ukb: /mnt/qdata/share/raecker1/ukbdata_70k/interim/\n' \
                        'nako: /mnt/qdata/share/raecker1/nako_30k/interim/',
                        default='/mnt/qdata/share/raecker1/ukbdata_70k/interim/')
    parser.add_argument('--output_file', help='Output h5 file to store processed files.\n' \
                        'ukb: ukb_heart_preprocessed.h5\n' \
                        'nako: nako_heart_preprocessed.h5',
                        default='ukb_heart_preprocessed.h5')
    parser.add_argument('--key_file', help='Output csv file to store keys.',
                        default='heart_imaging.dat')
    parser.add_argument('--heart_shape', help='Desired shape after cropping around segmentation masks.', default=[72, 76, 8], type=int, nargs=3)
    parser.add_argument('--cohort', help='Cohort to process: "nako" or "ukb"', default='ukb')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    h5_dir = Path(args.h5_dir)

    key_path = h5_dir.joinpath('keys', args.key_file)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f'Processing cohort: {args.cohort}')

    # run segmentation on whole cohort first to get shapes of segmentation masks
    nifti_files = [f for f in Path(input_dir).glob('*.nii.gz')]
    print(f'segmenting {len(nifti_files)} images')
    segment_heart(nifti_files, save_path=output_dir)

    convert_nifti_h5(input_dir, h5_dir, args.output_file, key_path, output_dir, args.cohort, sel_shape=args.heart_shape, single_file=False, df_redo=None, verbose=False)

if __name__ == '__main__':
    main()

