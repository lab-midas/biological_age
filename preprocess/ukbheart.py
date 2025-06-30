import os
import csv
import glob
import re
import time

import numpy as np
import pandas as pd
import dateutil.parser
from tqdm import tqdm
import matplotlib.pyplot as plt
from biobank_utils import *
import urllib.request
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import h5py
import math
from image_utils import rescale_intensity
from skimage.measure import label, regionprops
import argparse
from joblib import Parallel, delayed


def segment_heart(data_list, seq_name='sa', seg4=False, save_seg=True, save_path='./seg'):
    # data_list         list of nifti files to process (full string path)
    # seq_name          sequence name ['sa', 'la_2ch', 'la_4ch']
    # seg4              Segment all the 4 chambers in long-axis 4 chamber view. '
    #                             'This seg4 network is trained using 200 subjects from Application 18545.'
    #                             'By default, for all the other tasks (ventricular segmentation'
    #                             'on short-axis images and atrial segmentation on long-axis images,'
    #                             'the networks are trained using 3,975 subjects from Application 2964.'
    # save_seg          Save segmentation results to nifti files.
    # save_path         Path to save segmentation results including the patient name as folder.

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

    #os.system('CUDA_VISIBLE_DEVICES={0} python3 common/deploy_network.py --seq_name sa --data_dir demo_image '
    #          '--model_path trained_model/FCN_sa'.format(CUDA_VISIBLE_DEVICES))
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
                    #processed_list = f.read().splitlines()
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


        """      
        with open(Path(save_path).joinpath('boundingbox_ES.csv'), 'w') as out:
            csv_out = csv.writer(out)
            for row in bounding_boxes_ES:
                csv_out.writerow(row)

        with open(Path(save_path).joinpath('boundingbox_ED.csv'), 'w') as out:
            csv_out = csv.writer(out)
            for row in bounding_boxes_ED:
                csv_out.writerow(row)"""

    print('Average segmentation time = {:.3f}s per sequence'.format(np.mean(table_time)))
    process_time = time.time() - start_time
    print('Including image I/O, CUDA resource allocation, '
          'it took {:.3f}s for processing {:d} subjects ({:.3f}s per subjects).'.format(
        process_time, 1, process_time ))


def get_bounding_boxes(save_path):
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
    #bounding_boxes_ES = np.asarray([list(x) for x in bounding_boxes_ES if x[0] > -1])
    #bounding_boxes_ED = np.asarray([list(x) for x in bounding_boxes_ED if x[0] > -1])
    #min_loc = np.min(np.concatenate((bounding_boxes_ES, bounding_boxes_ED), axis=0), axis=0)[0:2]  # min_row, min_col, min_slice, max_row, max_col, max_slice
    #max_loc = np.max(np.concatenate((bounding_boxes_ES, bounding_boxes_ED), axis=0), axis=0)[3:5]
    #print('Bounding box min {} to max {}'.format(min_loc, max_loc))
    # bounding_boxes_ES = np.asarray([list(x) for x in bounding_boxes_ES if x[0] > -1])
    # bounding_boxes_ED = np.asarray([list(x) for x in bounding_boxes_ED if x[0] > -1])
    bounding_boxes_ES = np.asarray([list(x) for x in bounding_boxes_ES])
    bounding_boxes_ED = np.asarray([list(x) for x in bounding_boxes_ED])
    leave_out = np.any(bounding_boxes_ES < 0, axis=1)  # failed to create mask
    bounding_boxes_ES = bounding_boxes_ES[~leave_out, :]
    bounding_boxes_ED = bounding_boxes_ED[~leave_out, :]
    pat_failed = pat_list[leave_out]
    pat_list = pat_list[~leave_out]
    crop_shapes_ES = bounding_boxes_ES[:, 3:6] - bounding_boxes_ES[:, 0:3]
    crop_shapes_ED = bounding_boxes_ED[:, 3:6] - bounding_boxes_ED[:, 0:3]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(crop_shapes_ED[:, i], np.unique(crop_shapes_ED[:, i]))
    plt.show()
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(crop_shapes_ES[:, i], np.unique(crop_shapes_ES[:, i]))
    plt.show()

    leave_out = np.any(crop_shapes_ES[:, 0:2] < 5, axis=1) | np.any(crop_shapes_ED[:, 0:2] < 10, axis=1) | (
                crop_shapes_ED[:, 2] < 5)  # minimum shape
    pat_failed += pat_list[leave_out]
    pat_list = pat_list[~leave_out]
    crop_shapes_ES = crop_shapes_ES[~leave_out, :]
    crop_shapes_ED = crop_shapes_ED[~leave_out, :]

    min_shape = np.min(crop_shapes_ED, axis=0)
    max_shape = np.max(crop_shapes_ED, axis=0)
    per_shape = np.percentile(crop_shapes_ED, 99.5, axis=0)
    median_shape = np.median(crop_shapes_ED, axis=0)
    # min_loc = np.min(np.concatenate((bounding_boxes_ES, bounding_boxes_ED), axis=0), axis=0)[0:3]  # min_row, min_col, min_slice, max_row, max_col, max_slice
    # max_loc = np.max(np.concatenate((bounding_boxes_ES, bounding_boxes_ED), axis=0), axis=0)[3:6]
    # min_loc = min_loc[0:2] + np.array([-10, -10, 0])
    # max_loc = max_loc[3:6] + np.array([10, 10, 0])


def crop(x, s, c, shift_center=True):
    # x: input data
    # s: desired size
    # c: center
    # shift_center: shift the center of cropping so that the whole patch is inside the image (True), or symmetric-pad in case that patches extend beyond image borders (False)
    if type(s) is not np.ndarray:
        s = np.asarray(s, dtype='f')

    if type(c) is not np.ndarray:
        c = np.asarray(c, dtype='f')

    if type(x) is not np.ndarray:
        x = np.asarray(x, dtype='f')

    m = np.asarray(np.shape(x), dtype='f')
    if len(m) < len(s):
        m = [m, np.ones(1, len(s) - len(m))]

    if np.sum(m == s) == len(m):
        return x

    def get_limits(sin, cin):
        if np.remainder(sin, 2) == 0:
            lower = cin + 1 + np.ceil(-sin / 2) - 1
            upper = cin + np.ceil(sin / 2)
        else:
            lower = cin + np.ceil(-sin / 2) - 1
            upper = cin + np.ceil(sin / 2) - 1
        return lower, upper

    idx = list()
    idx_pad = ()
    for n in range(np.size(s)):
        #if np.remainder(s[n], 2) == 0:
        #    lower = c[n] + 1 + np.ceil(-s[n] / 2) - 1
        #    upper = c[n] + np.ceil(s[n] / 2)
        #else:
        #    lower = c[n] + np.ceil(-s[n] / 2) - 1
        #    upper = c[n] + np.ceil(s[n] / 2) - 1
        lower, upper = get_limits(s[n], c[n])

        # shift center
        if shift_center:
            if lower < 0:
                c[n] += np.abs(lower)

            if upper > m[n]:
                c[n] -= np.abs(upper - m[n])

            lower, upper = get_limits(s[n], c[n])

        # if even shifting the center is not helping, pad data
        if lower < 0:
            low_pad = int(np.abs(lower))
            lower = 0
            upper = s[n]
        else:
            low_pad = 0

        if upper > m[n]:
            upper_pad = int(np.abs(upper - m[n]))
            upper = m[n]
            lower = m[n] - s[n]
        else:
            upper_pad = 0

        idx.append(list(np.arange(lower, upper, dtype=int)))
        idx_pad += (low_pad, upper_pad),

    index_arrays = np.ix_(*idx)
    x = np.pad(x, idx_pad, mode='symmetric')
    return x[index_arrays]


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


def convert_nifti_h5(input_dir, output_dir, output_file, save_path, single_file=True, df_redo=None, verbose=False):
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

    if single_file and os.path.exists(h5_file):
        print('{} already exists'.format(h5_file))
        #return pickle.load(open(output_dir.joinpath('keys', 'all.dat'), 'r'))
        return [l.strip() for l in output_dir.joinpath('keys', 'all.dat').open().readlines()]

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

    sel_shape = [72, 76, 8]
    print('Extracted cardiac shapes: {}'.format(sel_shape))

    keys = []
    plot_path = Path(save_path).parent.joinpath('qualicheck')

    def process_file(nifti_file):
        keyh5 = nifti_file.stem.split('.')[0]
        key = keyh5.split('_')[0]

        if not single_file and h5_dir.joinpath(keyh5 + '_sa.h5').exists():
            if h5py.File(h5_dir.joinpath(keyh5 + '_sa.h5'), 'r')['image/' + f'{keyh5}' + '_sa'].shape[0:2] == sel_shape[0:2]:
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
        if not '_'.join(keyh5.split('_')[0:2]) in pat_list:
            print('Patient mask extraction failed: ' + keyh5)
            return
        box = bounding_boxes_ED[pat_list.index('_'.join(keyh5.split('_')[0:2]))]
        center = list(np.floor((np.asarray(box[0:3], dtype='int') + np.asarray(box[3:6], dtype='int')) / 2)) + [
            np.floor(np.shape(img_data)[3] / 2)]
        center = [int(x) for x in center]

        # list(box[0:2] + np.floor(np.asarry(sel_shape[0:2]) / 2)) + list(np.floor(np.asarry(box[5] + box[2])/2)) +

        # crop image around segmentation mask
        img_crop = crop(img_data, sel_shape + [np.shape(img_data)[3]], center)

        save_path_curr = Path(save_path).joinpath(key + keyh5.split('_')[1])
        Path(save_path_curr).mkdir(exist_ok=True)

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

        # Quality check
        # plt.imshow(img_crop[:, :, int(np.floor(np.shape(img_crop)[2]/2)), 0])
        # plt.savefig(plot_path.joinpath(keyh5 + '.png'))

    if single_file:
        hf = h5py.File(h5_file, 'w')
        grp_image = hf.create_group('image')
        grp_affine = hf.create_group('affine')
        for nifti_file in tqdm(nifti_files):
            process_file(nifti_file)
    else:
        num_cores = 10
        print(f'using {num_cores} CPU cores')

        t = time.time()
        results = Parallel(n_jobs=num_cores)(
            delayed(process_file)(f) for f in tqdm(nifti_files))
        elapsed_time = time.time() - t

        print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

def merge_hdf5(input_dir, output_dir, output_file):
    pats = os.listdir(input_dir)
    hdf5file = h5py.File(Path(output_dir).joinpath(output_file), 'w')
    #grp_image = hdf5file.create_group('image')

    for pat in pats:
        keyh5 = "_".join(pat.split('_')[0:2])
        hdf5file["/image/" + keyh5] = h5py.ExternalLink(Path(input_dir).joinpath(pat), "/image/" + keyh5 + '_sa')
    hdf5file.close()

def parse_hdf5(input_dir, output_dir, output_file, info='/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_all.csv',
               train_set='/mnt/qdata/share/rakuest1/data/UKB/interim/keys/train_imaging.dat',
               val_set='/mnt/qdata/share/rakuest1/data/UKB/interim/keys/test.dat',
               group='image'):
    hdf5file = h5py.File(Path(output_dir).joinpath(output_file), 'r')
    #info_df = pd.read_csv(info, index_col=0, usecols=[1, 2, 3, 4, 5], dtype={'key': 'string', column: np.float32})
    train_keys = [l.strip() for l in Path(train_set).open().readlines()]
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]
    all_keys = train_keys + val_keys
    datlist = get_shapes(hdf5file, input_dir, all_keys, group)
    #d = {'keys': key_proc, 'h5size': shapes_h5, 'niisize': shapes_nii}
    df = pd.DataFrame(datlist)
    df.to_csv(output_dir.joinpath('ukb_heart_sizes.csv'))
    print('Done with shapes')

def get_shapes(hdf5file, input_dir, keys, group):
    keys = [l.strip() for l in Path(keys).open().readlines()] if isinstance(keys, str) else keys
    #shapes_h5 = list()
    #shapes_nii = list()
    #key_proc = list()
    datlist = list()
    #for key in tqdm(keys):
    def get_shape_inner(key):
        group_str = group + '/' if group else ''

        for idx in [2, 3]:
            keyh5 = key + '_' + str(idx)
            # if Path(self.datapath).joinpath(keyh5 + '_sa.h5').exists():
            if f'{group_str}{keyh5}' in hdf5file:
                break
            else:
                keyh5 = ''

        shape_h5 = hdf5file[f'{group_str}{keyh5}'].shape
        #shapes_h5.append(np.shape(data))
        shape_nii = nib.load(input_dir.joinpath(keyh5 + '_sa.nii.gz')).shape
        #shapes_nii.append(img.shape)
        #key_proc.append(keyh5)
        sample = {'keys': keyh5, 'h5size': shape_h5, 'niisize': shape_nii}
        #datlist.append(sample)
        return sample
        #return np.shape(data), img.shape

    num_cores = 1
    print(f'using {num_cores} CPU cores')

    t = time.time()
    #_ = Parallel(n_jobs=num_cores, backend='threading')(
    #    delayed(get_shape_inner)(f) for f in tqdm(keys))
    for f in tqdm(keys):
        sample = get_shape_inner(f)
        datlist.append(sample)
    elapsed_time = time.time() - t

    print(f'elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    return datlist

def main():
    parser = argparse.ArgumentParser(description='Preprocessing pipeline for UK Biobank SA Heart MRI data.\n' \
                                                 'CSV creation\n' \
                                                 'Nifti to HDF5 conversion\n' \
                                                 'Key creation for train, test, val')
    parser.add_argument('input_dir', help='Input directory of all nifti files (*.nii.gz)')
    parser.add_argument('output_dir', help='Output directory for all files',
                        default='/mnt/qdata/share/raeckev1/nako_30k/interim/')
    parser.add_argument('--output_file', help='Output h5 file to store processed files.',
                        default='ukb_heart_preprocessed.h5')
    parser.add_argument('--csv_input', help='Input CSV file',
                        default='/mnt/qdata/rawdata/UKBIOBANK/baskets/4053862/ukb677731.csv')
    parser.add_argument('--csv_output', help='Output CSV file', default='ukb_brain.csv')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    #data_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/sa_heart/raw'
    save_path = '/mnt/qdata/share/raeckev1/nako_30k/sa_heart/processed/seg'  # for segmentations
    Path(save_path).mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # run segmentation on whole cohort first to get shapes of segmentation masks
    nifti_files = [f for f in Path(input_dir).glob('*.nii.gz')]
    #nifti_files = nifti_files[nifti_files.index(Path('/mnt/qdata/share/raecker1/ukbdata_70k/sa_heart/raw/4637290_2_sa.nii.gz')):]
    print(f'segmenting {len(nifti_files)} images')
    segment_heart(nifti_files, save_path=save_path)
    #get_bounding_boxes(save_path)
    #convert_nifti_h5(input_dir, output_dir, args.output_file, save_path, single_file=True, verbose=False)
    #merge_hdf5(input_dir, output_dir, args.output_file)
    #parse_hdf5(input_dir, output_dir, args.output_file)
    #df = pd.read_csv('/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_heart_sizes.csv')
    #df_redo = df.copy()
    #df_redo = df_redo.loc[df_redo['h5size'] != '(72, 76, 8, 50)']
    #df_redo['h5size'] = df_redo['h5size'].apply(lambda x: np.fromstring(x.replace('(', '').replace(')', ''), dtype=int, sep=","))
    #df_redo['idx'] = df_redo['h5size'].apply(lambda x: np.all(x != np.asarray([72, 76, 8, 50])))
    #df_redo = df_redo.loc[df_redo['idx'] == True]
    #convert_nifti_h5(input_dir, output_dir, args.output_file, save_path, single_file=False, df_redo=None, verbose=False)

if __name__ == '__main__':
    main()

