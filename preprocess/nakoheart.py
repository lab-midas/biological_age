import os
import math
import time
import pickle
import csv
import numpy as np
import nibabel as nib
import tensorflow as tf
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from pathlib import Path

import urllib.request

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