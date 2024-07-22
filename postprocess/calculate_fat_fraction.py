import os
import sys
import csv
from tqdm import tqdm
import h5py
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
sys.path.append('../preprocess')
from ukbheart import crop


def calculate_inner_organ_fat_fraction(resume):
    img_path = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/'
    mask_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg_ori'
    bounding_boxes_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg_ori/bounding_boxes_abdomen.csv'
    organs = ['liv', 'spl', 'rkd', 'lkd', 'pnc']
    sel_shapes = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50], 'pnc': [80, 50, 50]}
    #df_AT = pd.DataFrame(columns=['eid'] + organs)
    mode = 'w' if resume is None else 'a'
    with open('/mnt/qdata/share/raecker1/AT_examples/organ_fat_fraction.csv', mode) as out:
        csv_out = csv.writer(out)
        if resume is None:
            csv_out.writerow(['eid'] + organs)
            files = os.listdir(img_path)
        else:
            files = os.listdir(img_path)
            files = files[files.index(resume)+1:]
        for i, subject in enumerate(tqdm(files)):
            try:
                # load fat image
                nimf = nib.load(os.path.join(img_path, subject, 'fat.nii.gz'))
                fat_image = nimf.get_fdata()
                #load wat image
                nimw = nib.load(os.path.join(img_path, subject, 'wat.nii.gz'))
                wat_image = nimw.get_fdata()            
                # load seg mask
                nmsk = nib.load(os.path.join(mask_path, subject, 'prd.nii.gz'))
                mask = nmsk.get_fdata()
            except:
                print(f'skip {subject}')
                continue

            bounding_boxes = pd.read_csv(bounding_boxes_path)

            row = [subject]
            for j, class_name in enumerate(organs):
                sel_shape_curr = sel_shapes[class_name]
                box = bounding_boxes.loc[bounding_boxes['pat'] == int(subject)][class_name].values
                box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
                if list(box) == [-1, -1, -1, -1, -1, -1]:
                    row.append(str(np.nan))
                else:
                    center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
                    center = [int(x) for x in center]
                    fat_img_crop = crop(fat_image, sel_shape_curr, center)
                    wat_img_crop = crop(wat_image, sel_shape_curr, center)
                    mask_crop = crop(mask, sel_shape_curr, center)
                    mask_crop = np.where(mask_crop == j+1, 1, 0)
                    img_fat = np.multiply(fat_img_crop, mask_crop)
                    img_wat = np.multiply(wat_img_crop, mask_crop)
                    fat = img_fat.sum()
                    wat = img_wat.sum()
                    fat_fraction = fat / (fat + wat)
                    #fat_count = np.where(img_fat > 0.5, 1, 0).sum()
                    row.append(str(fat_fraction))
                    #plt.imshow(img_fat[:,:,img_fat.shape[2]//2], cmap='gray')
                    #plt.savefig(f'/mnt/qdata/share/raecker1/AT_examples/{subject}_AT_organ_{class_name}.png')
            csv_out.writerow(row)
            #df_AT.loc[i] = row
    #df_AT.to_csv('/mnt/qdata/share/raecker1/AT_examples/organ_fat.csv')


def calculate_outter_organ_fat_fraction(resume):
    img_path = '/mnt/qdata/rawdata/UKBIOBANK/ukbdata_50k/abdominal_MRI/raw/'
    mask_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg_ori'
    bounding_boxes_path = '/mnt/qdata/share/rakuest1/data/UKB/raw/abdominal_MRI/processed/seg_ori/bounding_boxes_abdomen.csv'
    organs = ['liv', 'spl', 'rkd', 'lkd', 'pnc']
    sel_shapes = {'liv': [120, 100, 70], 'spl': [60, 60, 50], 'rkd': [40, 40, 50], 'lkd': [40, 40, 50], 'pnc': [80, 50, 50]}
    #df_AT = pd.DataFrame(columns=['eid'] + organs)
    mode = 'w' if resume is None else 'a'
    with open('/mnt/qdata/share/raecker1/AT_examples/background_fat_fraction.csv', mode) as out:
        csv_out = csv.writer(out)
        if resume is None:
            csv_out.writerow(['eid'] + organs)
            files = os.listdir(img_path)
        else:
            files = os.listdir(img_path)
            files = files[files.index(resume)+1:]
        for i, subject in enumerate(tqdm(files)):
            try:
                # load fat image
                nimf = nib.load(os.path.join(img_path, subject, 'fat.nii.gz'))
                fat_image = nimf.get_fdata()
                #load wat image
                nimw = nib.load(os.path.join(img_path, subject, 'wat.nii.gz'))
                wat_image = nimw.get_fdata()            
                # load seg mask
                nmsk = nib.load(os.path.join(mask_path, subject, 'prd.nii.gz'))
                mask = nmsk.get_fdata()
            except:
                print(f'skip {subject}')
                continue

            bounding_boxes = pd.read_csv(bounding_boxes_path)

            row = [subject]
            for j, class_name in enumerate(organs):
                sel_shape_curr = sel_shapes[class_name]
                box = bounding_boxes.loc[bounding_boxes['pat'] == int(subject)][class_name].values
                box = np.asarray([list(ast.literal_eval(l)) for l in box])[0]
                if list(box) == [-1, -1, -1, -1, -1, -1]:
                    row.append(str(np.nan))
                else:
                    center = list(np.floor((box[0:3] + box[3:6]) / 2))  # + [np.floor(np.shape(img_data)[2] / 2)]
                    center = [int(x) for x in center]
                    fat_img_crop = crop(fat_image, sel_shape_curr, center)
                    wat_img_crop = crop(wat_image, sel_shape_curr, center)
                    mask_crop = crop(mask, sel_shape_curr, center)
                    mask_crop = np.where(mask_crop == j+1, 0, 1)
                    img_fat = np.multiply(fat_img_crop, mask_crop)
                    img_wat = np.multiply(wat_img_crop, mask_crop)
                    fat = img_fat.sum()
                    wat = img_wat.sum()
                    fat_fraction = fat / (fat + wat)
                    #fat_count = np.where(img_fat > 0.5, 1, 0).sum()
                    row.append(str(fat_fraction))
                    #plt.imshow(img_fat[:,:,img_fat.shape[2]//2], cmap='gray')
                    #plt.savefig(f'/mnt/qdata/share/raecker1/AT_examples/{subject}_AT_organ_{class_name}.png')
            csv_out.writerow(row)
            #df_AT.loc[i] = row
    #df_AT.to_csv('/mnt/qdata/share/raecker1/AT_examples/organ_fat.csv')


if __name__ == '__main__':
    resume = '3189733'
    calculate_inner_organ_fat_fraction(resume)
    #resume = '2873633'
    #calculate_outter_organ_fat_fraction(resume)
