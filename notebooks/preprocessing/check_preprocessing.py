import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import time
import os
import h5py
#import glob

from omegaconf import OmegaConf
from brainage.dataset.dataset3d import BrainDataset
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose
from torch.utils.data import DataLoader

import hydra
config = os.getenv('CONFIG')

def log_samples(batch, batch_idx):
    samples = []
    for img, label in zip(batch['data'], batch['label']):
        img = img[0, :, img.size()[1] // 2, :].cpu().numpy() * 255.0
        #samples.append(wandb.Image(img, caption=f'batch {batch_idx} age {label}'))
    #wandb.log({'samples': samples})


def print_nifti(nifti_file, doshow=False):
    img = nib.load(nifti_file)
    img_data = img.get_fdata().astype(np.float32)
    plt.imshow(np.abs(img_data[:, :, np.shape(img_data)[2] // 2]))
    plt.title(os.path.basename(nifti_file))
    if doshow:
        plt.show()


def print_h5(h5_file, key, doshow=False):
    fhandle = h5py.File(h5_file, 'r')
    group_str = 'image/'
    try:
        keyh5 = key + '_2'
        img_data = fhandle[f'{group_str}{keyh5}'][:]
    except:
        keyh5 = key + '_3'
        img_data = fhandle[f'{group_str}{keyh5}'][:]
    plt.imshow(np.abs(img_data[:, :, np.shape(img_data)[2] // 2]))
    plt.title(key)
    if doshow:
        plt.show()

#@hydra.main(config_path=os.path.dirname(config), config_name=os.path.splitext(os.path.basename(config))[0])
def check_loader(doshow=False):
    cfg = OmegaConf.load('/home/rakuest1/Dropbox/Promotion/Matlab/DeepLearning/nako_ukb_age/config/volume/config_ukb.yaml')
    os.environ['DATA'] = '/mnt/qdata/share/rakuest1/data/'

    project = cfg.project.name
    job = cfg.project.job
    data_path = cfg.dataset.data
    data_group = cfg.dataset.group
    info = cfg.dataset.info
    infocolumn = cfg.dataset.column
    train_set = cfg.dataset.train
    val_set = cfg.dataset.val
    debug_set = cfg.dataset.debug or None
    if debug_set:
        train_set = debug_set
        val_set = debug_set
    patch_size = cfg.dataset.patch_size
    data_mode = cfg.dataset.mode
    data_augmentation = cfg.dataset.data_augmentation
    crop_size = np.array(cfg.dataset.crop_size)
    crop_margins = np.array(cfg.dataset.crop_margins)
    gamma_range = cfg.dataset.gamma_range
    preload = cfg.dataset.preload
    seed = cfg.project.seed or 42
    #seed_everything(seed)
    ts = time.gmtime()
    job_id = 'fold' + f'-{cfg.dataset.fold}-' + time.strftime("%Y-%m-%d-%H-%M-%S", ts)
    train_keys = [l.strip() for l in Path(train_set).open().readlines()]
    transforms = []
    if np.any(crop_size):
        transforms.append(RandomCropTransform(crop_size=crop_size, data_key='data', margins=crop_margins))
    transforms.append(GammaTransform(gamma_range=gamma_range, data_key='data'))
    transforms.append(MirrorTransform(axes=[0], data_key='data'))
    train_transform = Compose(transforms)

    ds_train = BrainDataset(data=data_path,
                            keys=train_keys,
                            info=info,
                            group=data_group,
                            column=infocolumn,
                            preload=preload,
                            transform=train_transform)

    batch_idx = 0
    batch = ds_train.__getitem__(batch_idx)
    loader = DataLoader(ds_train, batch_size=8, num_workers=0, drop_last=True, shuffle=True, pin_memory=True)
    batch = next(iter(loader))

    fig, ax = plt.subplots()
    icnt = 1
    for img, label in zip(batch['data'], batch['label']):
    #for ibatch in range(0, 8):
        #img, label = batch['data'][ibatch, ...], batch['label'][ibatch]
        plt.subplot(2, 4, icnt)
        img = img[0, :, img.size()[1] // 2, :].cpu().numpy() * 255.0
        plt.imshow(img)
        plt.title(f'batch {batch_idx} age {label}')
        icnt += 1

    if doshow:
        plt.show()

if __name__ == '__main__':
    input_dir = Path('/mnt/qdata/rawdata/UKBIOBANK/ukbdata/brain/t1/Dicom')
    input_nifti = Path('/mnt/qdata/share/rakuest1/data/UKB/raw/t1_brain/processed')
    output_dir = Path('/mnt/qdata/share/rakuest1/data/UKB/interim/')
    output_h5 ='/mnt/qdata/share/rakuest1/data/UKB/interim/ukb_brain_preprocessed.h5'
    out_quality_check = Path('/mnt/qdata/share/rakuest1/data/UKB/interim/quality_check')

    nifti_files = [f for f in input_nifti.glob('*.nii.gz')]

    for nifti_file in nifti_files:
        fig, _ = plt.subplots()
        print_nifti(nifti_file)
        fig.savefig(out_quality_check.joinpath('nifti', os.path.splitext(os.path.basename(nifti_file))[0].split('_')[0] + '.png'))
        plt.close(fig)

    for nifti_file in nifti_files:
        fig, _ = plt.subplots()
        print_h5(output_h5, os.path.splitext(os.path.basename(nifti_file))[0].split('_')[0])
        fig.savefig(out_quality_check.joinpath('h5', os.path.splitext(os.path.basename(nifti_file))[0].split('_')[0] + '.png'))
        plt.close(fig)



    '''
    nmax = 5
    icnt = 0
    for nifti_file in nifti_files:
        if icnt >= nmax:
            break
        print_nifti(nifti_file)
        icnt += 1

    nmax = 5
    icnt = 0
    for nifti_file in nifti_files:
        if icnt >= nmax:
            break
        print_h5(output_h5, os.path.splitext(os.path.basename(nifti_file))[0].split('_')[0])
        icnt += 1

    nmax = 4
    icnt = 0
    for nifti_file in nifti_files:
        if icnt >= nmax:
            break
        plt.subplot(2,2,icnt+1)
        print_nifti(nifti_file)
        icnt += 1
    plt.show()

    nmax = 4
    icnt = 0
    for nifti_file in nifti_files:
        if icnt >= nmax:
            break
        plt.subplot(2, 2, icnt + 1)
        print_h5(output_h5, os.path.splitext(os.path.basename(nifti_file))[0].split('_')[0])
        icnt += 1
    plt.show()
    '''