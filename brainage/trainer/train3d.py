import os
import time
from pathlib import Path

import torch
import hydra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import NeptuneLogger
#from dotenv import load_dotenv
import wandb

from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose

from brainage.model.model3d import AgeModel3DVolume
from brainage.model.model2d import AgeModel2DChannels
from brainage.dataset.dataset3d import BrainDataset, BrainPatchDataset, HeartDataset
from brainage.dataset.dataset2d import FundusDataset
from brainage.utils import fix_dict_in_wandb_config, train_args

#load_dotenv()

config = os.getenv('CONFIG')

# @hydra.main(config_path=os.path.dirname(config), config_name=os.path.splitext(os.path.basename(config))[0])
def main(args):
    # config
    project = args.name
    job = args.name
    data_path = args.datapath
    data_group = args.group
    info = args.info
    infocolumn = args.column
    train_set = args.train
    val_set = args.val
    debug_set = None
    if debug_set:
        train_set = debug_set
        val_set = debug_set
        offline_wandb = True
        log_model = False
    else:
        offline_wandb = False
        log_model = True
    patch_size = args.patchsize
    data_mode = args.mode 
    data_augmentation = args.augmentation
    crop_size = np.array(args.cropsize)
    crop_margins = np.array(args.cropmargins)
    gamma_range = args.gammarange
    mirror_axis = args.mirror
    preload = args.preload
    seed = 42
    seed_everything(seed)
    ts = time.gmtime()
    job_id = 'fold' + f'-{args.fold}-' + time.strftime("%Y-%m-%d-%H-%M-%S", ts)
    if 'brain' in job:
        dataset = 'brain'
    elif 'heart' in job:
        dataset = 'heart'
    elif 'kidney' in job:
        dataset = 'abdominal'
    elif 'liver' in job:
        dataset = 'abdominal'
    elif 'spleen' in job:
        dataset = 'abdominal'
    elif 'fundus' in job:
        dataset = 'fundus'

    # logging
    if not offline_wandb:
        wandb.init(name=f'{job}-{job_id}', entity='lab-midas', project=project, config=args)
    wandb_logger = WandbLogger(name=f'{job}-{job_id}', entity='lab-midas', project=project, offline=offline_wandb, log_model=log_model)
    #neptune_logger = NeptuneLogger(project_name=f'lab-midas/{project}',
    #                               params=OmegaConf.to_container(cfg, resolve=True),
    #                               experiment_name=f'{job}-{job_id}',
    #                               tags=[job])
    
    # get keys and metadata
    train_keys = [l.strip() for l in Path(train_set).open().readlines()]
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]

    assert data_mode in ['patchwise', 'volume']
    if data_mode == 'patchwise':
        val_transform = None
        transforms = [
            GammaTransform(gamma_range=gamma_range, data_key='data'),
            MirrorTransform(axes=[0], data_key='data',),]
        train_transform = Compose(transforms)
        if data_augmentation:
            train_transform = val_transform

        ds_train = BrainPatchDataset(data=data_path,
                            keys=train_keys,
                            info=info,
                            group=data_group,
                            column=infocolumn,
                            patch_size=patch_size,
                            preload=preload,
                            transform=train_transform)

        ds_val = BrainPatchDataset(data=data_path,
                            keys=val_keys, 
                            info=info,
                            column=infocolumn,
                            patch_size=patch_size,
                            group=data_group,
                            preload=preload,
                            transform=val_transform) 

    elif data_mode == 'volume':
        if np.any(crop_size):
            val_transform = CenterCropTransform(crop_size=crop_size, data_key='data')
        else:
            val_transform = None

        transforms = []
        if np.any(crop_size):
            transforms.append(RandomCropTransform(crop_size=crop_size, data_key='data', margins=crop_margins))
        transforms.append(GammaTransform(gamma_range=gamma_range, data_key='data'))
        if np.any(mirror_axis):
            transforms.append(MirrorTransform(axes=[mirror_axis], data_key='data'))
        train_transform = Compose(transforms)
        if not data_augmentation:
            train_transform = val_transform

        if dataset == 'brain':
            print("Brain")
            ds_train = BrainDataset(data=data_path,
                                keys=train_keys,
                                info=info,
                                group=data_group,
                                column=infocolumn,
                                preload=preload,
                                transform=train_transform)

            ds_val = BrainDataset(data=data_path,
                                keys=val_keys,
                                info=info,
                                column=infocolumn,
                                group=data_group,
                                preload=preload,
                                transform=val_transform)
        elif dataset == 'heart':
            ds_train = HeartDataset(data=data_path,
                                    keys=train_keys,
                                    info=info,
                                    group=data_group,
                                    column=infocolumn,
                                    preload=preload,
                                    transform=train_transform)

            ds_val = HeartDataset(data=data_path,
                                  keys=val_keys,
                                  info=info,
                                  column=infocolumn,
                                  group=data_group,
                                  preload=preload,
                                  transform=val_transform)

        elif dataset == 'fundus':
            ds_train = FundusDataset(data=data_path,
                                    keys=train_keys,
                                    info=info,
                                    group=data_group,
                                    column=infocolumn,
                                    preload=preload,
                                    transform=train_transform)

            ds_val = FundusDataset(data=data_path,
                                  keys=val_keys,
                                  info=info,
                                  column=infocolumn,
                                  group=data_group,
                                  preload=preload,
                                  transform=val_transform)

    if dataset == 'fundus':
        model = AgeModel2DChannels(args,
                     ds_train, ds_val, offline_wandb, log_model)
    else:
        model = AgeModel3DVolume(args,
                     ds_train, ds_val, offline_wandb, log_model)

    trainer = Trainer(logger=[wandb_logger], gpus=args.gpus, max_epochs=args.max_epochs, benchmark=args.benchmark, val_check_interval=args.val_check_interval)
    trainer.fit(model)


if __name__ == '__main__':
    args = train_args()
    main(args)
