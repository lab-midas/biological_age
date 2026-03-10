import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose

sys.path.append('/home/raeckev1/nako_ukb_age')
sys.path.append('/home/raecker1/nako_ukb_age')

from brainage.model.model3d import AgeModel3DVolume
from brainage.model.model2d import AgeModel2DChannels
from brainage.dataset.dataset3d import BrainDataset, HeartDataset, AbdomenDataset
from brainage.dataset.dataset2d import FundusDataset
from brainage.utils import train_args, loadYaml


config = os.getenv('CONFIG')

def main():
    # copy config
    cfg, args = train_args()
    project = cfg['project']['name']
    job = cfg['project']['job']
    data_path = cfg['dataset']['data']
    data_group = cfg['dataset']['group']
    info = cfg['dataset']['info']
    infocolumn = cfg['dataset']['column']
    train_set = cfg['dataset']['train']

    if args.predict:
        val_set = cfg['dataset']['pred']
    else:
        val_set = cfg['dataset']['val']

    debug_set = cfg['dataset']['debug'] or None

    data_augmentation = cfg['dataset']['data_augmentation']
    crop_size = np.array(cfg['dataset']['crop_size'])
    crop_margins = np.array(cfg['dataset']['crop_margins'])
    gamma_range = cfg['dataset']['gamma_range']
    mirror_axis = cfg['dataset']['mirror_axis']
    preload = cfg['dataset']['preload']
    meta = cfg['model']['position']

    # debugging
    if debug_set:
        train_set = debug_set
        val_set = debug_set
        offline_wandb = True
        log_model = False
    else:
        offline_wandb = False
        log_model = True
        
    if args.predict:
        offline_wandb = True
        log_model = False

    seed = 42
    seed_everything(seed)
    ts = time.gmtime()
    job_id = 'fold' + f'-{cfg["dataset"]["fold"]}-' + time.strftime("%Y-%m-%d-%H-%M-%S", ts)
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
    elif 'pancreas' in job:
        dataset = 'abdominal'
    elif 'fundus' in job:
        dataset = 'fundus'

    if 'ukb' in Path(data_path).stem:
        ukb = True
        cohort = 'UK Biobank'
    elif 'nako' in Path(data_path).stem:
        ukb = False
        cohort = 'NAKO'
    else:
        raise ValueError("Dataset not found")
    
    print("=====================")
    print("Job: ", job)
    print("Dataset: ", dataset)
    print("Cohort: ", cohort)
    print("=====================")

    # get keys 
    train_keys = [str(l.strip()) for l in Path(train_set).open().readlines()]
    val_keys = [str(l.strip()) for l in Path(val_set).open().readlines()]

    # define transforms
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

    # get train and val datasets
    if dataset == 'brain':
        ds_train = BrainDataset(data=data_path,
                                keys=train_keys,
                                info=info,
                                group=data_group,
                                column=infocolumn,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=train_transform)

        ds_val = BrainDataset(data=data_path,
                                keys=val_keys,
                                info=info,
                                column=infocolumn,
                                group=data_group,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=val_transform)
    elif dataset == 'heart':
        ds_train = HeartDataset(data=data_path,
                                keys=train_keys,
                                info=info,
                                group=data_group,
                                column=infocolumn,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=train_transform)

        ds_val = HeartDataset(data=data_path,
                                keys=val_keys,
                                info=info,
                                column=infocolumn,
                                group=data_group,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=val_transform)

    elif dataset == 'abdominal':
        ds_train = AbdomenDataset(data=data_path,
                                keys=train_keys,
                                info=info,
                                group=data_group,
                                column=infocolumn,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=train_transform)

        ds_val = AbdomenDataset(data=data_path,
                                keys=val_keys,
                                info=info,
                                column=infocolumn,
                                group=data_group,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=val_transform)

    elif dataset == 'fundus':
        ds_train = FundusDataset(data=data_path,
                                keys=train_keys,
                                info=info,
                                group=data_group,
                                column=infocolumn,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=train_transform)

        ds_val = FundusDataset(data=data_path,
                                keys=val_keys,
                                info=info,
                                column=infocolumn,
                                group=data_group,
                                preload=preload,
                                meta=meta,
                                ukb=ukb,
                                transform=val_transform)

    # initialize model
    if dataset == 'fundus':
        model = AgeModel2DChannels(cfg, ds_train, ds_val, offline_wandb, log_model, dataset)
    else:
        model = AgeModel3DVolume(cfg, ds_train, ds_val, offline_wandb, log_model, dataset)

    # initialize wandb logger
    if not offline_wandb:
        wandb_logger = [WandbLogger(name=f'{job}-{job_id}', 
                                    entity='veronika-ecker', 
                                    project=project, 
                                    offline=offline_wandb, 
                                    log_model=log_model)]
    else:
        wandb_logger = False

    # inference
    if args.predict is not None:

        # load model checkpoint
        ckpt_config = loadYaml(args.predict)
        ckpt_path = os.path.join(os.environ['CKPT'], ckpt_config['checkpoints'][job][0], 'checkpoints')

        if 'best' in str(os.environ['OUT']):    # load best model checkpoint (best val loss)
            ckpt_path = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith('.ckpt') and f.startswith('best-val-loss')]
            print(f'Loading best model checkpoint from {ckpt_path[0]}')
        else:                                   # load last checkpoint (last epoch)
            ckpt_path = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith('.ckpt') and 'epoch=199' in f]
            print(f'Loading last model checkpoint from {ckpt_path[0]}')

        model.load_state_dict(torch.load(str(ckpt_path[0]))['state_dict'])
        trainer = Trainer(accelerator='gpu', 
                          devices=1, 
                          strategy="ddp")

        # start inference
        trainer.predict(model, model.dataloader(ds_val))
        result_path = os.path.join(os.environ['OUT'], job + '_val.csv')
        model.write_results(result_path)
        model.reset_results()

        trainer.predict(model, model.dataloader(ds_train))
        result_path = os.path.join(os.environ['OUT'], job + '_train.csv')
        model.write_results(result_path)

    else:  # training

        # define checkpoint directory
        ckpt_dir = os.environ.get('CKPT', 'checkpoints')
        checkpoint_dir = os.path.join(ckpt_dir, f'{job}-{job_id}', 'checkpoints')
        
        print(f'Checkpoints will be saved to: {checkpoint_dir}')
        
        # Create checkpoint callbacks
        # Save model with best validation loss
        checkpoint_callback_best = ModelCheckpoint(
            monitor='validation/loss',
            dirpath=checkpoint_dir,
            filename='best-val-loss-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
            save_last=False
        )
        
        # Save model from last epoch
        checkpoint_callback_last = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='last-epoch-{epoch:02d}',
            save_top_k=1,
            save_last=True
        )
        
        callbacks = [checkpoint_callback_best, checkpoint_callback_last]
        
        trainer = Trainer(logger=wandb_logger,
                          accelerator='gpu', 
                          devices=cfg['trainer']['gpus'], 
                          max_epochs=cfg['trainer']['max_epochs'],
                          benchmark=cfg['trainer']['benchmark'], 
                          val_check_interval=cfg['trainer']['val_check_interval'], 
                          strategy="ddp",
                          callbacks=callbacks)

        if trainer.global_rank == 0 and not offline_wandb:
            wandb_logger[0].experiment.config.update(cfg)

        if cfg['trainer']['resume'] is not None:
            trainer.fit(model, ckpt_path=cfg['trainer']['resume'])
        else:
            trainer.fit(model)


if __name__ == '__main__':
    main()