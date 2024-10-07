import os
import time
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
#from pytorch_lightning.loggers import WandbLogger
import zarr
import numpy as np
#import pandas as pd
#from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.abstract_transforms import Compose

import sys
sys.path.append('/home/raecker1/nako_ukb_age')
from brainage.model.model3d import AgeModel3DVolume
from brainage.model.model2d import AgeModel2DChannels
from brainage.dataset.dataset3d import BrainDataset, BrainPatchDataset, HeartDataset, AbdomenDataset
from brainage.dataset.dataset2d import FundusDataset
from brainage.utils import train_args, loadYaml

"""load_dotenv()
DATA = Path(os.getenv('DATA'))

@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))"""
config = os.getenv('CONFIG')

# @hydra.main(config_path=os.path.dirname(config), config_name=os.path.splitext(os.path.basename(config))[0])
def gradcam_volume():
    cfg, args = train_args()

    project = cfg['project']['name']
    job = cfg['project']['job']
    data_path = cfg['dataset']['data']
    data_group = cfg['dataset']['group']
    info = cfg['dataset']['info']
    infocolumn = cfg['dataset']['column']
    train_set = cfg['dataset']['train']
    val_set = cfg['dataset']['gradcam_val']
    debug_set = cfg['dataset']['debug'] or None
    train_set = val_set
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
    patch_size = cfg['dataset']['patch_size']
    data_mode = cfg['dataset']['mode']
    data_augmentation = cfg['dataset']['data_augmentation']
    crop_size = np.array(cfg['dataset']['crop_size'])
    crop_margins = np.array(cfg['dataset']['crop_margins'])
    gamma_range = cfg['dataset']['gamma_range']
    mirror_axis = cfg['dataset']['mirror_axis']
    preload = cfg['dataset']['preload']
    meta = cfg['model']['position']
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

    #neptune_logger = NeptuneLogger(project_name=f'lab-midas/{project}',
    #                               params=OmegaConf.to_container(cfg, resolve=True),
    #                               experiment_name=f'{job}-{job_id}',
    #                               tags=[job])
    
    # get keys and metadata
    train_keys = [l.strip() for l in Path(train_set).open().readlines()]
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]

    print("=====================")
    print("Job: ", job)
    print("Dataset: ", dataset)
    print("=====================")

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
            ds_train = BrainDataset(data=data_path,
                                keys=train_keys,
                                info=info,
                                group=data_group,
                                column=infocolumn,
                                preload=preload,
                                meta=meta,
                                transform=train_transform)

            ds_val = BrainDataset(data=data_path,
                                keys=val_keys,
                                info=info,
                                column=infocolumn,
                                group=data_group,
                                preload=preload,
                                meta=meta,
                                transform=val_transform)
        elif dataset == 'heart':
            ds_train = HeartDataset(data=data_path,
                                    keys=train_keys,
                                    info=info,
                                    group=data_group,
                                    column=infocolumn,
                                    preload=preload,
                                    meta=meta,
                                    transform=train_transform)

            ds_val = HeartDataset(data=data_path,
                                  keys=val_keys,
                                  info=info,
                                  column=infocolumn,
                                  group=data_group,
                                  preload=preload,
                                  meta=meta,
                                  transform=val_transform)

        elif dataset == 'abdominal':
            ds_train = AbdomenDataset(data=data_path,
                                    keys=train_keys,
                                    info=info,
                                    group=data_group,
                                    column=infocolumn,
                                    preload=preload,
                                    meta=meta,
                                    transform=train_transform)

            ds_val = AbdomenDataset(data=data_path,
                                  keys=val_keys,
                                  info=info,
                                  column=infocolumn,
                                  group=data_group,
                                  preload=preload,
                                  meta=meta,
                                  transform=val_transform)

        elif dataset == 'fundus':
            ds_train = FundusDataset(data=data_path,
                                    keys=train_keys,
                                    info=info,
                                    group=data_group,
                                    column=infocolumn,
                                    preload=preload,
                                    meta=meta,
                                    transform=train_transform)

            ds_val = FundusDataset(data=data_path,
                                  keys=val_keys,
                                  info=info,
                                  column=infocolumn,
                                  group=data_group,
                                  preload=preload,
                                  meta=meta,
                                  transform=val_transform)

    print('load model...')
    device = torch.device('cuda')
    if dataset == 'fundus':
        model = AgeModel2DChannels(cfg, ds_train, ds_val, offline_wandb, log_model, dataset)
    else:
        model = AgeModel3DVolume(cfg, ds_train, ds_val, offline_wandb, log_model, dataset)

    ckpt_config = loadYaml(args.predict)
    ckpt_path = os.path.join(os.environ['WGHT'], ckpt_config['checkpoints'][job][0], 'checkpoints')
    ckpt_path = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path) if f.endswith('.ckpt') and f.startswith('epoch=199')]
    model.load_state_dict(torch.load(str(ckpt_path[0]))['state_dict'])
    model.eval()
    model.to(device)

    out_dir = Path(f'/mnt/qdata/share/raecker1/age_experiments_new/{job}/')
    out_dir.mkdir(exist_ok=True)
    zarr_path = out_dir/'maps.zarr'
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    heatmaps_mean = root.require_group('heatmaps_mean')
    heatmaps_sigma = root.require_group('heatmaps_sigma')
    images = root.require_group('images')

    print('processing data')
    # compute mean activation heatmaps
    results = {'key': [], 'y': [], 'y_hat0': [], 'y_hat1': []}
    for step, sample in tqdm(enumerate(model.dataloader(ds_val))):
        x = sample['data'].float()
        x = x.to(device)
        y = sample['label'][0].float()
        key = sample['key'][0]
        y_hat, heatmap = model.gradcam(x, channel=0)

        # store heatmap/image to zarr
        hmap = heatmap.cpu().numpy().astype(np.float32)
        ds = heatmaps_mean.zeros(key ,shape=hmap.shape, chunks=False, dtype=hmap.dtype, overwrite=True)
        ds[:] = hmap 
        img = x.cpu().numpy().astype(np.float16)[0,0]
        ds = images.zeros(key ,shape=img.shape, chunks=False, dtype=img.dtype, overwrite=True)
        ds[:] = img 
        
        # store prediction
        results['key'].append(key)
        results['y'].append(y.item())
        results['y_hat0'].append(y_hat[0, 0].detach().cpu().item())
        results['y_hat1'].append(y_hat[0, 1].detach().cpu().item())
        if step == 50:
            break
    
    """# save predictions
    df = pd.DataFrame.from_dict(results)
    if (out_dir/f'predictions.feather').is_file():
        df_0 = pd.read_feather(out_dir/f'predictions.feather').set_index('key')
        df = df_0.combine_first(df.set_index('key'))
        df = df.reset_index()
    df.to_feather(out_dir/f'predictions.feather')"""

    # compute sigma activation heatmaps
    for step, sample in tqdm(enumerate(model.dataloader(ds_val))):
        x = sample['data'].float()
        x = x.to(device)
        y = sample['label'][0].float()
        key = sample['key'][0]
        y_hat, heatmap = model.gradcam(x, channel=1)

        # store heatmap/image to zarr
        hmap = heatmap.cpu().numpy().astype(np.float32)
        ds = heatmaps_sigma.zeros(key, shape=hmap.shape, chunks=False, dtype=hmap.dtype, doverwrite=True)
        ds[:] = hmap
        if step == 50:
            break

if __name__ == '__main__':
    gradcam_volume()

