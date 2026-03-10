import sys
import os
from pathlib import Path
import torch
from pytorch_lightning import seed_everything
import zarr
import numpy as np
from tqdm.auto import tqdm
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


def gradcam_volume():
    # copy config
    cfg, args = train_args()
    job = cfg['project']['job']
    data_path = cfg['dataset']['data']
    data_group = cfg['dataset']['group']
    info = cfg['dataset']['info']
    infocolumn = cfg['dataset']['column']
    train_set = cfg['dataset']['train']
    val_set = cfg['dataset']['gradcam']
    debug_set = cfg['dataset']['debug'] or None
    train_set = val_set
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

    # get job informations
    seed = 42
    seed_everything(seed)

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
    train_keys = [l.strip() for l in Path(train_set).open().readlines()]
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]

    # transforms
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

    # load train and val data
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

    print('load model...')
    device = torch.device('cuda')

    if dataset == 'fundus':
        model = AgeModel2DChannels(cfg, ds_train, ds_val, offline_wandb, log_model, dataset)
    else:
        model = AgeModel3DVolume(cfg, ds_train, ds_val, offline_wandb, log_model, dataset)

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
    model.eval()
    model.to(device)

    out_dir = Path(os.path.join(os.environ['RESULTS'], job))
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

    # compute sigma activation heatmaps
    for step, sample in tqdm(enumerate(model.dataloader(ds_val))):
        x = sample['data'].float()
        x = x.to(device)
        y = sample['label'][0].float()
        key = sample['key'][0]
        
        y_hat, heatmap = model.gradcam(x, channel=1)

        # store heatmap/image to zarr
        hmap = heatmap.cpu().numpy().astype(np.float32)
        ds = heatmaps_sigma.zeros(key, shape=hmap.shape, chunks=False, dtype=hmap.dtype, overwrite=True)
        ds[:] = hmap


if __name__ == '__main__':
    gradcam_volume()