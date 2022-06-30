import argparse

def fix_dict_in_wandb_config(wandb):
    """"Adapted from [https://github.com/wandb/client/issues/982]"""
    config = dict(wandb)
    for k, v in config.copy().items():
        if '.' in k:
            keys = k.split('.')
            if len(keys) == 2:
                new_key = k.split('.')[0]
                inner_key = k.split('.')[1]
                if new_key not in config.keys():
                    config[new_key] = {}
                config[new_key].update({inner_key: v})
                del config[k]
            elif len(keys) == 3:
                new_key_1 = k.split('.')[0]
                new_key_2 = k.split('.')[1]
                inner_key = k.split('.')[2]
                
                if new_key_1 not in config.keys():
                    config[new_key_1] = {}
                if new_key_2 not in config[new_key_1].keys():
                    config[new_key_1][new_key_2] = {}
                config[new_key_1][new_key_2].update({inner_key: v})
                del config[k]
            else: # len(keys) > 3
                raise ValueError('Nested dicts with depth>3 are currently not supported!')
    
    wandb.config = wandb.Config()
    for k, v in config.items():
        wandb.config[k] = v

def train_args():
    """Load train arguments
    """

    parser = argparse.ArgumentParser(description="Train age prediction network")
    parser.add_argument("--name", default="ukb_age_fundus", type=str, help="Jobname")
    parser.add_argument("--gpus", default=4, type=int, help="Number of GPUS")
    parser.add_argument("--max_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--benchmark", default=True, type=bool, help="Benchmark")
    parser.add_argument("--val_check_interval", default=1.0, type=float, help="Val Check Intervall")


    parser.add_argument("--model", default='resnet18', type=str, help="Model Name")
    parser.add_argument("--depth", default=18, type=int, help="Model Depth")
    parser.add_argument("--inputs", default=3, type=int, help="Inputs")
    parser.add_argument("--outputs", default=2, type=int, help="Output")
    parser.add_argument("--pretrained", default=True, type=bool, help="Pretrained")
    parser.add_argument("--loss", default="l2", type=str, help="Loss function")
    parser.add_argument("--heteroscedastic", default=True, type=bool, help="Heteroscedastic")
    parser.add_argument("--position", default=False, type=bool, help="Position")
    parser.add_argument("--norm", default=None, type=str, help="Norm")
    parser.add_argument("--strides", default=[1,1,1,2], type=list, help="Strides")
    parser.add_argument("--nomaxpool", default=False, type=bool, help="Dont use max pool")
    parser.add_argument("--use_layer", default=3, type=int, help="Layers to use")

    

    parser.add_argument("--lr", default=1e-4, type=float, help="Learning Rate")
    parser.add_argument("--wd", default=0, type=float, help="Weight Decay")

    parser.add_argument("--batch", default=8, type=int, help="Batch Size")
    parser.add_argument("--workers", default=12, type=int, help="Num Workers")
    
    parser.add_argument("--mode", default="volume", type=str, help="Data Mode")
    parser.add_argument("--datapath", default="/mnt/qdata/ukb/UKB/interim/ukb_fundus_preprocessed.h5", type=str, help="Path to Data")
    parser.add_argument("--group", default="image", type=str, help="Data Group")
    parser.add_argument("--info", default="/mnt/qdata/ukb/UKB/interim/ukb_all.csv", type=str, help="Data Info")
    parser.add_argument("--column", default="age", type=str, help="Data Info Column")
    parser.add_argument("--preload", default=False, type=bool, help="Preload Data")
    parser.add_argument("--fold", default=0, type=int, help="Fold")
    parser.add_argument("--train", default="/mnt/qdata/ukb/UKB/interim/keys/train_fundus.dat", type=str, help="Train Split")
    parser.add_argument("--val", default="/mnt/qdata/ukb/UKB/interim/keys/test_fundus.dat", type=str, help="Val Split")

    parser.add_argument("--augmentation", default=True, type=bool, help="Data Augmentation")
    parser.add_argument("--cropsize", default=None, type=int, help="Crop Size")
    parser.add_argument("--cropmargins", default=None, type=int, help="Crop Margins")
    parser.add_argument("--gammarange", default=[0.9, 1.1], type=list, help="Gamma Range")
    parser.add_argument("--patchsize", default=None, type=int, help="Patch Size")
    parser.add_argument("--mirror", default=None, type=int, help="Mirror Axis")

    parser.add_argument("--NODE_RANK", type=int, default=-1, metavar="N", help="Local process rank.")










    args = parser.parse_args()
    return args




