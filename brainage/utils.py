import argparse
import yaml
import re
import os

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


def loadYaml(cfile, experiment=None):
    """
    [SOURCE] https://medium.com/swlh/python-yaml-configuration-with-environment-variables-parsing-77930f4273ac
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile('.*?\${(\w+)}.*?')
    # loader = yaml.SafeLoader  # SafeLoader broken in pyyaml > 5.2, see https://github.com/yaml/pyyaml/issues/266
    loader = yaml.Loader
    tag = "!ENV"

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    with open(cfile, 'r') as f:
        config = yaml.load(f, Loader=loader)
        if experiment is not None:
            config = config[experiment]

        var_list = [(k, config[k]) for k in config.keys() if k.startswith('__') and k.endswith('__')]

        for var_key, var_val in var_list:
            del config[var_key]

        def replace(config, var_key, var_val):
            for key in config.keys():
                if isinstance(config[key], str):
                    config[key] = config[key].replace(var_key, f'{var_val}')
                if isinstance(config[key], dict):
                    replace(config[key], var_key, var_val)

        for var_key, var_val in var_list:
            replace(config, var_key, var_val)

    return config


def train_args():
    """Load train arguments
    """

    parser = argparse.ArgumentParser(description="Train age prediction network")
    parser.add_argument("--name", default="ukb_age_fundus", type=str, help="Jobname")
    parser.add_argument("--config", default="config/volume/config_ukb_brain.yaml", type=str, help="YAML config file")
    parser.add_argument("--gpus", default=4, type=int, help="Number of GPUS")
    parser.add_argument("--max_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("--benchmark", default=True, type=bool, help="Benchmark")
    parser.add_argument("--val_check_interval", default=1.0, type=float, help="Val Check Intervall")
    parser.add_argument("--predict", default=None, type=str, help="Prediction log for checkpoints (if provided prediction is run)")

    parser.add_argument("--modelName", default='resnet18', type=str, help="Model Name")
    parser.add_argument("--depth", default=18, type=int, help="Model Depth")
    parser.add_argument("--inputs", default=3, type=int, help="Inputs")
    parser.add_argument("--outputs", default=2, type=int, help="Output")
    parser.add_argument("--pretrained", default=True, type=bool, help="Pretrained")
    parser.add_argument("--loss", default="l2", type=str, help="Loss function")
    parser.add_argument("--heteroscedastic", default=True, type=bool, help="Heteroscedastic")
    parser.add_argument("--position", default=False, type=bool, help="Position")
    parser.add_argument("--norm", default=None, type=str, help="Norm")
    parser.add_argument("--strides", default=[1,1,1,2], type=list, help="Strides")
    parser.add_argument("--no_max_pool", default=False, type=bool, help="Dont use max pool")
    parser.add_argument("--use_layer", default=3, type=int, help="Layers to use")

    

    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning Rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight Decay")

    parser.add_argument("--batch_size", default=8, type=int, help="Batch Size")
    parser.add_argument("--num_workers", default=12, type=int, help="Num Workers")
    
    parser.add_argument("--mode", default="volume", type=str, help="Data Mode")
    parser.add_argument("--data", default="/mnt/qdata/ukb/UKB/interim/ukb_fundus_preprocessed.h5", type=str, help="Path to Data")
    parser.add_argument("--group", default="image", type=str, help="Data Group")
    parser.add_argument("--info", default="/mnt/qdata/ukb/UKB/interim/ukb_all.csv", type=str, help="Data Info")
    parser.add_argument("--column", default="age", type=str, help="Data Info Column")
    parser.add_argument("--preload", default=False, type=bool, help="Preload Data")
    parser.add_argument("--fold", default=0, type=int, help="Fold")
    parser.add_argument("--train", default="/mnt/qdata/ukb/UKB/interim/keys/train_fundus.dat", type=str, help="Train Split")
    parser.add_argument("--val", default="/mnt/qdata/ukb/UKB/interim/keys/test_fundus.dat", type=str, help="Val Split")

    parser.add_argument("--data_augmentation", default=True, type=bool, help="Data Augmentation")
    parser.add_argument("--crop_size", default=None, type=int, help="Crop Size")
    parser.add_argument("--crop_margins", default=None, type=int, help="Crop Margins")
    parser.add_argument("--gamma_range", default=[0.9, 1.1], type=list, help="Gamma Range")
    parser.add_argument("--patch_size", default=None, type=int, help="Patch Size")
    parser.add_argument("--mirror_axis", default=None, type=int, help="Mirror Axis")


    args = parser.parse_args()
    config = loadYaml(args.config)
    for arg in vars(args):
        if arg in config:
            print(f'Overriding {arg} from argparse')
        config[arg] = getattr(args, arg)
    return config, args




