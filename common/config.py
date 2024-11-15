import yaml
import os

def loadParam(config_file):
    # fold_path = os.path.split(os.path.realpath(__file__))[0] + "/.." # 当前路径
    # yaml_path = os.path.join(fold_path, config_file)
    yaml_path = config_file
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)