import os

import yaml


# must be run from project root directory
def load_cfg():
    path = os.getcwd() + "/configs/config.yaml"
    cfg = yaml.safe_load(open(path))
    return cfg


if __name__ == "__main__":
    # Sanity check
    path = os.getcwd() + "/configs/config.yaml"
    cfg = yaml.safe_load(open(path))
    print(cfg)
