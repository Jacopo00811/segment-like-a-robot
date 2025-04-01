import yaml
import os

# must be run from project root directory
def get_cfg():
    path = os.getcwd() + "/configs/config.yaml"
    cfg = yaml.safe_load(open(path))
    print(cfg)

if __name__ == "__main__":
    path = os.getcwd() + "/configs/config.yaml"
    cfg = yaml.safe_load(open(path))
    print(cfg)