import sonata
import torch
from pointcept.utils.config import Config

from pointcept.engines.test import TESTERS
from pointcept.engines.otta import SemSegTester
import os
from pointcept.engines.defaults import (
    default_argument_parser,
    default_setup,
    default_config_parser,
)

from pointcept.engines.launch import launch
from pointcept.utils.env import get_random_seed, set_seed

# cfg_path = "./Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py"
cfg_path = "./Pointcept/configs/scannet/semseg-pt-v3m2-0-sonata-scratch.py"
WEIGHTS = "./models/sonata/sonata.pth"
DATASET_ROOT = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"
SAVE_PATH = "./exp/scannet/sonata-debug"
FEAT_WEIGHTS = "./models/sonata/sonata.pth"

def config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    cfg.save_path = SAVE_PATH

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg

def main_worker(cfg):

    cfg = default_setup(cfg)

    test_cfg = dict(cfg=cfg, **cfg.test)
    cfg.data.test.data_root = DATASET_ROOT
    cfg.data.val.data_root = DATASET_ROOT

    cfg.weight = WEIGHTS

    device = torch.device("cuda")

    feat_model = sonata.model.load(FEAT_WEIGHTS)

    feat_model = feat_model.to(device)
    tester = SemSegTester(
        cfg=cfg,
        model=feat_model,
    )
    tester.test()


def main():
    cfg = config_parser(cfg_path, None)

    launch(
        main_worker,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        cfg=(cfg,),
    )

if __name__ == "__main__":
    main()