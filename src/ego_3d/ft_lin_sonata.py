from pointcept.engines.defaults import (
    default_argument_parser,
    default_setup,
    default_config_parser,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.utils.config import Config
import os
from pointcept.utils.env import get_random_seed, set_seed

cfg_path = "./Pointcept/configs/sonata/semseg-sonata-v1m1-0a-scannet-lin-ft.py"
WEIGHTS = "./models/sonata/pretrain-sonata-v1m1-0-base.pth"
DATASET_ROOT = "/dtu/blackhole/0e/169006/Mini-ScanNet/ego_sliced/preprocessed/"
SAVE_PATH = "./exp/ego/ft/sonata-lin"


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

    # Explicitly set the data root for each split
    cfg.data.test.data_root = DATASET_ROOT
    cfg.data.val.data_root = DATASET_ROOT
    cfg.data.train.data_root = DATASET_ROOT
    
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = config_parser(cfg_path, None)

    cfg.epoch = 10
    cfg.eval_epoch = 10
    cfg.data.train.loop = 1
    
    cfg.test = dict(
        type='SemSegTester',
        verbose=True
    )
    
    for i, hook in enumerate(cfg.hooks):
        if hook.get('type') == 'PreciseEvaluator':
            cfg.hooks[i] = dict(type='CheckpointSaver', save_freq=None)
    
    launch(
        main_worker,    
        num_gpus_per_machine=1,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )

if __name__ == "__main__":
    main()
