from pointcept.engines.defaults import (
    default_argument_parser,
    default_setup,
    default_config_parser,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch

cfg_path = "./Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py"
weights_path = "./models/model_best_PointTransformer_V3.pth"
DATASET_ROOT = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"


def main_worker(cfg):
    
    cfg = default_setup(cfg)

    test_cfg = dict(cfg=cfg, **cfg.test)
    cfg.test.dataset_root = DATASET_ROOT

    tester = TESTERS.build(test_cfg)
    tester.test()


def main():
    cfg = default_config_parser(cfg_path)

    launch(
        main_worker,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url='auto',
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()