"""Provider for data transformations."""

import sonata
from pointcept.datasets.transform import Compose


class TransformProvider:
    """Class to manage data transforms"""
    
    @staticmethod
    def get_default_transform():
        """
        Get default transform
        
        Returns:
            Default transformation pipeline
        """
        return sonata.transform.default()
    
    @staticmethod
    def get_test_time_transform():
        """
        Get transform for test-time augmentation
        
        Returns:
            Test-time transformation pipeline
        """
        ttt_config = [
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        
        return Compose(ttt_config)