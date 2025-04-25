"""
Main entry point for Sonata Test-Time Adaptation.
"""

import argparse
from adaptation.sonata_test_time_adapter import SonataTestTimeAdapter
import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sonata Test-Time Adaptation for Point Cloud Segmentation"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=config.MODEL_PATH,
        help="Path to the model weights"
    )
    parser.add_argument(
        "--seg_head_path", 
        type=str, 
        default=config.SEG_HEAD_PATH,
        help="Path to the segmentation head weights"
    )
    parser.add_argument(
        "--data_root", 
        type=str, 
        default=config.DATA_ROOT,
        help="Root directory containing dataset"
    )
    parser.add_argument(
        "--memory_size", 
        type=int, 
        default=config.MEMORY_SIZE,
        help="Maximum size of scene memory"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=config.NUM_SAMPLES,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--tta_epochs", 
        type=int, 
        default=config.TTA_EPOCHS,
        help="Number of epochs for test-time adaptation"
    )
    parser.add_argument(
        "--viz_output", 
        type=str, 
        default=config.VIZ_OUTPUT_PATH,
        help="Path to save visualization"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=config.SEED,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize and run the adapter
    adapter = SonataTestTimeAdapter(
        model_path=args.model_path,
        seg_head_path=args.seg_head_path,
        data_root=args.data_root,
        memory_size=args.memory_size,
        seed=args.seed
    )
    
    adapter.run(
        num_samples=args.num_samples, 
        memory_tta_epochs=args.tta_epochs,
        viz_output_path=args.viz_output
    )


if __name__ == "__main__":
    main()