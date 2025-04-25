"""Configuration parameters for the test-time adaptation."""

# Paths
MODEL_PATH = "./models/sonata/sonata.pth"
SEG_HEAD_PATH = "./models/sonata/sonata_linear_prob_head_sc.pth"
DATA_ROOT = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"

# Memory settings
MEMORY_SIZE = 5

# Training settings
LEARNING_RATE = 1e-5
TTA_EPOCHS = 2
NUM_SAMPLES = 20

# Visualization settings
VIZ_OUTPUT_PATH = "tta_memory_sem_seg_last_sample.html"

# Random seed for reproducibility
SEED = 24525867