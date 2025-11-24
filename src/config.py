# MobileNet model configuration
NUM_CLASSES = 1000       # for ImageNet
INPUT_SIZE = 224         # Input image size (H×W)
IN_CHANNELS = 3          # RGB input

# Width multiplier (α) ve Resolution multiplier (ρ) 
WIDTH_MULTIPLIER = 1.0
RESOLUTION_MULTIPLIER = 1.0

# Training-related parameters 
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model block configuration
BLOCKS_CONFIG = [
    # (in_channels, out_channels, stride)
    (32, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 512, 2),
    # repeat 512-block 5 times
    (512, 512, 1, 5),
    (512, 1024, 2),
    (1024, 1024, 1),
]

