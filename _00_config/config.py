import torch

class Config:
    # ========== 1. Hardware ==========
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device: GPU if available, otherwise CPU

    # ========== 2. Paths ==========
    ANN_FILE = "_01_data/result_0.json"                         # Path to annotation file in COCO format
    IMAGE_ROOT = ""     # Root directory containing training images
    LABEL_MAP_FILE = "_01_data/labels.json"                     # Path to class label mapping file

    # ========== 3. Architecture ==========
    NUM_CLASSES = 2             # Number of object classes (excluding background)
    INPUT_SIZE = (480, 640)     # Input image size (height, width)
    PRIOR_P = 0.01              # Confidence prior probability for anchor initialization
    STAGE_CHANNELS = [48, 96, 192, 192]   # Channel dimensions for each FPN stage (Original [64,128,256,256]), 4 stages only
    NUM_ANCHORS = [4, 4, 6, 6]            # Number of anchor boxes per location at each feature map scale
    FEATURE_MAPS = {       # This is based on the input size and the downsampling factor of each stage in model
        'f1': (120, 160),  # Feature map 1 dimensions (height, width) - 1/4 scale
        'f2': (60, 80),    # Feature map 2 dimensions - 1/8 scale
        'f3': (30, 40),    # Feature map 3 dimensions - 1/16 scale
        'f4': (15, 20),    # Feature map 4 dimensions - 1/32 scale
    }

    # ========== 4. Anchor ==========
    ANCHOR_SCALES = [0.1, 0.2, 0.4, 0.8]  # Scale multipliers for anchor boxes at each feature level
    ASPECT_RATIOS = [
        [1.0, 2.0, 0.5],              # Aspect ratios for feature map f1
        [1.0, 2.0, 0.5],              # Aspect ratios for feature map f2
        [1.0, 2.0, 0.5, 3.0, 0.33],   # Aspect ratios for feature map f3 (more shapes for smaller objects)
        [1.0, 2.0, 0.5, 3.0, 0.33],   # Aspect ratios for feature map f4
    ]

    # ========== 5. Data Augmentation ==========
    AUG_FACTOR = 8                                  # Data augmentation multiplier (creates 8x more augmented samples)
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]      # ImageNet mean for RGB normalization
    NORMALIZATION_STD = [0.229, 0.224, 0.225]       # ImageNet std for RGB normalization
    AUGMENTATION_PHOTOMETRIC_P = 0.5                # Probability of applying photometric augmentation (brightness, contrast, etc.)
    AUGMENTATION_HFLIP_P = 0.5                      # Probability of random horizontal flip
    AUGMENTATION_VFLIP_P = 0.5                      # Probability of random vertical flip
    AUGMENTATION_ROTATION_DEGREES = 15              # Maximum degrees for random rotation
    AUGMENTATION_GRAYSCALE_P = 0.05                 # Probability of converting image to grayscale
    AUGMENTATION_BLUR_KERNEL_SIZE = 3               # Gaussian blur kernel size
    AUGMENTATION_BLUR_SIGMA = (0.1, 2.0)            # Gaussian blur sigma range
    AUGMENTATION_ZOOMOUT_P = 0.5                    # Probability of random zoom out
    AUGMENTATION_ZOOMOUT_FILL = 0                   # Fill color for zoom out augmentation
    AUGMENTATION_ZOOMOUT_SIDE_RANGE = (1.0, 3.0)    # Side range for zoom out augmentation

    # ========== 6. Training ==========
    K_FOLD = 5                          # Number of folds for K-fold cross-validation
    LR = 1e-3                           # Initial learning rate
    EPOCHS = 40                         # Total number of training epochs
    BATCH_SIZE = 32                     # Samples per batch
    SCHEDULER_MILESTONES = [20, 32]     # Epochs at which to reduce learning rate
    SCHEDULER_GAMMA = 0.3               # Learning rate decay factor (multiply by this value at milestones)
    WEIGHT_DECAY = 1e-4                 # L2 regularization coefficient to prevent overfitting
    DROPOUT_P = 0.05                    # Dropout probability in classification head
    GRAD_CLIP = 5.0                     # Gradient clipping threshold to prevent exploding gradients
    CACHE_REFRESH_FREQUENCY = 20        # Refresh training data cache every N epochs
    PRINT_FREQUENCY = 10                # Print training logs every N epochs
    BEST_MODEL_SAVE_EPOCHS = 20         # Consider best model from last N epochs
    EARLY_STOP_PATIENCE = 8             # Stop training if no improvement in validation loss for N epochs

    # ========== 7. Loss ==========
    ALPHA = 1.0                 # Weighting factor for focal loss
    NEG_RATIO = 3               # Ratio of negative to positive samples in hard negative mining
    OVERLAP_THRESHOLD = 0.5     # IoU threshold for matching anchors to ground truth boxes
    VARIANCES = [0.1, 0.2]      # Variance for bounding box coordinate encoding/decoding

    # ========== 8. Inference ==========
    WEIGHT_PATHS = [                                            # List of model paths for ensemble inference
        "_04_training/models/best_1.pth",
        #"_04_training/models/latest_1.pth",
        "_04_training/models/best_2.pth",
        #"_04_training/models/latest_2.pth",
        "_04_training/models/best_3.pth",
        #"_04_training/models/latest_3.pth",
        "_04_training/models/best_4.pth",
        #"_04_training/models/latest_4.pth",
        "_04_training/models/best_5.pth",
        #"_04_training/models/latest_5.pth",
    ]

    ENSEMBLE_NMS_THRESHOLD = 0.5         # IoU threshold for NMS during box aggregation

    CONF_THRESHOLD = 0.5                # Confidence threshold for detection filtering
    NMS_OVERLAP_THRESHOLD = 0.5         # IoU threshold for Non-Maximum Suppression
    NMS_TOP_K = 100                     # Maximum number of boxes to keep after NMS
    CAMERA_WIDTH = 640                  # Camera frame width in pixels
    CAMERA_HEIGHT = 480                 # Camera frame height in pixels
    DETECTION_COLORS = [
        (0, 255, 0),      # Green
        (0, 165, 255),    # Orange
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (0, 0, 255),      # Red
        (255, 128, 0),    # Blue
    ]                                  # BGR color palette for visualization of detected objects

