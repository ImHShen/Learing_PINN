import torch
import numpy as np
import random

# Central place to determine and export the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional: Helper for reproducibility
def set_seed(seed):
    """Sets random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Potentially add settings for deterministic algorithms if needed
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False