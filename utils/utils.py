import torch
import  os
import sys 
from dotenv import load_dotenv
from collections import Counter


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# apped parent directory to sys.path to import utils
sys.path.append(parent_dir)

FEATURE_FILES = []
FEATURE_PATHS = []
features_dir = os.path.join(parent_dir, "feature_extraction", "output")
if not os.path.exists(features_dir):
    print(f"WARNING: {features_dir} not exists. Before the analyze, you must extract features")
else:
    sys.path.append(features_dir)
    FEATURE_FILES = [f for f in os.listdir(features_dir) if f.endswith('.ft')]
    FEATURE_PATHS = [os.path.join(features_dir, f) for f in FEATURE_FILES]



load_dotenv()

TORCH_MODEL_CACHE = os.path.join(os.getenv("TORCH_MODEL_CACHE"))
torch.hub.set_dir(TORCH_MODEL_CACHE)




CUSTOM_WEIGHT_CACHE = os.path.join(os.getenv("CUSTOM_WEIGHT_CACHE"), "checkpoints")
CUSTOM_WEIGHT_FILES = []
CUSTOM_WEIGHT_PATHS = []

if not os.path.exists(CUSTOM_WEIGHT_CACHE):
    os.mkdir(os.getenv("CUSTOM_WEIGHT_CACHE"))
    os.mkdir(CUSTOM_WEIGHT_CACHE)    
    print(f"WARNING: {CUSTOM_WEIGHT_CACHE} not exists. Created {CUSTOM_WEIGHT_CACHE} but no custom weights exists")    
else:
    # find in subdirectories of custom_models_dir *.pth files

    subdir_list = [os.path.join(CUSTOM_WEIGHT_CACHE, o) for o in os.listdir(CUSTOM_WEIGHT_CACHE) if os.path.isdir(os.path.join(CUSTOM_WEIGHT_CACHE,o))]
    for subdir in subdir_list:
        for f in os.listdir(subdir):
            if f.endswith(".pth"):
                CUSTOM_WEIGHT_FILES.append(f)
                CUSTOM_WEIGHT_PATHS.append(os.path.join(subdir, f))


DATASET_DIR = os.getenv("DATASET_CACHE")
DATASET_DIR = os.path.join(DATASET_DIR)
DATASET_FILES = []
DATASET_PATHS = []
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)
    print(f"WARNING: {DATASET_DIR} not exists. Created {DATASET_DIR} but no dataset exists")
else:
    DATASET_FILES = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pt')]
    DATASET_PATHS = [os.path.join(DATASET_DIR, f) for f in DATASET_FILES]

def find_by_keyword(keyword, path_list):
    for path in path_list:
        if keyword in path:
            return path
    return None

def load_custom_weight(keyword):
    path = find_by_keyword(keyword, CUSTOM_WEIGHT_PATHS)
    if path is None:
        raise ValueError(f"keyword {keyword} not found")
    weights = torch.load(path)["state_dict"]
    return weights

def load_features(keyword):
    path = find_by_keyword(keyword, FEATURE_PATHS)
    if path is None:
        raise ValueError(f"keyword {keyword} not found")
    features = torch.load(path)
    return features

def load_dataset(keyword):
    path = find_by_keyword(keyword, DATASET_PATHS)
    if path is None:
        raise ValueError(f"keyword {keyword} not found")
    dataset = torch.load(path)
    return dataset




__all__ = [
    "FEATURE_FILES",
    "FEATURE_PATHS",
    "CUSTOM_WEIGHT_FILES",
    "CUSTOM_WEIGHT_PATHS",
    "DATASET_FILES",
    "DATASET_PATHS",
    "TORCH_MODEL_CACHE",
    "load_custom_weight", 
    "load_features", 
    "load_dataset"
    ]