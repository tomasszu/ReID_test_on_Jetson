from __future__ import print_function, division

import argparse
import math
import time
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import pandas as pd
import psutil
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

# from load_model_ModelArchChange_ForInfer_partial import load_model_from_opts
from load_model import load_model_from_opts
from dataset import ImageDataset
from tool.extract import extract_feature

torchvision_version = list(map(int, torchvision.__version__.split(".")[:2]))

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--model_opts",default="vehicle_reid_repo2/vehicle_reid/model/vric_unmodified/opts.yaml", 
                    type=str, help="model saved options")
parser.add_argument("--checkpoint", default="vehicle_reid_repo2/vehicle_reid/model/vric_unmodified/net_19.pth",
                    type=str, help="model checkpoint to load")
parser.add_argument("--query_csv_path", default="data/vric_query.csv",
                    type=str, help="csv to contain query image data")
parser.add_argument("--gallery_csv_path", default="data/vric_gallery.csv",
                    type=str, help="csv to contain gallery image data")
parser.add_argument("--data_dir", type=str, default='data',
                    help="root directory for image datasets")
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
opt = parser.parse_args()

# Function to print memory usage on the GPU
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

def get_memory_usage():
    process = psutil.Process(os.getpid())  # Get the current process
    return process.memory_info().rss  # Return resident memory (RAM in bytes)

def fliplr(img):
    """flip images horizontally in a batch"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, X, device="cuda"):
    """Exract the embeddings of a single image tensor X"""
    if len(X.shape) == 3:
        X = torch.unsqueeze(X, 0)
    X = X.to(device)
    feature = model(X).reshape(-1)

    X = fliplr(X)
    flipped_feature = model(X).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)

print("RAM Memory before inference: {:.2f} MB".format(get_memory_usage() / 1024 / 1024))


use_gpu = torch.cuda.is_available()
if not use_gpu:
    device = torch.device("cpu")
else:
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    device = torch.device("cuda")

######################################################################
# Load Data
# ---------
#
h, w = 224, 224
interpolation = 3 if torchvision_version[0] == 0 and torchvision_version[1] < 13 else \
    transforms.InterpolationMode.BICUBIC

data_transforms = transforms.Compose([
    transforms.Resize((h, w), interpolation=interpolation),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


query_df = pd.read_csv(opt.query_csv_path)
gallery_df = pd.read_csv(opt.gallery_csv_path)
classes = list(pd.concat([query_df["id"], gallery_df["id"]]).unique())

######################################################################
# Load model
# ----------

print("Initial Memory Usage:")
print_gpu_memory()

print('-------test-----------')
print("Running on: {}".format(device))

model = load_model_from_opts(opt.model_opts, ckpt=opt.checkpoint,
                             remove_classifier=True)
model.eval()
model.to(device)

print("RAM Memory after Loading: {:.2f} MB".format(get_memory_usage() / 1024 / 1024))

print("Load GPU Memory Usage:")
print_gpu_memory()

random.seed(420)

data_dir = opt.data_dir

image_datasets = ImageDataset(opt.data_dir, query_df, "id", classes, transform=data_transforms)

with torch.no_grad():
    try:
        while True: 
            image, _ = image_datasets[random.choice(range(0,2000))]
            since = time.time()

            query_feature = extract_feature(
            model, image , device)
            time_elapsed = time.time() - since
            print('Complete in {}ms'.format(
             time_elapsed * 1000))
            print("RAM Memory after inference: {:.2f} MB".format(get_memory_usage() / 1024 / 1024))
            print("Inference Memory Usage:")
            print_gpu_memory()
    except KeyboardInterrupt:
        pass

print("Last features = ")
print(query_feature)
torch.cuda.empty_cache()