import numpy as np
import torch
import os

import logging
import argparse
from dataset_caption import imagenet_label
from torchvision import transforms as T
from utils import model_transfer, get_adv_images_loders, get_clean_images_loders
from transformers import ResNetForImageClassification
import torchvision.models as models


logger = logging.getLogger(__name__)



def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="logfiles", type=str,
                        help='Where to save the result of transferbility and asr')
    parser.add_argument('--attack_method', type=str, default="venom")
    parser.add_argument('--clean_images_root', default=None, type=str,
                        help='The clean images root directory')
    parser.add_argument('--clean_images_labels', default=None, type=str,
                        help='The directory of the labels of the clean images')
    parser.add_argument('--adv_images_root', default=None, type=str,
                        help='The adversarial images root directory')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--npz_file_path', type=str, default=None)
    parser.add_argument('--dataset_name', default="imagenet_compatible", type=str)

    arguments = parser.parse_args()
    return arguments


args = get_args()

out_dir = os.path.join(args.save_dir, args.attack_method)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)



adv_dir = args.adv_images_root
clean_dir = args.clean_images_root
label_dir = args.clean_images_labels

batch_size = args.batch_size




clean_loader = get_clean_images_loders(clean_dir, label_dir, batch_size)
adv_loader = get_adv_images_loders(adv_dir,batch_size)
method = adv_dir.split("/")[-2]

logger.setLevel(logging.INFO)

logfile_path = os.path.join(out_dir, method + ".log")
if os.path.exists(logfile_path):
    os.remove(logfile_path)

file_handler = logging.FileHandler(logfile_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


logger.info(f"\n Test transferbility of method: {method} ")

model_transfer(adv_loader, batch_size, res=224, logger=logger)










