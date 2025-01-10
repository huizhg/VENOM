import pyiqa
import torch
import logging
import os
import argparse
from torchvision import datasets, transforms
from PIL import Image
import glob
from utils import *
from utils import key2label
from torchvision import transforms
from PIL import Image
from natsort import ns, natsorted

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--save_dir', default="logfiles/image_quality", type=str,
                    help='Where to save the result of transferbility and asr')
parser.add_argument('--metric_name', type=str, default="fid")

args= parser.parse_args()






out_dir = args.save_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

metric_name = args.metric_name

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Set up logger
logger.setLevel(logging.INFO)
logfile_path = os.path.join(out_dir, metric_name + ".log")
if os.path.exists(logfile_path):
    os.remove(logfile_path)

file_handler = logging.FileHandler(logfile_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

logger.info(f"Image Quality Assessment with {metric_name} score")
metric = pyiqa.create_metric(metric_name, device=device)
logger.info(f"Lower_better: {metric.lower_better}")
print(metric.lower_better)

clean_dir = "path-to-clean-images"
label_dir = "path-to-the-labels-of-clean-images"
adv_images_dir = "path-to-the-adversarial-images"

transform = transforms.Compose([
                            transforms.Lambda(lambda x: x.convert("RGB")),
                            transforms.Resize((256,256)),
                            transforms.CenterCrop((224,224)),
                            transforms.ToTensor(),
                        ])




clean_image_paths = glob.glob(os.path.join(clean_dir, "*"))
clean_image_paths = natsorted(clean_image_paths, alg=ns.PATH)

# only fid score take the whole folder, other metrics take the image path

if metric_name == "fid":
    score = metric(adv_images_dir, clean_dir)
    logger.info(f"{metric_name} score for the adversarial images is {score}")

elif metric_name == "clipscore":
    adv_loader = get_adv_images_loders(adv_images_dir, batch_size=1)
    sum = 0
    for ind, (img, label) in enumerate(adv_loader):
        img = img.to(device)
        caption = key2label[label.squeeze(0).item()]
        caption_list = [caption]

        sum += metric(img,caption_list=caption_list)
    avg_score = sum / len(glob.glob(os.path.join(dir, "*")))
    logger.info(f"{metric_name} score for adversarial images is {avg_score}")
    
elif metric_name == "brisque":
    sum = 0 
    adv_image_paths = glob.glob(os.path.join(dir, "*"))
    for ind, adv_image_path in enumerate(adv_image_paths):
        sum += metric(adv_image_path, clean_image_paths[ind])
    avg_score = sum / len(adv_image_paths)
    logger.info(f"{metric_name} score for adversarial images is {avg_score}")
    
else:
    adv_loader = get_adv_images_loders(adv_images_dir, batch_size=1)
    for ind, (adv_img, label) in enumerate(adv_loader):
        adv_img = adv_img.to(device)
        clean_image = Image.open(clean_image_paths[ind])
        clean_image = transform(clean_image).unsqueeze(0).to(device)

        sum += metric(adv_img, clean_image)
    avg_score = sum / len(glob.glob(os.path.join(dir, "*")))
    logger.info(f"{metric_name} score for adversarial images is {avg_score}")

    





