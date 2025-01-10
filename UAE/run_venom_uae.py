import sys
import os 
import random
import argparse

sys.path.append(".")
from dataset_caption import imagenet_label
from venom_uae import  ddim_sample_adv_momentum, ddim_sample_adv_momentum_untargeted
import torch


from diffusers import DDIMScheduler, StableDiffusionPipeline
from torch.backends import cudnn
from utils import model_selection

import numpy as np 
from PIL import Image

import glob
from natsort import ns, natsorted
    
parser = argparse.ArgumentParser()

parser.add_argument('--res', default=224, type=int, help='Input image resized resolution')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--images_root', default="./images", type=str,
                    help='The clean images root directory')
parser.add_argument('--label_path', default="./labels.txt", type=str,
                    help='The clean images labels.txt')
parser.add_argument('--timesteps', type=int, default=100)
parser.add_argument('--start_step', default=80, type=int, help='Which DDIM step to start the attack')
parser.add_argument('--ddim_eta', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=0.7)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--save_dir', type=str, default='./out/')
parser.add_argument('--model_name', default="resnet", type=str,
                    help='The surrogate model from which the adversarial examples are crafted')
args = parser.parse_args()

# label:  default= [107, 99, 113, 130, 207, 309,340]





def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    save_dir = os.path.join(args.save_dir, "outimages")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base",
        torch_dtype=torch.float32,
    )
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.to(device)
    #model.enable_xformers_memory_efficient_attention()
    model.enable_vae_slicing()
    
    
    
    vic_model = model_selection(args.model_name)
    vic_model.eval()
    vic_model.requires_grad_(False)
    vic_model.to(device)
    


    timesteps = args.timesteps
    scale = args.scale   # for unconditional guidance
    res = args.res
    start_step = args.start_step
    iterations = args.K

    images_root = args.images_root  # The clean images' root directory.
    label_path = args.label_path  # The clean images' labels.txt.
    with open(label_path, "r") as f:
        label = []
        for i in f.readlines():
            label.append(int(i.rstrip()) - 1)  # The label number of the imagenet-compatible dataset starts from 1.
        label = np.array(label)


    all_images = glob.glob(os.path.join(images_root, "*"))
    all_images = natsorted(all_images, alg=ns.PATH)

    adv_images = []
    images = []
    clean_all_acc = 0
    adv_all_acc = 0

    for ind, image_path in enumerate(all_images):
        tmp_image = Image.open(image_path).convert('RGB')
        adv_image, clean_acc, adv_acc  = ddim_sample_adv_momentum(model, label[ind:ind + 1],
                                            vic_model,
                                            save_path=os.path.join(save_dir, str(ind).rjust(4, '0')),
                                            timesteps=timesteps, 
                                            guidance_scale=scale,
                                            image = tmp_image,
                                            res=res, 
                                            start_step=start_step,
                                            eta=0.0,
                                            iterations=iterations,
                                            s=args.s,
                                            a=args.a,
                                            beta=args.beta,
                                            )

        adv_image = adv_image.astype(np.float32) / 255.0
        adv_images.append(adv_image[None].transpose(0, 3, 1, 2))

        tmp_image = tmp_image.resize((res, res), resample=Image.LANCZOS)
        tmp_image = np.array(tmp_image).astype(np.float32) / 255.0
        tmp_image = tmp_image[None].transpose(0, 3, 1, 2)
        images.append(tmp_image)

        clean_all_acc += clean_acc
        adv_all_acc += adv_acc

    print("Clean acc: {}%".format(clean_all_acc / len(all_images) * 100))
    print("Adv acc: {}%".format(adv_all_acc / len(all_images) * 100))

    adv_images = np.concatenate(adv_images)
    np.savez(os.path.join(args.save_dir, 'adv_images.npz'), adv_images, label)
            
if __name__ == '__main__':
    main()




