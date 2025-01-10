import sys
import os 
import random
import argparse

sys.path.append(".")
from dataset_caption import imagenet_label
from venom_nae import ddim_sample_adv_momentum_no_text_target, ddim_sample_adv_momentum_with_text_target, preprocess
import torch
from tqdm import tqdm

from diffusers import DDIMScheduler, StableDiffusionPipeline
from torch.backends import cudnn
from utils import model_selection

import numpy as np 

from torchvision.transforms.functional import to_pil_image
    
parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--n_samples_per_class', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--timesteps', type=int, default=100)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=0.7)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--save_dir', type=str, default='./images/')
parser.add_argument('--label', type=int, nargs="+", default= [107, 99, 113, 130, 207, 309])
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--target_label', type=str, default='goldfish')
args = parser.parse_args()

# label:  default= [107, 99, 113, 130, 207, 309]


def generate_latents(model, n_samples_per_class,img_size):
    latents_shape = (
                n_samples_per_class,
                model.unet.config.in_channels,
                img_size // model.vae_scale_factor,
                img_size // model.vae_scale_factor,
            )
    latents = torch.randn(
        latents_shape, device=model.device)
    latents = latents * model.scheduler.init_noise_sigma

    return latents

def generate_context(model, text_label):
    max_length = 77
    uncond_input = model.tokenizer(
        [""], padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        [text_label],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)
    return context






def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StableDiffusionPipeline.from_pretrained(
        "bguisard/stable-diffusion-nano-2-1",
        torch_dtype=torch.float32,
    )
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.to(device)
    #model.enable_xformers_memory_efficient_attention()
    model.enable_vae_slicing()
    
    
    
    vic_model = model_selection(args.model_name)
    vic_model.to(device)
    vic_model.requires_grad_(False)
    vic_model.eval()


    if args.test:
        #labels = np.arange(200)
        labels = args.label
    else:
        labels = np.arange(1000)


    n_samples_per_class = args.n_samples_per_class


    target_label = args.target_label
    timesteps = args.timesteps
    scale = args.scale   # for unconditional guidance


    save_path = os.path.join(args.save_dir, str(args.beta))
    test_iamge_path = os.path.join(save_path, 'adv_images_momentum)')
    os.makedirs(test_iamge_path, exist_ok=True)

    with torch.no_grad():
        for class_label in labels:
            text_label = imagenet_label.refined_Label[class_label]
            print(f"rendering {n_samples_per_class} examples of class {class_label}: {text_label} in {timesteps} steps and using s={scale:.2f}.")

            #latents, context, target_text, model, vic_model, timesteps: int, guidance_scale:float=2.5, eta:float=0.0, label=None, iterations: int=5, s:float=1.0, a:float=0.5, beta=0.5):

            
            
            label = torch.tensor([class_label]).long().unsqueeze(0)
            context =  generate_context(model, text_label)
            latents_batch = generate_latents(model, n_samples_per_class, args.image_size)
            
            for j in tqdm(
                range(n_samples_per_class),
                total=n_samples_per_class,
                desc="Samples",
                leave=False,
            ): 
                
                latents = latents_batch[j].unsqueeze(0)
                samples = ddim_sample_adv_momentum_no_text_target(
                            latents=latents, 
                            context=context, 
                            model=model, 
                            vic_model=vic_model, 
                            timesteps = timesteps, 
                            guidance_scale=scale,
                            eta = args.ddim_eta,
                            label=label.to(model.device),
                            iterations=args.K,
                            s=args.s,
                            a=args.a,
                            beta=args.beta
                            )
            
                samples = samples / model.vae.config.scaling_factor
                images = model.vae.decode(samples, return_dict=False)[0]
                images = model.image_processor.postprocess(images, output_type="pt")
                for i in range(images.shape[0]):
                    text_label = imagenet_label.refined_Label[class_label]
                    image_name = f"{text_label}_{j}"
                    to_pil_image( images[i].cpu()).save(os.path.join(test_iamge_path, f"{image_name}.png"))

    # save the test image
     


            
if __name__ == '__main__':
    main()




