import sys
import os 
import random
import argparse

sys.path.append(".")
from dataset_caption import imagenet_label
from venom_demo import preprocess, ddim_sample_adv_momentum_with_text_target
import torch

from diffusers import DDIMScheduler, StableDiffusionPipeline

from torch.backends import cudnn
from tqdm import tqdm
import numpy as np 
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
    
parser = argparse.ArgumentParser()

parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=2.0)
parser.add_argument('--timesteps', type=int, default=100)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=0.7)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--save-dir', type=str, default='./images/')
parser.add_argument('--label', type=int, nargs="+", default= [107, 99, 113, 130, 207, 309,340])
parser.add_argument('--test', type=bool, default=True)
parser.add_argument('--target_text', type=str, default="panda")
args = parser.parse_args()


num2text = imagenet_label.refined_Label

text2num = dict((v,k) for k,v in num2text.items())


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
        "stabilityai/stable-diffusion-2-base",
        torch_dtype=torch.float32,
    )
    model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
    model.to(device)
    #model.enable_xformers_memory_efficient_attention()
    model.enable_vae_slicing()
    
    
    vic_model = models.resnet50(pretrained=True)

    vic_model.to(device)
    vic_model.eval()

    fire_truck = "A bright red fire truck parked on a city street, ready for action with its lights flashing."

    cheeseburger =   "A juicy cheeseburger with lettuce, tomato, and pickles, on a toasted sesame seed bun, served on a diner table." 
    
    espresso_machine = "A sleek espresso maker on a kitchen counter, brewing a fresh cup of coffee with steam rising."

    labels = [fire_truck, cheeseburger, espresso_machine]

    text_labels = ["fire engine", "cheeseburger", "espresso maker"]
    labels_in_number = []
    for text_label in text_labels:
        labels_in_number.append(text2num[text_label])

    n_samples_per_class = args.batch_size

    timesteps = args.timesteps
    scale = args.scale   # for unconditional guidance

    test_iamge_path = os.path.join(args.save_dir, 'demo_output')
    os.makedirs(test_iamge_path, exist_ok=True)

    with torch.no_grad():
        for ind, text_label in enumerate(labels):

            print(f"rendering {n_samples_per_class} examples of class {labels_in_number[ind]}: {text_label} in {timesteps} steps and using s={scale:.2f}.")

            # ddim_sample_adv(latents, context, model, vic_model, batch_size, timesteps: int, \
            #  guidance_scale:float=2.5, eta:float=0.0, label=None, iterations: int=5, s:float=1.0, a:float=0.5):

            label = torch.tensor([labels_in_number[ind]]).long().unsqueeze(0)
            context =  generate_context(model, text_label)
            latents_batch = generate_latents(model, n_samples_per_class, args.image_size)

            for j in tqdm(
                range(n_samples_per_class),
                total=n_samples_per_class,
                desc="Samples",
                leave=False,
            ):
                latents = latents_batch[j].unsqueeze(0)
                
                samples = ddim_sample_adv_momentum_with_text_target(
                    latents=latents, 
                    context=context, 
                    target_text=args.target_text,
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
                test_image = preprocess(images).to(device)
                pred = vic_model(test_image)
                pred = torch.argmax(pred,1)[0].cpu().item()
                pred_text = num2text[pred]
                print(f"{text_labels[ind]} is classified as {pred_text}")

                for i in range(images.shape[0]):
                    text_label = text_labels[ind]
                    image_name = f"{text_label}_{j} classified as {pred_text}"
                    to_pil_image( images[i].cpu()).save(os.path.join(test_iamge_path, f"{image_name}.png"))
     


            
if __name__ == '__main__':
    main()




