import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from dataset_caption import imagenet_label


num2text = imagenet_label.refined_Label

text2num = dict((v,k) for k,v in num2text.items())




def get_target_label(logits, label): # seond-like label for attack
    
    rates, indices = logits.sort(1, descending=True) 
    #rates, indices = rates.squeeze(0), indices.squeeze(0)  
    
    tar_label = torch.zeros_like(label).to(label.device)
    
    for i in range(label.shape[0]):
        if label[i] == indices[i][0]:  # classify is correct
            tar_label[i] = indices[i][1]
        else:
            tar_label[i] = indices[i][0]
    
    return tar_label


def get_target_label_from_text(text_label, label):
    assert text_label in text2num, f"the target category {text_label} is not in the Imagenet labels"
    target_label = torch.tensor(text2num[text_label]).long()
    target_label = target_label.expand_as(label).to(label.device)
    return target_label

preprocess = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def diffusion_step(model, latents, context, t, guidance_scale, extra_step_kwargs):
    latent_input = torch.cat([latents] * 2)
    latent_input = model.scheduler.scale_model_input(latent_input, t)
    noise_pred = model.unet(latent_input, t, encoder_hidden_states=context, return_dict=False)[0]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

    latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
    return latents

def latents2image(latents, model):

    latents = latents.detach() / model.vae.config.scaling_factor
    image = model.vae.decode(latents, return_dict=False)[0]
    image = model.image_processor.postprocess(image, output_type='pt')
    image = preprocess(image).to(model.device)

    return image







def ddim_sample_adv_momentum_with_text_target(latents, context, target_text, model, vic_model, timesteps: int, guidance_scale:float=2.5, eta:float=0.0, label=None, iterations: int=5, s:float=1.0, a:float=0.5, beta=0.5):


    model.scheduler.set_timesteps(timesteps)
    timesteps_tensor = model.scheduler.timesteps.to(model.device)
    extra_step_kwargs = model.prepare_extra_step_kwargs(None, eta)
    total_steps = len(timesteps_tensor)
 
    pri_latents = latents.detach().requires_grad_(True)

    for k in range(iterations):   
        latents = pri_latents.detach().requires_grad_(True)


        v_inner = torch.zeros_like(latents)
        beta = beta
        adversarial_guidance = True
        for ind, t in tqdm(
            enumerate(timesteps_tensor),
            desc= "Adv sampling",
            total = total_steps,
            leave=False,
        ):
            index = total_steps - ind -1

            ## If the generation fails more than 3 time, forcefully apply strong adversarial guidance
            if k >2:
                adversarial_guidance = True


            latents = diffusion_step(model, latents, context, t, guidance_scale, extra_step_kwargs)

            # The adversarial examples generated with early stop might be purified after a few more
            # diffusion steps. Check if the adversary is still valid after early stop. If it is not 
            # valid, apply the adversarial guidance again
            if adversarial_guidance == False:
                with torch.no_grad():
                    image = latents2image(latents, model)
                    logits = vic_model(image)
                    log_probs = F.log_softmax(logits, dim=-1)
                    pred = torch.argmax(log_probs, dim=1)
                    success_num = (pred == target_label).sum().item()
                    if success_num < 1:
                        adversarial_guidance = True                

            if (index > total_steps *0 and index <= total_steps * 0.2 and adversarial_guidance):
                with torch.enable_grad():
                    latents_n = latents.detach().requires_grad_(True)
                    latents_n = latents_n / model.vae.config.scaling_factor
                    image = model.vae.decode(latents_n, return_dict=False)[0]
                    image = model.image_processor.postprocess(image, output_type='pt')
                    image = preprocess(image).to(model.device)
                    logits = vic_model(image)
                    log_probs = F.log_softmax(logits, dim=-1)
                    #target_label = get_target_label(logits, label, model.device)
                    target_label = get_target_label_from_text(target_text, label)
                    pred = torch.argmax(log_probs, dim=1)
                    success_num = (pred == target_label).sum().item()
                    if success_num > 0:
                        adversarial_guidance = False
                    selected = log_probs[range(len(logits)), target_label]
                    gradient = torch.autograd.grad(selected.sum(), latents_n)[0]
                    if (index == total_steps * 0.2):
                        v_inner = gradient
                    v_inner = beta * v_inner + (1 - beta) * gradient
                latents = latents + s * v_inner.float()

        

        x_samples = latents / model.vae.config.scaling_factor
        x_samples = model.vae.decode(x_samples, return_dict=False)[0]
        x_samples = model.image_processor.postprocess(x_samples, output_type='pt')

        with torch.enable_grad():
            latents_n = latents.detach().requires_grad_(True)
            latents_n = latents_n / model.vae.config.scaling_factor
            image = model.vae.decode(latents_n, return_dict=False)[0]
            image = model.image_processor.postprocess(image, output_type='pt')
            image = preprocess(image).to(model.device)
            logits = vic_model(image)
            log_probs = F.log_softmax(logits, dim=-1)
            target_label = get_target_label_from_text(target_text,label)
            #target_label = get_target_label(logits, label, model.device)
            selected = log_probs[range(len(logits)), target_label]
            gradient = torch.autograd.grad(selected.sum(), latents_n)[0]
        image = preprocess(x_samples).to(model.device)
        logits = vic_model(image)
        log_probs = F.log_softmax(logits, dim=-1)
        pred = torch.argmax(log_probs, dim=1)  # [B]
        success_num = (pred == target_label).sum().item()
        print(pred)
        print(f"Success {success_num} / {label.shape[0]}")
        if success_num > 0 : # early exit
            break
        #gradient = torch.clamp(gradient, min=-0.3, max=0.3)
        pri_latents = pri_latents + a * gradient.float()
            
    return latents

