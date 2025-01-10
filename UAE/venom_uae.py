import numpy as np
from tqdm import tqdm
from PIL import Image
from dataset_caption import imagenet_label
from torch import optim


import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

import gc 


def get_target_label(logits, label, device): # seond-like label for attack (for the targeted attack)
    
    rates, indices = logits.sort(1, descending=True) 
    rates, indices = rates.squeeze(0), indices.squeeze(0)  
    
    tar_label = torch.zeros_like(label).to(device)
    

    if label == indices[0]:  # classification is correct
        tar_label = indices[1]
    else:
        tar_label = indices[0]
    
    return tar_label

process_latent = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop((224,224)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)

def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale, extra_step_kwargs):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def ddim_reverse_sample(image, prompt, model, num_inference_steps: int = 20, guidance_scale: float = 2.5,
                        res=512):
    """
            ==========================================
            ============ DDIM Inversion ==============
            ==========================================
    """
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    #  Not inverse the last step, as the alpha_bar_next will be set to 0 which is not aligned to its real value (~0.003)
    #  and this will lead to a bad result.
    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        next_timestep = t + model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
        alpha_bar_next = model.scheduler.alphas_cumprod[next_timestep] \
            if next_timestep <= model.scheduler.config.num_train_timesteps else torch.tensor(0.0)

        "leverage reversed_x0"
        reverse_x0 = (1 / torch.sqrt(model.scheduler.alphas_cumprod[t]) * (
                latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t])))

        latents = reverse_x0 * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * noise_pred

        all_latents.append(latents)

    #  all_latents[N] -> N: DDIM steps  (X_{T-1} ~ X_0)
    return latents, all_latents


#image, label, model, vic_model, batch_size, timesteps: int, guidance_scale:float=2.5, eta:float=0.0, \
# label=None, iterations: int=5, s:float=1.0, a:float=0.5
        


def ddim_sample_adv_momentum(
        model,
        label,
        vic_model,
        save_path,
        timesteps:int=200,
        guidance_scale:float=2.5,
        image=None,
        res=224,
        start_step=150,
        eta:float=0.0,
        iterations=5,
        s:float=1.0, 
        a:float=0.5,
        beta:float=0.5
        ):


    model.scheduler.set_timesteps(timesteps)
    timesteps_tensor = model.scheduler.timesteps.to(model.device)
    extra_step_kwargs = model.prepare_extra_step_kwargs(None, eta)
    total_steps = len(timesteps_tensor)


    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    height = width = res


    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = vic_model(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    prompt = [imagenet_label.refined_Label[label.item()]]

    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    timesteps,
                                                    0, res=height)

    inversion_latents = inversion_latents[::-1]
    batch_size = len(prompt)
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]


    latent, latents = init_latent(latent, model, height, width, batch_size)

    context_no_tune = torch.cat([uncond_embeddings, text_embeddings])

    pri_latents = latent.detach().requires_grad_(True)

    for k in tqdm(range(iterations)):   
        latents = pri_latents.detach().requires_grad_(True)

        v_inner = torch.zeros_like(latents)
        beta = beta
        adv_guidance = True
        for ind, t in tqdm(
            enumerate(model.scheduler.timesteps[1 + start_step - 1:]),
            desc= "Adv sampling",
            total = total_steps - start_step,
            leave=False,
        ):
            index = total_steps -start_step - ind - 1

            if k > 2:
                adv_guidance = True

            latents = diffusion_step(model, latents, context_no_tune, t, guidance_scale, extra_step_kwargs)

            if adv_guidance == False:
                with torch.no_grad():
                    latents_n = latents.detach()
                    latents_n = latents_n / model.vae.config.scaling_factor
                    image = model.vae.decode(latents_n, return_dict=False)[0]
                    image = model.image_processor.postprocess(image, output_type='pt')
                    image = process_latent(image).to(model.device)
                    logits = vic_model(image)
                    log_probs = F.log_softmax(logits, dim=-1)
                    pred = torch.argmax(log_probs, dim=1)
                    success_num = batch_size - (pred == label).sum().item()
                    if success_num < 1:
                        adv_guidance = True
                    del image
            

            if index > 0 and adv_guidance:
                with torch.enable_grad():
                    latents_n = latents.detach().requires_grad_(True)
                    latents_n = latents_n / model.vae.config.scaling_factor
                    image = model.vae.decode(latents_n, return_dict=False)[0]
                    image = model.image_processor.postprocess(image, output_type='pt')
                    image = process_latent(image).to(model.device)
                    logits = vic_model(image)
                    log_probs = F.log_softmax(logits, dim=-1)
                    target_label = get_target_label(logits, label, model.device)
 
                    pred = torch.argmax(log_probs, dim=1)
                    success_num = batch_size - (pred == label).sum().item()
                    if success_num > 0:
                        adv_guidance = False

                    selected = log_probs[range(len(logits)), target_label]
                    gradient = torch.autograd.grad(selected.sum(), latents_n)[0]
                    if (ind==0):
                        v_inner = gradient
                    v_inner = beta * v_inner + (1 - beta) * gradient
                latents = (latents + s * v_inner.float())
                del image
        

        x_samples = latents / model.vae.config.scaling_factor
        x_samples = model.vae.decode(x_samples, return_dict=False)[0]
        x_samples = model.image_processor.postprocess(x_samples, output_type='pt')

        
        image = process_latent(x_samples).to(model.device)
        logits = vic_model(image)
        log_probs = F.log_softmax(logits, dim=-1)
        pred = torch.argmax(log_probs, dim=1)  # [B]
        success_num = (pred == label).sum().item()
        print(pred)
        print(f"Success {batch_size-success_num} / {batch_size}")
        if batch_size-success_num > 0 : # early exit
            break
        with torch.enable_grad():
            latents_n = latents.detach().requires_grad_(True)
            latents_n = latents_n / model.vae.config.scaling_factor
            image = model.vae.decode(latents_n, return_dict=False)[0]
            image = model.image_processor.postprocess(image, output_type='pt')
            image = process_latent(image).to(model.device)
            logits = vic_model(image)
            log_probs = F.log_softmax(logits, dim=-1)
            target_label = get_target_label(logits, label, model.device)
            selected = log_probs[range(len(logits)), target_label]
            gradient = torch.autograd.grad(selected.sum(), latents_n)[0]
        pri_latents = (pri_latents + a * gradient.float())
        gc.collect()


    image = latent2image(model.vae, latents.detach())
    perturbed = image.astype(np.float32) / 255 
    image = (perturbed * 255).astype(np.uint8)
    if save_path is not None:
        save_path = save_path + prompt[0] + "_adv.png"
        to_pil_image(image[0]).save(save_path)

    return image[0], pred_accuracy_clean, (1-success_num)








def ddim_sample_adv_momentum_untargeted(
        model,
        label,
        vic_model,
        save_path,
        timesteps:int=200,
        guidance_scale:float=2.5,
        image=None,
        res=224,
        start_step=150,
        eta:float=0.0,
        iterations=5,
        s:float=1.0, 
        a:float=0.5,
        beta:float=0.5
        ):


    model.scheduler.set_timesteps(timesteps)
    timesteps_tensor = model.scheduler.timesteps.to(model.device)
    extra_step_kwargs = model.prepare_extra_step_kwargs(None, eta)
    total_steps = len(timesteps_tensor)


    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    height = width = res


    test_image = image.resize((height, height), resample=Image.LANCZOS)
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]
    test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = vic_model(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    prompt = [imagenet_label.refined_Label[label.item()]]

    latent, inversion_latents = ddim_reverse_sample(image, prompt, model,
                                                    timesteps,
                                                    0, res=height)

    inversion_latents = inversion_latents[::-1]
    batch_size = len(prompt)
    latent = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]


    latent, latents = init_latent(latent, model, height, width, batch_size)

    context_no_tune = torch.cat([uncond_embeddings, text_embeddings])

    pri_latents = latent.detach().requires_grad_(True)

    for k in tqdm(range(iterations)):   
        latents = pri_latents.detach().requires_grad_(True)

        v_inner = torch.zeros_like(latents)
        beta = beta
        adv_guidance = True
        for ind, t in tqdm(
            enumerate(model.scheduler.timesteps[1 + start_step - 1:]),
            desc= "Adv sampling",
            total = total_steps - start_step,
            leave=False,
        ):
            index = total_steps -start_step - ind - 1

            if k > 2:
                adv_guidance = True

            latents = diffusion_step(model, latents, context_no_tune, t, guidance_scale, extra_step_kwargs)

            if adv_guidance == False:
                with torch.no_grad():
                    latents_n = latents.detach()
                    latents_n = latents_n / model.vae.config.scaling_factor
                    image = model.vae.decode(latents_n, return_dict=False)[0]
                    image = model.image_processor.postprocess(image, output_type='pt')
                    image = process_latent(image).to(model.device)
                    logits = vic_model(image)
                    log_probs = F.log_softmax(logits, dim=-1)
                    pred = torch.argmax(log_probs, dim=1)
                    success_num = batch_size - (pred == label).sum().item()
                    if success_num < 1:
                        adv_guidance = True
                    del image
            

            if index > 0 and adv_guidance:
                with torch.enable_grad():
                    latents_n = latents.detach().requires_grad_(True)
                    latents_n = latents_n / model.vae.config.scaling_factor
                    image = model.vae.decode(latents_n, return_dict=False)[0]
                    image = model.image_processor.postprocess(image, output_type='pt')
                    image = process_latent(image).to(model.device)
                    logits = vic_model(image)
                    log_probs = F.log_softmax(logits, dim=-1)
                    #target_label = get_target_label(logits, label, model.device)
 
                    pred = torch.argmax(log_probs, dim=1)
                    success_num = batch_size - (pred == label).sum().item()
                    if success_num > 0:
                        adv_guidance = False

                    #target_label = get_target_label(logits, label, model.device)
                    selected = log_probs[range(len(logits)), label]
                    gradient = torch.autograd.grad(selected.sum(), latents_n)[0]
                    if (ind==0):
                        v_inner = gradient
                    v_inner = beta * v_inner + (1 - beta) * gradient
                latents = latents - s * v_inner.float()
                del image
        

        x_samples = latents / model.vae.config.scaling_factor
        x_samples = model.vae.decode(x_samples, return_dict=False)[0]
        x_samples = model.image_processor.postprocess(x_samples, output_type='pt')
        image = process_latent(x_samples).to(model.device)
        logits = vic_model(image)
        log_probs = F.log_softmax(logits, dim=-1)
        pred = torch.argmax(log_probs, dim=1)  # [B]
        success_num = (pred == label).sum().item()
        print(pred)
        print(f"Success {batch_size-success_num} / {batch_size}")
        if batch_size-success_num > 0 : # early exit
            break
        with torch.enable_grad():
            latents_n = latents.detach().requires_grad_(True)
            latents_n = latents_n / model.vae.config.scaling_factor
            image = model.vae.decode(latents_n, return_dict=False)[0]
            image = model.image_processor.postprocess(image, output_type='pt')
            image = process_latent(image).to(model.device)
            logits = vic_model(image)
            log_probs = F.log_softmax(logits, dim=-1)
            #target_label = get_target_label(logits, label, model.device)
            selected = log_probs[range(len(logits)), label]
            gradient = torch.autograd.grad(selected.sum(), latents_n)[0]

        pri_latents = (pri_latents + a * gradient.float())
        gc.collect()


    image = latent2image(model.vae, latents.detach())
    perturbed = image.astype(np.float32) / 255 
    image = (perturbed * 255).astype(np.uint8)
    if save_path is not None:
        save_path = save_path + prompt[0] + "_adv.png"
        to_pil_image(image[0]).save(save_path)

    return image[0], pred_accuracy_clean, (1-success_num)







