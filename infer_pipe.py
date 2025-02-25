import os
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm

from ours_net import OurModel
from custom_pipeline import OursPipeline
from unet_2d_condition_copy import UNet2DConditionModel_copy
from diffusers import UniPCMultistepScheduler

def generate(pipe, seed, prompt_i, prompt_m, negative_prompt=None, img_size=384):
    generator = torch.manual_seed(seed)
    res = pipe(
        prompt_i=prompt_i,
        prompt_m=prompt_m,
        # negative_prompt=negative_prompt,
        height=img_size,
        width=img_size,
        generator=generator,
        num_inference_steps=30,    # 30
    ).images
    image, mask = res[0], res[1]
    return image, mask

base_model_path = "pre-training_SD_1.5_path"
controlnet_path = "checkpoints_save_path/checkpoint-4000/controlnet/"

unet_copy = UNet2DConditionModel_copy.from_pretrained(base_model_path, subfolder="unet", torch_dtype=torch.float16)
controlnet = OurModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

pipe = OursPipeline.from_pretrained(
    base_model_path, 
    unet_copy=unet_copy,
    controlnet=controlnet, 
    torch_dtype=torch.float16
)

father_path = "./to_save_path"
os.makedirs(os.path.join(father_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(father_path, 'masks'), exist_ok=True)


# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler_i = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler_m = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()
pipe.safety_checker = lambda images, clip_input: (images, None)    # disable safety checker

# db
prompt_json = "./data/prompt.json"

with open(prompt_json, "r") as f:
    lines = f.readlines()
    # lines = lines[:len(lines)//2]
    seed = 0
    for line in tqdm(lines):
        line = line.strip()
        data = json.loads(line)
        key = data["image"].split(".")[0]
        value = data["text"]

        prompt_i, prompt_m = value.split(" &&& ")
        negative_prompt = None

        # generate image
        # for i in range(5):
        for i in range(1):
            image, mask = generate(pipe, seed, prompt_i, prompt_m, negative_prompt, img_size=320)
            seed += 1

            # check if mask is all black
            mask_color = len(np.unique(mask[0]))
            if mask_color == 1:
                image, mask = generate(pipe, seed, prompt_i, prompt_m, negative_prompt, img_size=320)
                seed += 1

            image[0].save(os.path.join(father_path, 'images', f"{key}_{i}.png"))
            mask[0].save(os.path.join(father_path, 'masks', f"{key}_{i}.png"))