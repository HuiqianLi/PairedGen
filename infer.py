import os
import cv2
import torch
import json
import numpy as np
from ours_net import OurModel
from custom_pipeline import OursPipeline
from unet_2d_condition_copy import UNet2DConditionModel_copy
from diffusers import UniPCMultistepScheduler

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
os.makedirs("./out_img", exist_ok=True)


# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler_i = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler_m = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()
pipe.safety_checker = lambda images, clip_input: (images, None)    # disable safety checker

prompt = "a sks endoscope image. &&& Polyp."
prompt_i, prompt_m = prompt.split(" &&& ")
negative_prompt = None

# generate image
size = 320
for i in range(5):
    generator = torch.manual_seed(i)
    res = pipe(
        prompt_i=prompt_i,
        prompt_m=prompt_m,
        # negative_prompt=negative_prompt,
        height=size,
        width=size,
        generator=generator,
        num_inference_steps=30,
    ).images
    image, mask = res[0], res[1]
    mask_gray = np.asarray(mask[0])

    # save image and mask
    # image[0].save(f"./out_img/output_{i}.png")
    # cv2.imwrite(f"./out_img/mask_{i}.png", mask_gray)

    # concat image and mask
    image = np.asarray(image[0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    # img_mask = np.hstack((image, mask)) # horizontal
    img_mask = np.vstack((image, mask)) # vertical
    cv2.imwrite(f"./out_img/img_mask_{i}.png", img_mask)