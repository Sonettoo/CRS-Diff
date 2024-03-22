import sys
import random
from datasets import load_dataset
from PIL import Image
from utils.share import *
import utils.config as config
from annotator.util import resize_image, HWC3
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
from annotator.prompt_m import PromptDetector


import cv2
import einops
import gradio as gr
import numpy as np
import os
from tqdm import tqdm

import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
from PIL import Image

# image_resolution=224
num_samples=3
ddim_steps=50
H=512
W=512
strength=1
n_prompt="Blurry, distorted, overexposed"
a_prompt="best quality, extremely detailed"

prompt="text prompt"
model = create_model('./configs/crs.yaml').cpu()
model.load_state_dict(load_state_dict('path/to/ckpt', location='cuda'))

model = model.cuda()
ddim_sampler = DDIMSampler(model)

#you can load iamge condition directly,here is the sample code using a single conditional image
condition_path="path/image/condition/"
anchor_image=np.array(Image.open(condition_path)).astype(np.uint8)

anchor_image = np.stack((anchor_image, anchor_image, anchor_image), axis=-1)
anchor_image = cv2.resize(anchor_image, (W, H))
condition_detected_map=anchor_image

# detected_maps_list=[mlsd_detected_map, hed_detected_map,sketch_detected_map,road_detected_map,midas_detected_map,seg_detected_map]
with torch.no_grad():
    zero_map=np.zeros((512, 512, 3)).astype(np.uint8)
    detected_maps_list = [zero_map, 
                        zero_map,
                        zero_map,
                        condition_detected_map,
                        zero_map,
                        zero_map                          
                        ]
    
content_emb = np.zeros((768))
metadata_emb=np.zeros((7))
global_maps=np.concatenate((content_emb,content_emb),axis=0)
detected_maps = np.concatenate(detected_maps_list, axis=2)

local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
global_control = torch.from_numpy(global_maps.copy()).float().cuda().clone()
global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)
metadata_control=torch.from_numpy(metadata_emb.copy()).float().cuda().clone().squeeze()

if config.save_memory:
    model.low_vram_shift(is_diffusing=False)
uc_local_control = local_control
uc_global_control = torch.zeros_like(global_control)
cond = {"local_control": [local_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], 'global_control': [global_control]}
un_cond = {"local_control": [uc_local_control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)], 'global_control': [uc_global_control]}
shape = (4, H // 8, W // 8)

if config.save_memory:
    model.low_vram_shift(is_diffusing=True)

model.control_scales = [strength] * 13
## add x_T
samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                shape, metadata_control,conditioning=cond, verbose=False, eta=0.2,
                                                unconditional_guidance_scale=7.5,
                                                unconditional_conditioning=un_cond, global_strength=1)
if config.save_memory:
    model.low_vram_shift(is_diffusing=False)
x_samples = model.decode_first_stage(samples)
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
results = [x_samples[i] for i in range(num_samples)]

for i, image_path in enumerate(results):
    image = Image.fromarray(image_path)
    image.save(f'path/to/save/f{i}.png')