import torch, diffusers, argparse
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, DDPMScheduler,
)
from controlnet_aux import (
    OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector,
    MidasDetector,
)
from utils import load_gif

CONTROLNET_MODEL_IDS = {
    'canny': 'lllyasviel/sd-controlnet-canny',
    'hough': 'lllyasviel/sd-controlnet-mlsd',
    'hed': 'lllyasviel/sd-controlnet-hed',
    'scribble': 'lllyasviel/sd-controlnet-scribble',
    'pose': 'lllyasviel/sd-controlnet-openpose',
    'seg': 'lllyasviel/sd-controlnet-seg',
    'depth': 'lllyasviel/sd-controlnet-depth',
    'normal': 'lllyasviel/sd-controlnet-normal',
}

gif = 'images/Michael_Jackson.gif'
frames = load_gif(gif)
model_id = 'runwayml/stable-diffusion-v1-5'
open_pose = OpenposeDetector.from_pretrained(
    'lllyasviel/ControlNet', cache_dir='.',
)
controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_IDS['pose'], torch_dtype=torch.float32,
    cache_dir='.',
)
ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, scheduler=ddim_scheduler,
    torch_dtype=torch.float32, cache_dir='.',
)
