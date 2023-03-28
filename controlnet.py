import torch, diffusers, argparse
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, DDPMScheduler,
)

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

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_IDS['pose'], torch_dtype=torch.float16,
    cache_dir='.',
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', controlnet=controlnet,
    torch_dtype=torch.float16, cache_dir='.',
)
