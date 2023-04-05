import torch, diffusers, argparse
from diffusers import (
    ControlNetModel, StableDiffusionControlNetPipeline,
    DDIMScheduler, DDPMScheduler,
)
from controlnet_aux import (
    OpenposeDetector, MLSDdetector, HEDdetector, CannyDetector,
    MidasDetector,
)
from utils import load_gif, image_grid

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

model_id = 'runwayml/stable-diffusion-v1-5'
open_pose = OpenposeDetector.from_pretrained(
    'lllyasviel/ControlNet', cache_dir='.',
)

gif = 'images/Michael_Jackson.gif'
frames = load_gif(gif)
images = []
for frame in frames:
    pose = open_pose(frame.permute(1, 2, 0))
    images.append(pose)
image = image_grid(images, rows=2, cols=len(images)//2)
image.save('pose.png')

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL_IDS['pose'], torch_dtype=torch.float32,
    cache_dir='.',
)
ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, scheduler=ddim_scheduler,
    torch_dtype=torch.float32, cache_dir='.',
)
