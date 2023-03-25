import torch, diffusers, argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from utils import image_grid, latent_to_image
from zero import DDIMBackward

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--prompt', type=str, default='Chinese Panda')
parser.add_argument('--num_inference_steps', type=int, default=50)

args = parser.parse_args()
device = args.device
model_id = args.model_id
prompt = args.prompt
steps = args.num_inference_steps

DDIM_backward = DDIMBackward()
ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
ddpm_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
SD = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=ddim_scheduler, torch_dtype=torch.float32,
    cache_dir='.',
).to(device)
generator = torch.Generator(device).manual_seed(19491001)

images = SD(
    prompt, generator=generator, num_inference_steps=steps,
    callback=DDIM_backward,
).images
image = image_grid(images, rows=1, cols=1)
image.save('panda.png')

images = []
for i in range(0, len(DDIM_backward.all_latents), 2):
    images += latent_to_image(DDIM_backward.all_latents[i][1], SD)

image = image_grid(images, rows=5, cols=5)
image.save('pandas.png')
