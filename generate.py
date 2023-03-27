import torch, diffusers, argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
from utils import image_grid, latent_to_image
from zero import DDIMBackward, MotionDynamics, CrossFrameAttnProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--prompt', type=str, default='Chinese Panda')
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument('--t_start', type=int, default=941)
parser.add_argument('--delta_t', type=int, default=60)

args = parser.parse_args()
device = args.device
model_id = args.model_id
prompt = args.prompt
steps = args.num_inference_steps
t_start, delta_t = args.t_start, args.delta_t

ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
ddpm_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
SD = DDIMBackward.from_pretrained(
    model_id, scheduler=ddim_scheduler, torch_dtype=torch.float32,
    cache_dir='.', t_start=t_start, delta_t=delta_t,
    processor=CrossFrameAttnProcessor(),
).to(device)
generator = torch.Generator(device).manual_seed(19491001)

images = SD(
    prompt, generator=generator, num_inference_steps=steps,
    callback=SD.record,
).images
image = image_grid(images, rows=1, cols=1)
image.save('panda.png')

images = []
for i in range(0, len(SD.all_latents), 2):
    images += latent_to_image(SD.all_latents[i][1], SD)

image = image_grid(images, rows=5, cols=5)
image.save('pandas.png')

start, end = 2, 7
t_start = SD.all_latents[start][0]
t_end = SD.all_latents[end][0]
latent_start = SD.all_latents[start][1]
latent_end = SD.all_latents[end][1]
motion_dynamics = MotionDynamics(ddpm_scheduler, 1000, t_end, device)
x_1_m = [latent_start] + motion_dynamics(latent_end, delta_t=(end-start)*20)

images = []
for x in x_1_m:
    images += SD(
        prompt, generator=generator, num_inference_steps=steps,
        latents=x, t_start=t_start,
    ).images
image = image_grid(images, rows=1, cols=len(images))
image.save('frames.png')
