import torch, diffusers, argparse
from diffusers import StableDiffusionPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--prompt', type=str, default='a high quality realistic photo of a panda walking alone down the street')

args = parser.parse_args()
device = args.device
model_id = args.model_id
prompt = args.prompt

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
generator = torch.Generator(device).manual_seed(0)
image = pipe(prompt, generator=generator).images[0]
