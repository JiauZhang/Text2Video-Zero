import torch, diffusers, argparse
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from utils import image_grid

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
parser.add_argument('--prompt', type=str, default='a high quality realistic photo of a panda walking alone down the street')
parser.add_argument('--num_inference_steps', type=int, default=50)

args = parser.parse_args()
device = args.device
model_id = args.model_id
prompt = args.prompt
steps = args.num_inference_steps

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
generator = torch.Generator(device).manual_seed(0)
images = pipe(prompt, generator=generator, num_inference_steps=steps).images
image_grid(images, rows=1, cols=1)
