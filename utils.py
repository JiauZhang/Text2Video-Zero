from PIL import Image
import torch
import numpy as np

def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

@torch.no_grad()
def latent_to_image(latents, SD):
    image = SD.decode_latents(latents)
    image = SD.numpy_to_pil(image)
    return image

def load_gif(path):
    gif = Image.open(path)
    frames = []
    for i in range(1, gif.n_frames):
        gif.seek(i)
        frame = np.array(gif)
        frames.append(torch.tensor(frame).unsqueeze(0))
        frames[-1] = frames[-1].permute(0, 3, 1, 2)
    return torch.cat(frames, dim=0)
