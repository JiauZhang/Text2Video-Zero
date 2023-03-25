from PIL import Image
import torch

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
