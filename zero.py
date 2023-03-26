import torch, cv2
import numpy as np
from diffusers.utils import torch_utils
from diffusers import StableDiffusionPipeline

class DDIMBackward(StableDiffusionPipeline):
    def __init__(
        self, vae, text_encoder, tokenizer, unet, scheduler,
        safety_checker, feature_extractor,
        requires_safety_checker: bool = True,
        t_start=941, delta_t=60,
    ):
        super().__init__(
            vae, text_encoder, tokenizer, unet, scheduler,
            safety_checker, feature_extractor, requires_safety_checker,
        )
        self.t_start = t_start
        self.delta_t = delta_t
        self.t_star_dot = t_start - delta_t
        self.latents = []
        self.all_latents = []

    def record(self, t, timestep, latent):
        if timestep == self.t_start:
            self.latents = [latent]
        elif timestep == self.t_star_dot:
            self.latents.append(latent)
        self.all_latents.append([timestep, latent])

@torch.no_grad()
def DDPM_forward(x_t_dot, t_start, delta_t, ddpm_scheduler):
    # just simple implementation, this should have an analytical expression
    # TODO: implementation analytical form
    for delta in range(1, delta_t):
        noise = torch.randn_like(x_t_dot)
        beta = ddpm_scheduler.betas[t_start+delta]
        std_ = beta ** 0.5
        mu_ = ((1 - beta) ** 0.5) * x_t_dot
        x_t_dot = mu_ + std_ * noise
    return x_t_dot

class MotionDynamics():
    def __init__(self, ddpm_scheduler, num_inference_steps, t_start, device):
        self.ddpm_scheduler = ddpm_scheduler
        self.steps = num_inference_steps
        self.t_start = t_start
        self.device = device
        self.ddpm_scheduler.set_timesteps(self.steps, device=device)

    @torch.no_grad()
    def __call__(self, x_t_dot, delta_t=60, scale=1.8, m=8, direction=(1, 1)):
        x_2_m = []
        np_x_t_dot = x_t_dot.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze(0)
        for k in range(2, m+1):
            dx = scale * (k - 1) * direction[0]
            dy = scale * (k - 1) * direction[1]
            W_k = np.array([[1, 0, dx], [0, 1, dy]])
            x_t_k_ = cv2.warpAffine(np_x_t_dot, W_k, (64, 64), borderMode=cv2.BORDER_REFLECT)
            x_t_k_ = torch.tensor(x_t_k_, device=x_t_dot.device).unsqueeze(0)
            x_t_k_ = x_t_k_.permute(0, 3, 1, 2).detach()
            x_t_k = DDPM_forward(
                x_t_k_, self.t_start, delta_t, self.ddpm_scheduler)
            x_2_m.append(x_t_k)

        return x_2_m
