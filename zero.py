import torch, cv2
import numpy as np
from diffusers.utils import torch_utils

@torch.no_grad()
def DDIM_backward(x_t, delta_t, SD):
    pass

@torch.no_grad()
def DDPM_forward(x_t, delta_t, SD):
    pass

@torch.no_grad()
def motion_dynamics(SD, delta_t=60, scale=1e-1, m=8, direction=(1, 1)):
    x_t = torch.randn((1, SD.in_channels, SD.sample_size, SD.sample_size))
    x_t_dot = DDIM_backward(x_t, delta_t, SD)
    x_1_m = [x_t]

    for k in range(2, m+1):
        dx = scale * (k - 1) * direction[0]
        dy = scale * (k - 1) * direction[1]
        W_k = np.array([[1, 0, dx], [0, 1, dy]])
        x_t_k_ = cv2.warpAffine(x_t_dot, W_k, (512, 512))
        x_t_k = DDPM_forward(x_t_k_, delta_t, SD)
        x_1_m.append(x_t_k)

    return x_1_m
