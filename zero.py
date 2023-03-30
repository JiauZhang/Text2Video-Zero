import torch, cv2
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from typing import Any, Callable, Dict, List, Optional, Union

class DDIMBackward(StableDiffusionPipeline):
    def __init__(
        self, vae, text_encoder, tokenizer, unet, scheduler,
        safety_checker, feature_extractor,
        requires_safety_checker: bool = True,
        t_start=941, delta_t=60, processor=None,
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
        self.processor = processor

    def record(self, t, timestep, latent):
        self.all_latents.append([timestep, latent])

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        t_start=None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if callback and self.processor:
            self.unet.set_attn_processor(self.processor)
            self.processor.record = True
        elif self.processor:
            self.processor.record = False

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if t_start and t >= t_start:
                    progress_bar.update()
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if self.processor:
                    self.processor.timestep = t.item()

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

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

class CrossFrameAttnProcessor:
    def __init__(self, device):
        # id --> key, value, device: self.__kv will be saved on cpu or gpu
        self.__kv = {}
        self.device = device
        self.record = False
        self.timestep = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        self_attn = encoder_hidden_states is None
        if self_attn:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        if not self_attn or self.record:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if self_attn:
            attn_id = id(attn)
            if self.record:
                kv = [key.to(self.device), value.to(self.device)]
                if attn_id in self.__kv:
                    self.__kv[attn_id][self.timestep] = kv
                else:
                    self.__kv[attn_id] = {self.timestep: kv}
            else:
                key, value = self.__kv[attn_id][self.timestep]
                key, value = key.to(query.device), value.to(query.device)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
