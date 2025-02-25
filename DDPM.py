import torch
from diffusers import DDPMScheduler

class DDPMScheduler_copy(DDPMScheduler):

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        self.alphas_cumprod = self.alphas_cumprod.to(device=x_t.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=x_t.dtype)
        timesteps = timesteps.to(x_t.device)

        # from alphas_cumprod extract sqrt_alpha_prod and sqrt_one_minus_alpha_prod
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_t.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # predict original samples
        predicted_original_samples = (x_t - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod
        return predicted_original_samples