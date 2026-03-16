import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import xformers

"""
This generator is a training-free model, and it is sensitive to hyperparameters. Please make sure the pipeline can generate
images before using it for evaluation. Please refer this for details: https://github.com/YangLing0818/RPG-DiffusionMaster.
"""

def _memory_efficient_attention_xformers(module, query, key, value):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
    hidden_states = module.batch_to_head_dim(hidden_states)
    return hidden_states



class GeneratorPipe:
    def __init__(self, model_id, dtype=torch.float32, device="cuda"):

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder='text_encoder').eval().to(device,
                                                                                                        dtype=dtype)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae').eval().to(device, dtype=dtype)
        self.vae.enable_slicing()

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder='unet').eval().to(device, dtype=dtype)
        self.unet.set_use_memory_efficient_attention_xformers(True)

        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.dtype = dtype
        self.device = device
        self.hook_forwards(self.unet)

    def encode_prompts(self, prompts):
        '''
        text_encoder output the hidden states
        prompts are based on list
        '''
        with torch.no_grad():
            tokens = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding=True, truncation=True,
                                    return_tensors='pt').input_ids.to(self.device)
            embs = self.text_encoder(tokens, output_hidden_states=True).last_hidden_state.to(self.device,
                                                                                             dtype=self.dtype)
        return embs

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def __call__(self, prompts, negative_prompt, batch_size=1, height: int = 512, width: int = 512, guidance_scale: float = 7.0,
            num_inference_steps: int = 50, base_ratio=0.3, end_steps: float = 1, direction=None):
        '''
        prompts: base prompt + regional prompt
        direction: defines how regional latents are merged; pass {'left','right'} for horizontal
                    or {'top','bottom'} for vertical composition.
        '''

        self.base_ratio = base_ratio
        direction = direction or {"left", "right"}
        direction = set(direction)
        if direction & {"top", "bottom"}:
            self.combine_mode = "vertical"
        elif direction & {"left", "right"}:
            self.combine_mode = "horizontal"
        else:
            raise ValueError(f"Unsupported direction spec {direction}. "
                             f"Expected entries like 'left'/'right' or 'top'/'bottom'.")

        all_prompts = []
        for prompt in prompts:
            all_prompts.extend([prompt] * batch_size)
        all_prompts.extend([negative_prompt] * batch_size)

        text_embs = self.encode_prompts(all_prompts)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = torch.randn(batch_size, 4, height // 8, width // 8).to(self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma
        self.height = height // 8
        self.width = width // 8
        self.pixels = self.height * self.width

        progress_bar = tqdm(range(num_inference_steps), desc="Total Steps", leave=False)
        self.double = True

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # attention_double version ending condition
            if i > num_inference_steps * end_steps and self.double:
                # print(i)
                cond, _, _, negative = text_embs.chunk(4)  # cond, left, right, negative
                text_embs = torch.cat([cond, negative])
                self.double = False
            with torch.no_grad():
                noise_pred = self.unet(sample=latent_model_input, timestep=t, encoder_hidden_states=text_embs).sample
            noise_pred_text, noise_pred_negative = noise_pred.chunk(2)
            noise_pred = noise_pred_negative + guidance_scale * (noise_pred_text - noise_pred_negative)

            # Get denoised latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            progress_bar.update(1)

        images = self.decode_latents(latents)
        return images

    def hook_forward(self, module):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            context = encoder_hidden_states
            batch_size, sequence_length, _ = hidden_states.shape

            query = module.to_q(hidden_states)
            if self.double:
                query_cond, query_uncond = query.chunk(2)
                query = torch.cat([query_cond, query_cond, query_cond, query_uncond])  # 4*8960*320

            context = context if context is not None else hidden_states
            key = module.to_k(context)
            value = module.to_v(context)

            dim = query.shape[-1]

            query = module.head_to_batch_dim(query)
            key = module.head_to_batch_dim(key)
            value = module.head_to_batch_dim(value)

            # if module._use_memory_efficient_attention_xformers:
            hidden_states = _memory_efficient_attention_xformers(module, query, key, value)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
            # else:
            #     if module._slice_size is None or query.shape[0] // module._slice_size == 1:
            #         hidden_states = module._attention(query, key, value)
            #     else:
            #         hidden_states = module._sliced_attention(query, key, value, sequence_length, dim)

            if self.double:
                rate = int((self.pixels // query.shape[1]) ** 0.5)  # down sample rate

                height = self.height // rate
                width = self.width // rate

                cond, region_a, region_b, uncond = hidden_states.chunk(4)

                # reshape to the image shape
                region_a = region_a.reshape(region_a.shape[0], height, width, region_a.shape[2])
                region_b = region_b.reshape(region_b.shape[0], height, width, region_b.shape[2])

                if self.combine_mode == "horizontal":
                    combined = torch.cat([region_a[:, :, :width // 2, :],
                                          region_b[:, :, width // 2:, :]], dim=2)
                else:
                    combined = torch.cat([region_a[:, :height // 2, :, :],
                                          region_b[:, height // 2:, :, :]], dim=1)
                combined = combined.reshape(cond.shape[0], -1, cond.shape[2])  # 1*8960*320

                cond = combined * (1 - self.base_ratio) + cond * self.base_ratio

                hidden_states = torch.cat([cond, uncond])

            hidden_states = module.to_out[0](hidden_states)
            hidden_states = module.to_out[1](hidden_states)

            return hidden_states
        return forward

    def hook_forwards(self, root_module: torch.nn.Module):
        for name, module in root_module.named_modules():
            if "attn2" in name and module.__class__.__name__ == "Attention":
                module.forward = self.hook_forward(module)
