import torch
import streamlit as st
from PIL import Image
from diffusers import AutoPipelineForInpainting

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def _seed_gen(seed: int):
    if seed is None or seed < 0:
        return None
    return torch.Generator(device=_device()).manual_seed(int(seed))

@st.cache_resource
def get_inpaint_pipe():
    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype(),
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    ).to(_device())
    return pipe

def inpaint(prompt, negative_prompt, image: Image.Image, mask: Image.Image,
            steps, guidance, seed, strength=0.75):
    pipe = get_inpaint_pipe()
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image.convert("RGB"),
        mask_image=mask.convert("RGB"),
        strength=float(strength),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=_seed_gen(seed)
    )
    return out.images
