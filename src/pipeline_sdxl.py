import torch
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

@st.cache_resource
def get_txt2img():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype(),
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    ).to(_device())
    return pipe

@st.cache_resource
def get_img2img():
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=_dtype(),
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    ).to(_device())
    return pipe

def _seed_gen(seed: int):
    if seed is None or seed < 0:
        return None
    return torch.Generator(device=_device()).manual_seed(int(seed))

def txt2img(prompt, negative_prompt, width, height, steps, guidance, seed, num_images):
    pipe = get_txt2img()
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=_seed_gen(seed)
    )
    return out.images

def img2img(prompt, negative_prompt, init_image: Image.Image, strength, steps, guidance, seed, num_images):
    pipe = get_img2img()
    init_image = init_image.convert("RGB")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=_seed_gen(seed)
    )
    return out.images
