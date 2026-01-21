import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import streamlit as st

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_txt2img_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    pipe.to(_device())
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None
    return pipe

@st.cache_resource
def get_img2img_pipeline():
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    pipe.to(_device())
    if torch.cuda.is_available():
        pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None
    return pipe

def _seed_generator(seed: int):
    device = _device()
    if seed is None or seed < 0:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))

def generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    num_images: int,
    ref_image: Image.Image = None,
    strength: float = 0.65,
):
    generator = _seed_generator(seed)

    if ref_image is None:
        pipe = get_txt2img_pipeline()
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            num_images_per_prompt=num_images,
            generator=generator,
        )
        return out.images

    # img2img
    pipe = get_img2img_pipeline()
    ref_image = ref_image.convert("RGB")
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=ref_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=generator,
    )
    return out.images
