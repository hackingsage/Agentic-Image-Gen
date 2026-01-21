import torch
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

CONTROLNETS = {
    "Canny": "diffusers/controlnet-canny-sdxl-1.0",
    "Depth": "diffusers/controlnet-depth-sdxl-1.0",
}

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32

def _seed_gen(seed: int):
    if seed is None or seed < 0:
        return None
    return torch.Generator(device=_device()).manual_seed(int(seed))

def _to_canny(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, 100, 200)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)

@st.cache_resource
def get_controlnet_pipe(kind: str):
    cn_id = CONTROLNETS[kind]
    controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=_dtype())
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_ID,
        controlnet=controlnet,
        torch_dtype=_dtype(),
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    ).to(_device())
    return pipe

def controlnet_generate(kind, prompt, negative_prompt, ref_image: Image.Image,
                        width, height, steps, guidance, seed, num_images, control_strength=0.8):
    pipe = get_controlnet_pipe(kind)

    if kind == "Canny":
        control_img = _to_canny(ref_image)
    else:
        control_img = ref_image.convert("RGB")

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_img,
        controlnet_conditioning_scale=float(control_strength),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=_seed_gen(seed)
    )
    return out.images, control_img
