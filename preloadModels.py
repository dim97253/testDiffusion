from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import sys # to access the system
import base64
from PIL import Image
import io
from io import BytesIO

models = []
models.append('prompthero/openjourney-v4')
models.append('runwayml/stable-diffusion-v1-5')
models.append('andite/anything-v4.0')
models.append('stabilityai/stable-diffusion-2-inpainting')

for model in models :
    print ('loading '+str(model))
    StableDiffusionPipeline.from_pretrained(model)

print ('all models downloaded')


