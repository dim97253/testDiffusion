
from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import sys # to access the system
import base64
from PIL import Image
import io
from io import BytesIO

app = Flask(__name__)

repo_anything = 'andite/anything-v4.0'
repo_openjourney = 'prompthero/openjourney-v4'
repo_diffusion15 = 'runwayml/stable-diffusion-v1-5'
repo_inpainting = 'stabilityai/stable-diffusion-2-inpainting'

@app.route('/', methods=['POST'])
def img():
    data = request.get_json()
    prompt = data.get('prompt', '')
    negative_prompt = data.get('neg_prompt', '')
    model_name = data.get('model', '')
    if data.get('steps', '')!= '':
       steps = int(data.get('steps', ''))
    else:
        steps=10
    src_img64 = str(data.get('srcimage', ''))    
    src_mask64 = str(data.get('srcmask', ''))

    if (src_img64 != 'undefined') and (len(src_mask64)>0): #imgInpaint
        pipe = StableDiffusionInpaintPipeline.from_pretrained(repo_inpainting, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))        
        src_mask = Image.open(io.BytesIO(base64.decodebytes(bytes(src_mask64, "utf-8"))))
        images = pipe(prompt = prompt,negative_prompt = negative_prompt, image=src_img, mask_image=src_mask).images #inpainting  
    elif src_img64 != 'undefined': #img2img
        if model_name == 'openjourney2':
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_openjourney, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        elif model_name == 'diffusion15':
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_diffusion15, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        elif model_name == 'anything4':
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_anything, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_openjourney, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            
        src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
        images = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images
    else: #text2img
        if model_name == 'openjourney2':
            pipe = StableDiffusionPipeline.from_pretrained(repo_openjourney, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        elif model_name == 'diffusion15':
            pipe = StableDiffusionPipeline.from_pretrained(repo_diffusion15, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        elif model_name == 'anything4':
            pipe = StableDiffusionPipeline.from_pretrained(repo_anything, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        else:
            pipe = StableDiffusionPipeline.from_pretrained(repo_openjourney, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
        images = pipe(prompt = prompt, negative_prompt  = negative_prompt, num_inference_steps=steps, width =512, height=512).images    
    
    image=images[0]

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
