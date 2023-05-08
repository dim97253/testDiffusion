
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
repo_diffusion_inpainting = 'runwayml/stable-diffusion-inpainting'

pipe_diffusion15 = StableDiffusionPipeline.from_pretrained(repo_diffusion15, torch_dtype=torch.float16)
pipe_diffusion15.scheduler = DPMSolverMultistepScheduler.from_config(pipe_diffusion15.scheduler.config)
pipe_diffusion15 = pipe_diffusion15.to('cuda:0')

pipe_openjourney = StableDiffusionPipeline.from_pretrained(repo_openjourney, torch_dtype=torch.float16)
pipe_openjourney.scheduler = DPMSolverMultistepScheduler.from_config(pipe_openjourney.scheduler.config)
pipe_openjourney = pipe_openjourney.to('cuda:1')

pipe_anything = StableDiffusionPipeline.from_pretrained(repo_anything, torch_dtype=torch.float16)
pipe_anything.scheduler = DPMSolverMultistepScheduler.from_config(pipe_anything.scheduler.config)
pipe_anything = pipe_anything.to('cuda:2')

pipe_inpainting = StableDiffusionPipeline.from_pretrained(repo_diffusion_inpainting, torch_dtype=torch.float16, revision="fp16")
pipe_inpainting = pipe_inpainting.to('cuda:3')

#text2img = StableDiffusionPipeline(**pipe.components)
#img2img = StableDiffusionImg2ImgPipeline(**pipe.components)

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

    pipe
    if model_name == 'openjourney2':
        print('Selected model: openjourney2')
        pipe = pipe_openjourney
    elif model_name == 'diffusion15':
        print('Selected model: diffusion15')
        pipe = pipe_diffusion15
    elif model_name == 'anything4':
        print('Selected model: anything4')
        pipe = pipe_anything
    else:
        print('Running default model: openjourney2')
        pipe = pipe_openjourney  

    images   
    if (src_img64 != 'undefined') and (src_mask64!= 'undefined'): #imgInpaint
        imgInpainting = StableDiffusionInpaintPipeline(**pipe_inpainting.components)
        src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))        
        src_mask = Image.open(io.BytesIO(base64.decodebytes(bytes(src_mask64, "utf-8"))))
        images = imgInpainting(prompt = prompt,negative_prompt = negative_prompt, image=src_img, mask_image=src_mask).images #inpainting  
    elif src_img64 != 'undefined': #img2img
        img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
        src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
        images = img2img(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images
    else: #text2img
        text2img = StableDiffusionPipeline(**pipe.components)
        images = text2img(prompt = prompt, negative_prompt  = negative_prompt, num_inference_steps=steps, width =512, height=512).images

    image=images[0]

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
