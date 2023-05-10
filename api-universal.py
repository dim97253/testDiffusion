
from asyncio.windows_events import NULL
from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import sys # to access the system
import base64
from PIL import Image
import io
from io import BytesIO


app = Flask(__name__)

repo_anything = "I:/Models/models--andite--anything-v4.0/snapshots/d0966af39e715d6e97a7664eafcd19930e8efb84" #Local anything-4.0
repo_openjourney = "I:/Models/models--prompthero--openjourney-v2/snapshots/47257274a40e93dab7fbc0cd2cfd5f5704cfeb60" #Local openjourney-v2
repo_diffusion15 = "I:/Models/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819" #Local diffusion
repo_inpainting = "I:/Models/stable-diffusion-2-inpainting" #Local inpainting

pipe = StableDiffusionPipeline.from_pretrained(repo_openjourney, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()    

#text2img = StableDiffusionPipeline(**pipe.components)
#img2img = StableDiffusionImg2ImgPipeline(**pipe.components)
#inpaint = StableDiffusionInpaintPipeline(**pipe.components)

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

    if (src_img64 != 'undefined') and (src_mask64!= 'undefined'): #imgInpaint
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