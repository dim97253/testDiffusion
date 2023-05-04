from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import sys # to access the system
import base64
from PIL import Image
import io
from io import BytesIO


app = Flask(__name__)
repo_anything = "andite/anything-v4.0"
repo_openjourney = "prompthero/openjourney-v2"
repo_diffusion15 = "runwayml/stable-diffusion-v1-5"

def safety_checker(images, clip_input):
    return images, False


@app.route('/', methods=['POST'])
def img():
    data = request.get_json()
    prompt = data.get('prompt', '')
    negative_prompt = data.get('neg_prompt', '')
    model_name = data.get('model', '')
    steps = int(data.get('steps', ''))
    src_img64 = str(data.get('srcimage', ''))
    if src_img64 != 'undefined':
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_openjourney,local_files_only=True)
        if model_name == "openjourney":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_openjourney,local_files_only=True)
        if model_name == "anything4":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_anything,local_files_only=True)
        if model_name == "diffusion15":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_diffusion15,local_files_only=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to('cuda:0')
        pipe.enable_attention_slicing()    
        pipe.safety_checker = safety_checker
        src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
        #images = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images #Img2Img anything
        images = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images #Img2Img openjourney
    else:
        pipe = StableDiffusionPipeline.from_pretrained(repo_openjourney,local_files_only=True)
        if model_name == "openjourney":
            pipe = StableDiffusionPipeline.from_pretrained(repo_openjourney,local_files_only=True)
        if model_name == "anything4":
            pipe = StableDiffusionPipeline.from_pretrained(repo_anything,local_files_only=True)
        if model_name == "diffusion15":
            pipe = StableDiffusionPipeline.from_pretrained(repo_diffusion15,local_files_only=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to('cuda:0')
        pipe.enable_attention_slicing()    
        pipe.safety_checker = safety_checker
        images = pipe(prompt = prompt, negative_prompt  = negative_prompt, num_inference_steps=steps, width =512, height=512).images

        

    
    
    image=images[0]

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    #response = make_response(img_bytes)
    #response.headers.set('Content-Type', 'image/pmg')
    #response.headers.set(
    #   'Content-Disposition', 'attachment', filename='output.png')
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
