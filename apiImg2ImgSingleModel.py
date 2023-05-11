from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import sys # to access the system
import base64
from PIL import Image
import io
from io import BytesIO

app = Flask(__name__)

def safety_checker(images, clip_input):
    return images, False

repo = "andite/anything-v4.0"
#repo_openjourney = "prompthero/openjourney-v4"
#repo_diffusion15 = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo) 
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda:0')
pipe.safety_checker = safety_checker

@app.route('/', methods=['POST'])
def img():
    data = request.get_json()
    prompt = data.get('prompt', '')
    negative_prompt = data.get('neg_prompt', '')
    model_name = data.get('model', '')
    src_img64 = str(data.get('srcimage', ''))
    strength = float(data.get('strength', ''))
    guidance_scale = float(data.get('strength', ''))
    src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
    images = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images #Img2Img openjourney

    image=images[0]

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
