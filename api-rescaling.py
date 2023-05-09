from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import sys # to access the system
import base64
from PIL import Image
import io
from io import BytesIO


app = Flask(__name__)
repo_id = "stabilityai/stable-diffusion-x4-upscaler" #Local inpainting

@app.route('/', methods=['POST'])
def img():
    data = request.get_json()
    prompt = data.get('prompt', '')
    negative_prompt = data.get('neg_prompt', '')
    model_name = data.get('model', '')
    src_img64 = str(data.get('srcimage', ''))
    pipe = StableDiffusionUpscalePipeline.from_pretrained(repo_id,local_files_only=True, torch_dtype=torch.float16) #upscaling
          
    pipe = pipe.to('cuda:0')
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()

    src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8")))).convert("RGB")
    
    images = pipe(prompt = prompt, image=src_img).images #upscaling        
    image=images[0]

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)