from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import gc
import base64
from PIL import Image
import io
from io import BytesIO

app = Flask(__name__)

def flush():
  gc.collect()
  torch.cuda.empty_cache()

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
        image = pipe(prompt = prompt,negative_prompt = negative_prompt, image=src_img, mask_image=src_mask).images[0] #inpainting  
    elif src_img64 != 'undefined': #img2img
        if model_name == 'openjourney2':
            model_id = "prompthero/openjourney-v4"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
            image = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images[0]
        elif model_name == 'diffusion15':
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
            image = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images[0]
        elif model_name == 'anything4':
            model_id = "xyn-ai/anything-v4.0"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
            image = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images[0]
        else:
            model_id = "prompthero/openjourney-v4"
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            src_img = Image.open(io.BytesIO(base64.decodebytes(bytes(src_img64, "utf-8"))))
            image = pipe(prompt = prompt, image=src_img, strength=0.60, guidance_scale=18.0, negative_prompt = negative_prompt).images[0]
    else: #text2img
        if model_name == 'openjourney2':
            model_id = "prompthero/openjourney-v4"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            image = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=steps, width=1136, height=640).images[0]
        elif model_name == 'diffusion15':
            model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            image = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=steps, width=1136, height=640).images[0]
        elif model_name == 'anything4':
            model_id = "xyn-ai/anything-v4.0"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            image = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=steps, width=1136, height=640).images[0]
        else:
            model_id = "prompthero/openjourney-v4"
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            image = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=steps, width=1136, height=640).images[0]    

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    flush()
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
