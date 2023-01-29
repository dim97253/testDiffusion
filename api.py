from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import sys # to access the system
import base64
from io import BytesIO


app = Flask(__name__)
repo_id = "prompthero/openjourney-v2"  
pipe = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()


@app.route('/', methods=['GET'])
def img():
    args = request.args
    prompt = args['prompt']
    negative_prompt = args['neg_prompt']
    steps=int(args['steps'])

    images = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=steps, width =512, height= 512).images
    image=images[0]
    image.save("output.png")

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    #img_b64 = base64.b64encode(img_bytes)
    response = make_response(img_bytes)
    response.headers.set('Content-Type', 'image/pmg')
    response.headers.set(
        'Content-Disposition', 'attachment', filename='output.png')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
