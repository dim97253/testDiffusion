from flask import Flask, send_file, request, render_template, make_response
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import sys # to access the system
import base64
from io import BytesIO


#devices
devices=(torch.cuda.device(0),torch.cuda.device(1),torch.cuda.device(2),torch.cuda.device(3))

#model openjourney
repo_id = "prompthero/openjourney-v2"  
pipe_openjourney2 = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
pipe_openjourney2.scheduler = DPMSolverMultistepScheduler.from_config(pipe_openjourney2.scheduler.config)
#model stable diffusion 1.5
repo_id = "runwayml/stable-diffusion-v1-5"  
pipe_diffusion15 = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
pipe_diffusion15.scheduler = DPMSolverMultistepScheduler.from_config(pipe_diffusion15.scheduler.config)
#model stable diffusion 2.1
repo_id = "stabilityai/stable-diffusion-2-1"  
pipe_diffusion21 = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
pipe_diffusion21.scheduler = DPMSolverMultistepScheduler.from_config(pipe_diffusion21.scheduler.config)
#model anything 4
repo_id = "andite/anything-v4.0"  
pipe_anything4 = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
pipe_anything4.scheduler = DPMSolverMultistepScheduler.from_config(pipe_anything4.scheduler.config)

#Set devices
pipe_openjourney2 = pipe_openjourney2.to(devices[0])
pipe_diffusion15 = pipe_diffusion15.to(devices[1])
pipe_diffusion21 = pipe_diffusion21.to(devices[2])
pipe_anything4 = pipe_anything4.to(devices[3])

app = Flask(__name__)
@app.route('/', methods=['GET'])
def img():
    #args
    args = request.args
    prompt = args['prompt']
    negative_prompt = args['neg_prompt']
    steps=int(args['steps'])
    model_name=args['model']
    
    if model_name == 'openjourney2':
        print('Selected model: openjourney2')
        pipe = pipe_openjourney2
    elif model_name == 'diffusion15':
        print('Selected model: diffusion15')
        pipe = pipe_diffusion15
    elif model_name == 'diffusion21':
        print('Selected model: diffusion21')
        pipe = pipe_diffusion21
    elif model_name == 'anything4':
        print('Selected model: anything4')
        pipe = pipe_anything4
    else:
        print('Running default model: openjourney2')
        pipe = pipe_openjourney2        
    
    images = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=steps, width =512, height= 512).images
    image=images[0]

    buffer = BytesIO()
    image.save(buffer, format='png')
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes)
    return img_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
