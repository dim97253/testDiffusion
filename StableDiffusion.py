from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, HeunDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, StableDiffusionUpscalePipeline, StableDiffusionLatentUpscalePipeline
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import sys # to access the system
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#repo_id = "stabilityai/stable-diffusion-2-1" 
#repo_id = "runwayml/stable-diffusion-v1-5"
#repo_id = "Linaqruf/anything-v3.0"
#repo_id = "prompthero/openjourney-v2"  
#repo_id = "stb/anything-of-f222"  
#repo_id = "C:/Users/Heron/.cache/huggingface/diffusers/models--prompthero--openjourney-v2/snapshots/47257274a40e93dab7fbc0cd2cfd5f5704cfeb60" #Local openjourney-v2
#repo_id = "C:/Users/Heron/.cache/huggingface/diffusers/models--Linaqruf--anything-v3.0/snapshots/31d62f73e8cb52d7818e34bf7c3c87424de99ad9" #Local anything-3.0
#repo_id = "C:/Users/Heron/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1/snapshots/36a01dc742066de2e8c91e7cf0b8f6b53ef53da1" #Local stable-diffusion-2-1
#repo_id = "andite/anything-v4.0"
#repo_id = "C:/Users/Heron/.cache/huggingface/diffusers/models--andite--anything-v4.0/snapshots/d0966af39e715d6e97a7664eafcd19930e8efb84" #Local anything-4.0
#repo_id = "D:/repos/StableDiffusion/StableDiffusion/models/stable-diffusion-2-inpainting" #Local inpainting
repo_id = "D:/repos/StableDiffusion/StableDiffusion/models/models--stabilityai--stable-diffusion-x4-upscaler/snapshots/19b610c68ca7572defb6e09e64d1063f32b4db83" #Local upscaler
#repo_id = "D:/repos/StableDiffusion/StableDiffusion/models/models--dreamlike-art--dreamlike-photoreal-2.0/snapshots/d9e27ac81cfa72def39d74ca673219c349f0a0d5" #Local dreamlike photoreal 2

#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
#pipe = StableDiffusionPipeline.from_pretrained(repo_id,local_files_only=True, vae=vae) #Local
#pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id,local_files_only=True, vae=vae) #img2img vae
#pipe = StableDiffusionInpaintPipeline.from_pretrained(repo_id,local_files_only=True, torch_dtype=torch.float16) #inpainting
pipe = StableDiffusionUpscalePipeline.from_pretrained(repo_id, torch_dtype=torch.float16) #upscaling


#pipe = StableDiffusionPipeline.from_pretrained(repo_id,torch_dtype=torch.float16)
#pipe = StableDiffusionPipeline.from_pretrained(repo_id)
#pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id) #Img2Img

#pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
prompt = "Photo of dog on park bench, high resolution"
negative_prompt = "blur, artifacts"

pipe.enable_attention_slicing()
#pipe.enable_vae_slicing()
pipe.enable_sequential_cpu_offload()

def safety_checker(images, clip_input):
    return images, False
pipe.safety_checker = safety_checker


init_image = Image.open("Z:/src.png").convert("RGB")
#init_image= init_image.resize((512, 512))
#init_image.thumbnail((512, 512))
#mask = Image.open("Z:/src_mask.png").convert("RGB")
#mask= mask.resize((64, 64))
#mask.thumbnail((512, 512))

for i in range(22):
    #images = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=40, width=512, height=512).images
    #images = pipe(prompt = prompt, negative_prompt = negative_prompt, num_inference_steps=30).images

    #strength=i*4*0.01+0.06
    #strength=0.55
    #images = pipe(prompt = prompt, image=init_image, strength=strength, guidance_scale=4.0, negative_prompt = negative_prompt).images #Img2Img
    #images = pipe(prompt = prompt, image=init_image, mask_image=mask).images #inpainting
    images = pipe(prompt = prompt, image=init_image).images #upscaling
    images[0].save("Z:/"+str(i)+".png")
    i+=1

    #plt.imshow(images[0], interpolation='nearest')
    #plt.show()
    #plt.waitforbuttonpress(timeout=-1)

