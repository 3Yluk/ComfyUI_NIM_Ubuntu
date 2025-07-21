import base64
from io import BytesIO
import os
import sys
import tempfile
import time
import numpy as np
import requests
import torch
from PIL import Image
from typing import Dict, Tuple

from .nimubuntu import ModelType, NIMManager_ubuntu


manager = NIMManager_ubuntu()

class NIMFLUXNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "is_nim_started": ("STRING", {"forceInput": True}),
                "width": (["768", "832", "896", "960", "1024", "1088", "1152", "1216", "1280", "1344"], {  # noqa: E501
                    "default": "1024",
                    "tooltip": "Width of the image to generate, in pixels."
                }),
                "height": (["768", "832", "896", "960", "1024", "1088", "1152", "1216", "1280", "1344"], {
                    "default": "1024",
                    "tooltip": "Height of the image to generate, in pixels."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "beautiful scenery nature glass bottle landscape, purple galaxy bottle",
                    "tooltip": "The attributes you want to include in the image."
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 9.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4294967295,
                    "display": "number",
                    "tooltip": "The seed which governs generation. Use 0 for a random seed"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Number of diffusion steps to run"
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The image used for depth and canny mode."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NVIDIA/NIM"


    def generate(self, width, height, prompt, cfg_scale, seed, steps, is_nim_started, image=None):
        if is_nim_started[0] == "":
            raise Exception("Please make sure use 'Load NIM' before this node to start NIM.")
        model_name = ModelType[is_nim_started[0]]
        port = manager.get_port(model_name)
        invoke_url = f"http://localhost:{port}/v1/infer"

        #mode = model_name.value.split("_")[-1].lower().replace("dev", "base")
        mode = NIMManager_ubuntu._get_variant(self, model_name)

        if model_name.value.split('_')[-1].lower() == 'schnell':
            cfg_scale = 0
            if steps > 4:
                raise Exception ("Flux Schnell step value must be between 1-4 steps")
    
        payload = {
            "width": int(width),
            "height": int(height),
            "text_prompts": [
                {
                    "text": prompt,
                },
            ],
            "mode": mode,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "steps": steps
        }
        
        print(payload)
        
        if mode != "base":
            if image is None:
                raise Exception("Please use load image node to select image input for FLUX depth and canny mode.")
        
            def _comfy_image_to_bytes(img: torch.tensor, depth: int = 8):
                max_val = 2**depth - 1
                img = torch.clip(img * max_val, 0, max_val).to(dtype=torch.uint8)
                pil_img = Image.fromarray(img.squeeze(0).cpu().numpy())

                img_byte_arr = BytesIO()
                pil_img.save(img_byte_arr, format="PNG")
                return img_byte_arr.getvalue(), ".png"
            
            image_bytes, _ = _comfy_image_to_bytes(img=image)
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            image = f"data:image/png;base64,{base64_string}"
            payload.update({"image": image})

        try:
            response = requests.post(invoke_url, json=payload)
            print(response)
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Unable to connect to NIM API.")
            
        data = response.json()
        response.raise_for_status()
        img_base64 = data["artifacts"][0]["base64"]
        img_bytes = base64.b64decode(img_base64)

        print("Result: " + data["artifacts"][0]["finishReason"])

        image = Image.open(BytesIO(img_bytes))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]  

        return (image,)


class LoadNIMNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": ([e.value for e in ModelType], {
                    "default": ModelType.FLUX_DEV.value,
                    "tooltip": "The type of NIM model to use"
                }),
                "hf_token": ("STRING", {
                    "default": os.environ.get("HF_TOKEN", ""),
                    "tooltip": "Input your Huggingface API Token"
                })
            }
        }
    
    # RETURN_TYPES = ()
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("is_nim_started",)
    OUTPUT_NODE = True
    FUNCTION = "process_nim"
    CATEGORY = "NVIDIA/NIM"

   
    def process_nim(self, model_type: str, hf_token: str ):
        return (model_type,)

# Update the mappings
NODE_CLASS_MAPPINGS = {
    "LoadNIMNode": LoadNIMNode,
    "NIMFLUXNode": NIMFLUXNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadNIMNode": "Load NIM",
    "NIMFLUXNode": "NIM FLUX"
}
