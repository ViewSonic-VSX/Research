"""
Implementation of ControlNet with Canny Edge map.
Other controlling styles please refer to https://github.com/lllyasviel/ControlNet/tree/main/annotator
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler


class ControlNetOperator:
    __controlnet: ControlNetModel
    __pipeline: StableDiffusionControlNetPipeline

    def __init__(self, device):
        self.__controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        )
        self.__pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "emilianJR/chilloutmix_NiPrunedFp32Fix",
            controlnet=self.__controlnet,
            torch_dtype=torch.float16,
        ).to(device)
        self.__pipeline.scheduler = DDIMScheduler.from_config(
            self.__pipeline.scheduler.config
        )

    def infer(
        self,
        image: Image,
        prompt: str,
        h: int,
        w: int,
        CFG: float,
        steps: int,
        seed: int = 0,
    ):
        generator = torch.manual_seed(seed)

        prompt = f"{prompt}, best quality, extremely detailed"

        output = self.__pipeline(
            prompt=prompt,
            image=image,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            generator=generator,
            guidance_scale=CFG,
            num_inference_steps=steps,
            height=h,
            width=w,
        )
        return output.images[0]
