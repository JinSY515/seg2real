import argparse
import gc
import json
import logging
import math
import os
import os.path as osp
import random
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import (AutoencoderKL, ControlNetModel, DDIMScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from PIL import Image
from src.dataset.real_seg_image_bdd import RealSegDataset
from src.dataset.etri_seg_image import EtriSegDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
#from src.models.unet_2d_condition import UNet2DConditionModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img_etri import Pose2ImagePipeline
from src.utils.util import (delete_additional_ckpt, import_filename,
                            seed_everything)
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet2DConditionModel,
        pose_guider: PoseGuider,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.pose_guider = pose_guider
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        # ref_seg,
        seg_text_prompt_embeds,
        ref_text_prompt_embeds,
        content_img, #pose_img,
        uncond_fwd: bool = False,
        guess_mode=False,
    ):
        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=ref_text_prompt_embeds, 
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        controlnet_latents=noisy_latents

        pose_cond_tensor = content_img.to(device="cuda")
        down_block_res_samples, mid_block_res_sample  = self.pose_guider(
            noisy_latents,
            timesteps,
            encoder_hidden_states=seg_text_prompt_embeds,
            controlnet_cond=pose_cond_tensor, 
            conditioning_scale=9.0,
            # guess_mode=guess_mode,
            return_dict=False,     
        )
        
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states= seg_text_prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0] 
        del seg_text_prompt_embeds, ref_text_prompt_embeds, pose_cond_tensor
        gc.collect()
        torch.cuda.empty_cache()
        return model_pred


def main(args):
    cfg = OmegaConf.load(args.config)
    inference_config_path = cfg.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)
    
    generator = torch.Generator().manual_seed(args.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else: 
        weight_dtype = torch.float32
    
    # initiliaze network, ecncoder, guider
    reference_unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2",
        subfolder="unet",
    ).to(dtype=weight_dtype,device="cuda")

    denoising_unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2",
        subfolder="unet",
    ).to(dtype=weight_dtype,device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = ControlNetModel.from_unet(denoising_unet).to(dtype=weight_dtype,device="cuda")
    pose_guider.load_state_dict(torch.load(cfg.pose_guider_path, map_location="cpu"))
    # pose_guider = ControlNetModel.from_pretrained(
    #     cfg.pose_guider_path, 
    #     low_cpu_mem_usage = False, 
    #     device_map = None,
    #     strict=False,
    # ).to(device="cuda")
    
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)


    width, height = args.width, args.height
   
    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(cfg.denoising_unet_path),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(cfg.reference_unet_path),
    )
    

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,

    ).to("cuda",dtype=weight_dtype)

    pil_images = []

    if args.ref_path == "":
        ref_lst = [x for x in cfg["test_cases"].keys() if x != 'default']
    else:
        ref_lst = [args.ref_path]
    
    for ref_image_path in tqdm(ref_lst):
        print("Running inference...")
        print(f"############ Current Reference image : {ref_image_path.split('/')[-1]} ############")
        
        if args.ref_path == "":
            pose_lst = cfg["test_cases"][ref_image_path]
        else:
            pose_lst = [args.seg_path]
        
        assert len(pose_lst) != 0, "No segmentation map image exists!"
        
        for pose_image_path in pose_lst:
            pose_name = pose_image_path.split("/")[-1].replace(".png", "")

            # pose_panoptic_name = pose_image_path.replace("/gtFine/", "/gtFinePanopticImages/").replace("_gtFine_color", "_gtFinePanopticImages")
            ref_name = ref_image_path.split("/")[-1].replace(".jpg", "")
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            cond_image_pil = Image.open(pose_image_path).convert("RGB")
            
            image = pipe(
                ref_image_pil,
                cond_image_pil,
                width,
                height,
                20,
                1.0,
                generator=generator,
            ).images
            image = image[0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            res_image_pil = Image.fromarray((image * 255).astype(np.uint8))

            # Save ref_image, src_image and the generated_image
            w, h = res_image_pil.size
            if cond_image_pil is not None:
                canvas = Image.new("RGB", (w * 3, h), "white")
                ref_image_pil = ref_image_pil.resize((w, h))
                cond_image_pil = cond_image_pil.resize((w ,h ))
                canvas.paste(ref_image_pil, (0, 0))
                canvas.paste(cond_image_pil, (w , 0))
                canvas.paste(res_image_pil, (w * 2, 0))
            else:
                canvas = Image.new("RGB", (w * 2, h), "white")
                ref_image_pil = ref_image_pil.resize((w, h))
                canvas.paste(ref_image_pil, (0, 0))
                canvas.paste(res_image_pil, (w  , 0))

            pil_images.append({"ref_name": ref_name, "pose_name": pose_name,"img": canvas})
            
    for img in tqdm(pil_images):
        save_ref_dir = f"{args.output_dir}/{img['ref_name']}"
        if not os.path.exists(save_ref_dir) : os.makedirs(save_ref_dir)
        img['img'].save(f"{save_ref_dir}/{img['pose_name']}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    # config.data -> 256,256
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=784)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,default="./results")

    parser.add_argument("--ref_path",type=str,default="./test_imgs/reference/frankfurt_000000_001016.png")
    parser.add_argument("--seg_path",type=str,default="./test_imgs/seg_map/lindau_000014_000019.png")

    args, unknown = parser.parse_known_args()

    main(args)