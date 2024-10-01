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
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
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
        ref_seg,
        seg_text_prompt_embeds,
        ref_text_prompt_embeds,
        content_img, #pose_img,
        uncond_fwd: bool = False,
        guess_mode=False,
    ):
        if not uncond_fwd:
            # import pdb; pdb.set_trace()
            ref_timesteps = torch.zeros_like(timesteps)
            ref_controlnet_latents=ref_image_latents 
            ref_cond_tensor = ref_seg.to(device="cuda")
            # import pdb; pdb.set_trace()
            ref_down_block_res_samples, ref_mid_block_res_sample = self.pose_guider(
                ref_controlnet_latents,
                ref_timesteps,
                encoder_hidden_states=ref_text_prompt_embeds,
                controlnet_cond=ref_cond_tensor,
                return_dict=False,
            )
            # self.reference_unet(
            #     ref_image_latents,
            #     ref_timesteps,
            #     encoder_hidden_states=ref_text_prompt_embeds, 
            #     return_dict=False,
            # )
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states= ref_text_prompt_embeds,
                down_block_additional_residuals=[
                    sample for sample in ref_down_block_res_samples
                ],
                mid_block_additional_residual=ref_mid_block_res_sample,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        controlnet_latents=noisy_latents
        pose_cond_tensor = content_img.to(device="cuda")
        down_block_res_samples, mid_block_res_sample  = self.pose_guider(
            controlnet_latents,
            timesteps,
            encoder_hidden_states=seg_text_prompt_embeds,
            controlnet_cond=pose_cond_tensor, 
            # guess_mode=guess_mode,
            return_dict=False,     
        )
        
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states= seg_text_prompt_embeds,
            down_block_additional_residuals=[
                sample for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0] 
        del seg_text_prompt_embeds, ref_text_prompt_embeds, pose_cond_tensor
        gc.collect()
        torch.cuda.empty_cache()
        return model_pred

def log_validation(
    cfg,
    vae,
    image_enc,
    net,
    scheduler,
    width,
    height,
):
    reference_unet = net.reference_unet
    denoising_unet = net.denoising_unet
    pose_guider = net.pose_guider
    generator = torch.Generator().manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    image_enc = image_enc.to(dtype=torch.float32)

    pipe = Pose2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,

    )
    ref_image_paths = cfg.validation.real_paths 
    pose_image_paths = cfg.validation.seg_paths
    pil_images = []
    for ref_image_path in ref_image_paths:
        for pose_image_path in pose_image_paths:
            pose_name = pose_image_path.split("/")[-1].replace(".png", "").replace(".jpg", "")

            # pose_panoptic_name = pose_image_path.replace("/gtFine/", "/gtFinePanopticImages/").replace("_gtFine_color", "_gtFinePanopticImages")
            ref_name = ref_image_path.split("/")[-1].replace(".jpg", "").replace(".png","")
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB")
            ref_seg_image_path = ref_image_path.replace("/images/10k/val/", "/labels/sem_seg/colormaps/val/").replace(".jpg",".png")
            ref_seg_image_pil = Image.open(ref_seg_image_path).convert("RGB")
            cond_image_pil = None
            image = pipe(
                ref_image_pil,
                # ref_seg_image_pil,
                pose_image_pil,
                cond_image_pil,
                # ref_json,
                # pose_json,
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
                canvas = Image.new("RGB", (w * 4, h), "white")
                ref_image_pil = ref_image_pil.resize((w, h))
                pose_image_pil = pose_image_pil.resize((w, h))
                cond_image_pil = cond_image_pil.resize((w ,h ))
                canvas.paste(ref_image_pil, (0, 0))
                canvas.paste(pose_image_pil, (w, 0))
                canvas.paste(cond_image_pil, (w * 2, 0))
                canvas.paste(res_image_pil, (w * 3, 0))
            else:
                canvas = Image.new("RGB", (w * 3, h), "white")
                ref_image_pil = ref_image_pil.resize((w, h))
                pose_image_pil = pose_image_pil.resize((w, h))
                canvas.paste(ref_image_pil, (0, 0))
                canvas.paste(pose_image_pil, (w, 0))
                canvas.paste(res_image_pil, (w * 2, 0))

            pil_images.append({"name": f"{ref_name}_{pose_name}", "img": canvas})

    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)

    del vae
    del pipe
    torch.cuda.empty_cache()

    return pil_images

def load_models(cfg):
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    
    denoising_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")
    
    pose_guider = ControlNetModel.from_pretrained(
        cfg.controlnet_path, 
        low_cpu_mem_usage=False, 
        device_map=None, 
        strict=False,
    ).to(device="cuda")
    
    reference_unet.load_state_dict(
        torch.load(cfg.net.reference_unet_path, map_location='cpu')
    )
    
    denoising_unet.load_state_dict(
        torch.load(cfg.net.denoising_unet_path, map_location="cpu")
    )
    
    pose_guider.load_state_dict(
        torch.load(cfg.net.controlnet_path, map_location='cpu')
    )
    
    reference_control_writer = ReferenceAttentionControl(
        reference_unet, 
        do_classifier_free_guidance=False, 
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet, 
        do_classifier_free_guidance=False, 
        mode="read",
        fusion_blocks="full"
    )
    
    net = Net(
        reference_unet, 
        denoising_unet, 
        pose_guider,
        reference_control_writer, 
        reference_control_reader,
    )
    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path
    ).to(device="cuda")
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    sched_kwargs.update(
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing",
        prediction_type="v_prediction",
    )
    # sched_kwargs.update({"beta_schedule": "scaled_linear"})
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model_path
    ).to("cuda")

    return net, image_enc, val_noise_scheduler, vae

def main(cfg):
    net, image_enc, scheduler, vae = load_models(config)  
    net.denoising_unet.eval()
    net.reference_unet.eval() 
    net.pose_guider.eval()
    vae.eval()
    image_enc.eval()
    sample_dicts = log_validation(
        cfg=cfg,
        vae=vae, 
        image_enc=image_enc,
        net=net, 
        scheduler=scheduler,
        width=cfg.data.val_width,
        height=cfg.data.val_height,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    for sample_id, sample_dict in enumerate(sample_dicts):
        sample_name = sample_dict["name"]
        img = sample_dict["img"]
        with TemporaryDirectory() as temp_dir:
            os.makedirs(f"{temp_dir}", exist_ok=True)
            out_file =f"{cfg.output_dir}/{sample_name}.png"

            img.save(out_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default="./configs/inference/stage1.yaml")
    args = parser.parse_args() 
    
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
        
    main(config)
    