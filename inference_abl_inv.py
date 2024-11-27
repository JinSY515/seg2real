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
from utils.visualize_hacked_attention import visualize_hacked_attention
import diffusers
# import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import (AutoencoderKL,  DDIMScheduler, DDPMScheduler,
                       StableDiffusionControlNetPipeline, UniPCMultistepScheduler)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from PIL import Image
from src.dataset.real_seg_image_bdd import RealSegDataset
# from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
# from src.models.unet import UNet2DConditionModel 
# from src.pipelines.pipeline_seg2real_masactrl import Pose2ImagePipeline
from src.utils.util import (delete_additional_ckpt, import_filename,
                            seed_everything)
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)
from src.models.controlnext import ControlNeXtModel
from safetensors.torch import load_file
mode = "sa_aug"
    
if mode == "controlnext":
    from src.models.mutual_self_attention import ReferenceAttentionControl
    # from src.pipelines.pipeline_controlnext import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline
    from src.pipelines.pipeline_seg2real_controlnext_abl_inv import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline 
elif mode == "masactrl":
    from src.models.mutual_self_attention_masactrl import ReferenceAttentionControl 
    from src.pipelines.pipeline_seg2real_controlnext_abl_inv import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline 
elif mode == "sa_aug":
    from src.models.mutual_self_attention import ReferenceAttentionControl
    from src.pipelines.pipeline_seg2real_controlnext_abl_inv import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline

class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet2DConditionModel,
        controlnext: ControlNeXtModel,
        reference_control_writer,
        reference_control_reader,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.controlnext = controlnext
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
    
    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        ref_timesteps,
        ref_seg_img,
        seg_text_prompt_embeds,
        ref_text_prompt_embeds,
        content_img, #pose_img,
        uncond_fwd: bool = False,
        guess_mode=False,
    ):
        ref_controlnext_output = self.controlnext(ref_seg_img, ref_timesteps)
        
        out = self.reference_unet(
            ref_image_latents,
            ref_timesteps,
            encoder_hidden_states= seg_text_prompt_embeds, #ref_text_prompt_embeds,
            conditional_controls=ref_controlnext_output,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)
        
        den_controlnext_output = self.controlnext(content_img, timesteps)
        
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states= seg_text_prompt_embeds,
            conditional_controls=den_controlnext_output,
            return_dict=False,
        )[0] 
        del seg_text_prompt_embeds, ref_text_prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()
        return model_pred


       
        
def log_validation(
    cfg,
    vae,
    reference_unet,
    denoising_unet,
    controlnext,
    scheduler,
    width,
    height,
):
    generator = torch.Generator().manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    reference_unet.enable_xformers_memory_efficient_attention()
    denoising_unet.enable_xformers_memory_efficient_attention()
    controlnext.enable_xformers_memory_efficient_attention()
    pipe = Pose2ImagePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        controlnext=controlnext,
        scheduler=scheduler,
    )

    pipe= pipe.to("cuda", dtype=torch.float32) 
    ref_image_paths = cfg.validation.real_paths 
    pose_image_paths = cfg.validation.seg_paths
    pil_images = []
    val_json = cfg.validation.val_json
    for ref_image_path in ref_image_paths:
        for pose_image_path in pose_image_paths:
            pose_name = pose_image_path.split("/")[-1]#.replace(".png", "")
            ref_name = ref_image_path.split("/")[-1]#.replace(".jpg", "")
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB")
            ref_seg_path = ref_image_path.replace("/images/10k/val/","/labels/10k/sem_seg/colormaps/val/").replace(".jpg", ".png")
            ref_seg_pil = Image.open(ref_seg_path).convert("RGB")
            cond_image_pil = None
            # image = pipe(
            #     ref_image_pil,
            #     ref_seg_pil,
            #     pose_image_pil,
            #     cond_image_pil,
            #     val_json, 
            #     ref_name, 
            #     pose_name,
            #     # ref_json,
            #     # pose_json,
            #     width,
            #     height,
            #     20,
            #     1.0,
            #     generator=generator,
            # ).images
            os.makedirs(f"/mnt/data3/siyoon/attn_vis/1031_gs7.5/{mode}/ref_{ref_name}", exist_ok=True)
            save_name = f"/mnt/data3/siyoon/attn_vis/1031_gs7.5/{mode}/ref_{ref_name}/tgt_{pose_name}"
            image = pipe(
                prompt="A photo of <sks> driving scene",
                ref_image=ref_image_pil,
                ref_seg_image=ref_seg_pil,
                tgt_seg_image=pose_image_pil,
                height=height,
                width=width,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=generator,
                save_name=save_name,
                mode=mode,
            ).images
            # image = image[0].permute(1, 2, 0).cpu().numpy()  # (3, 512, 512)
            # res_image_pil = Image.fromarray((image * 255).astype(np.uint8))
            res_image_pil = image[0]
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

    del vae
    del pipe
    torch.cuda.empty_cache()

    return pil_images

def load_safetensors(model, safetensors_path, strict=True, load_weight_increasement=False):
    if not load_weight_increasement:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        # pretrained_state_dict = model.state_dict()
        # # for k in state_dict.keys():
        # #     pretrained_state_dict[k] = state_dict[k]
        model.load_state_dict(state_dict, strict=strict)
    else:
        if safetensors_path.endswith('.safetensors'):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

def load_models(cfg):
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )
    denoising_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )
    weight_dtype=torch.float32
    controlnext = ControlNeXtModel(controlnext_scale=1.0)
    if cfg.pretrained_unet_path is not None:
        load_safetensors(reference_unet, cfg.net.reference_unet_path, strict=True, load_weight_increasement=False)
        load_safetensors(denoising_unet, cfg.net.denoising_unet_path, strict=True, load_weight_increasement=False)
    if cfg.net.controlnext_path is not None:
        load_safetensors(controlnext, cfg.net.controlnext_path, strict=True)
    
    denoising_unet.to(device="cuda", dtype=weight_dtype) # dtype=weight_dtype)
    reference_unet.to(device="cuda",dtype=weight_dtype)
    controlnext.to(dtype=weight_dtype)
    # net = net.to(dtype=weight_dtype)
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    # sched_kwargs.update(
    #     rescale_betas_zero_snr=True,
    #     timestep_spacing="leading",
    #     # set_alpha_to_one=False,         
    #     # prediction_type="v_prediction",
    # )
    # sched_kwargs.update(
    #     clip_sample=True,
    #     rescale_betas_zero_snr=True,
    #     # timestep_spacing="leading",
    #     # set_alpha_to_one=False,         
    #     # prediction_type="v_prediction",
    # )
    # sched_kwargs.update({
    #     "beta_schedule": "scaled_linear",
    # })
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    # val_noise_scheduler_config = {'num_train_timesteps': 1000, 
    #                     'beta_start': 0.00085, 
    #                     'beta_end': 0.012, 
    #                     'beta_schedule': 'scaled_linear', 
    #                     'trained_betas': None, 
    #                     'solver_order': 2, 
    #                     'prediction_type': 'epsilon', 
    #                     'thresholding': False,
    #                     'dynamic_thresholding_ratio': 0.995, 
    #                     'sample_max_value': 1.0, 
    #                     'predict_x0': True, 
    #                     'solver_type': 'bh2', 
    #                     'lower_order_final': True, 
    #                     'disable_corrector': [], 
    #                     'solver_p': None, 
    #                     'use_karras_sigmas': False, 
    #                     'use_exponential_sigmas': False, 
    #                     'use_beta_sigmas': False, 
    #                     'timestep_spacing': 'linspace', 
    #                     'steps_offset': 1, 
    #                     'final_sigmas_type': 'zero', 
    #                     '_use_default_values': ['prediction_type', 'disable_corrector', 'use_beta_sigmas', 'solver_p', 'sample_max_value', 'use_exponential_sigmas', 'use_karras_sigmas', 'predict_x0', 'timestep_spacing', 'lower_order_final', 'thresholding', 'rescale_betas_zero_snr', 'dynamic_thresholding_ratio', 'solver_order', 'solver_type', 'final_sigmas_type'],
    #                     'skip_prk_steps': True, 
    #                     'set_alpha_to_one': False, 
    #                     '_class_name': 'UniPCMultistepScheduler', 
    #                     '_diffusers_version': '0.6.0', 
    #                     'clip_sample': False}
    # val_noise_scheduler =  UniPCMultistepScheduler.from_config(val_noise_scheduler_config)
    # val_noise_scheduler =  DDIMScheduler.from_config(val_noise_scheduler_config)
    # val_noise_scheduler = UniPCMultistepScheduler(**sched_kwargs) 
    # import pdb; pdb.set_trace()
    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model_path, subfolder="vae"
    ).to("cuda", dtype=torch.float16)

    return reference_unet, denoising_unet, controlnext, val_noise_scheduler, vae

def main(cfg):
    reference_unet, denoising_unet, controlnext, scheduler, vae = load_models(config)  
    denoising_unet.eval()
    reference_unet.eval() 
    controlnext.eval()
    vae.eval()
    sample_dicts = log_validation(
        cfg=cfg,
        vae=vae, 
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        controlnext=controlnext,
        scheduler=scheduler,
        width=cfg.data.val_width,
        height=cfg.data.val_height,
    )
    os.makedirs(f"{cfg.output_dir}/{mode}", exist_ok=True)
    for sample_id, sample_dict in enumerate(sample_dicts):
        sample_name = sample_dict["name"]
        img = sample_dict["img"]
        with TemporaryDirectory() as temp_dir:
            os.makedirs(f"{temp_dir}", exist_ok=True)
            out_file =f"{cfg.output_dir}/{mode}/{sample_name}.png"

            img.save(out_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default="./configs/inference/stage1_ours.yaml")
    parser.add_argument("--mode")
    args = parser.parse_args() 
    
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    
    main(config)
    