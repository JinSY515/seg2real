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
from diffusers.models.attention_processor import XFormersAttnProcessor, LoRAXFormersAttnProcessor

mode = "ours"
    
if mode == "controlnext":
    from src.models.mutual_self_attention import ReferenceAttentionControl
    # from src.pipelines.pipeline_controlnext import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline
    from src.pipelines.pipeline_seg2real_controlnext_abl import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline 
# elif mode == "masactrl":
#     from src.models.mutual_self_attention_masactrl import ReferenceAttentionControl 
#     from src.pipelines.pipeline_seg2real_controlnext_abl import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline 
elif mode == "sa_aug":
    from src.models.mutual_self_attention import ReferenceAttentionControl
    from src.pipelines.pipeline_seg2real_controlnext_abl import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline
elif mode == "ours":
    from src.models.mutual_self_attention import ReferenceAttentionControl
    from src.pipelines.pipeline_seg2real_controlnext_ours_ctrlx import StableDiffusionControlNeXtPipeline as Pose2ImagePipeline
    from src.models.matching_module import OurModel
    from src.models.attention import BasicTransformerBlock
    from src.models.attention_processor import MatchAttnProcessor

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


def set_up_lora_attn_processor_(unet, blocks, fuse_block="down"):
    blocks = [m for m in torch_dfs(blocks) if isinstance(m,BasicTransformerBlock)]
    for idx, m in enumerate(blocks):
        processor = LoRAXFormersAttnProcessor(
            hidden_size=m.attn1.to_q.out_features,
            cross_attention_dim=None
        )
        processor.requires_grad_(True)
        m.attn1.processor = processor 
   

def set_up_lora_attn_processor(unet, fusion_blocks="full"):
    if fusion_blocks == "full":
        set_up_lora_attn_processor_(unet, unet.down_blocks, fuse_block="down")
        set_up_lora_attn_processor_(unet, unet.mid_block, fuse_block="mid")
        set_up_lora_attn_processor_(unet, unet.up_blocks, fuse_block="up")
    elif fusion_blocks == "midup":
        set_up_lora_attn_processor_(unet, unet.mid_block, fuse_block="mid")
        set_up_lora_attn_processor_(unet, unet.up_blocks, fuse_block="up")
    elif fusion_blocks == "up":
        set_up_lora_attn_processor_(unet, unet.up_blocks, fuse_block="up")
    
def set_up_match_attn_processor_(blocks, block_type="down"):
    blocks_ = [m for m in torch_dfs(blocks) if isinstance(m,BasicTransformerBlock)]
    for idx, m in enumerate(blocks_):
        if block_type == "down" and idx == 0:
            continue
        embed_dim = 32 
        processor = MatchAttnProcessor(
            embed_dim=embed_dim,
            hidden_size=m.attn1.to_q.out_features,
        )
        processor.requires_grad_(True)
        m.attn1.processor = processor 

def set_up_match_attn_processor(unet, fusion_blocks):
    device, dtype = unet.conv_in.weight.device, unet.conv_in.weight.dtype
    scale_idx=0
    
    if fusion_blocks == "full":
        set_up_match_attn_processor_(unet.down_blocks, block_type="down")
        set_up_match_attn_processor_(unet.mid_block, block_type="mid")
        set_up_match_attn_processor_(unet.up_blocks, block_type="up")
    elif fusion_blocks == "midup":
        set_up_match_attn_processor_(unet.mid_block, block_type="mid")
        set_up_match_attn_processor_(unet.up_blocks, block_type="up")
    elif fusion_blocks == "down":
        set_up_match_attn_processor_(unet.down_blocks, block_type="down")
    elif fusion_blocks == "up":
        set_up_match_attn_processor_(unet.up_blocks, block_type="up")
    
    

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
    matcher,
    width,
    height,
):
    generator = torch.Generator().manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)
    
    pipe = Pose2ImagePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        controlnext=controlnext,
        scheduler=scheduler,
        matcher=matcher,
    )

    pipe= pipe.to("cuda", dtype=torch.float32) 
    ref_image_paths = cfg.validation.real_paths 
    pose_image_paths = cfg.validation.seg_paths
    pil_images = []
    val_json = cfg.validation.val_json
    set_up_match_attn_processor(denoising_unet, fusion_blocks="full")
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
            os.makedirs(f"/mnt/data3/siyoon/attn_vis/1108_matcher_deepmatch27000/{mode}/ref_{ref_name}", exist_ok=True)
            save_name = f"/mnt/data3/siyoon/attn_vis/1108_matcher_deepmatch27000/{mode}/ref_{ref_name}/tgt_{pose_name}"
            image = pipe(
                prompt="A photo of a <sks> driving scene",
                ref_image=ref_image_pil,
                ref_seg_image=ref_seg_pil,
                tgt_seg_image=pose_image_pil,
                height=height,
                width=width,
                num_inference_steps=20,
                guidance_scale=5.5,
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
            os.makedirs(f"{cfg.output_dir}/{mode}", exist_ok=True)
            sample_name = f"{ref_name}_{pose_name}"
            img = canvas 
            with TemporaryDirectory() as temp_dir:
                os.makedirs(f"{temp_dir}", exist_ok=True)
                out_file =f"{cfg.output_dir}/{mode}/{sample_name}.png"

                img.save(out_file)
         
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
        load_safetensors(reference_unet, cfg.pretrained_unet_path, strict=True, load_weight_increasement=False)
        load_safetensors(denoising_unet, cfg.pretrained_unet_path, strict=True, load_weight_increasement=False)
    if cfg.net.controlnext_path is not None:
        load_safetensors(controlnext, cfg.net.controlnext_path, strict=True)
    
    reference_unet.enable_xformers_memory_efficient_attention()
    denoising_unet.enable_xformers_memory_efficient_attention()
    denoising_unet.to(device="cuda", dtype=weight_dtype) # dtype=weight_dtype)
    if cfg.net.load_lora:
        set_up_lora_attn_processor(reference_unet, fusion_blocks="full")
        lora_sd = torch.load(cfg.net.reference_unet_lora_path) 
        key_mappings = {
            'to_q_lora': 'to_q.lora_layer',
            'to_v_lora': 'to_v.lora_layer',
            'to_out_lora': 'to_out.lora_layer',
            'to_k_lora' : 'to_k.lora_layer'
        }

        lora_sd = {key_mappings.get(k, k): v for k, v in lora_sd.items()}
        
        reference_unet.load_state_dict(
            lora_sd, strict=False
        )
    
    if cfg.net.load_trained_reference:
        reference_unet.load_state_dict(
            torch.load(cfg.net.reference_unet_full_path)
        )
    if cfg.net.matcher_path is not None:
        set_up_match_attn_processor(denoising_unet, fusion_blocks="full")
        matcher = OurModel()
        matcher_ = torch.load(cfg.net.matcher_path)
    # denoising_unet.load_state_dict(
    #     torch.load("/mnt/data4/siyoon/i2i_control/bdd_1029/controlnext_final_checkpoint/train_semantic_lora_whole_frozen_dataaug_consistenttimesteps/matching_module_include64_lr1.0e-4_res512_bs1x1/stage1/denoising_unet-13200.pth")
    # )
    if cfg.net.trained_den:
        den_sd = torch.load(cfg.net.denoising_unet_full_path)
        set_up_match_attn_processor(denoising_unet, fusion_blocks="full")
        matcher = OurModel()
        matcher_ = torch.load(cfg.net.matcher_path)
        missing_keys, _ = denoising_unet.load_state_dict(
            den_sd, strict=False
        )
        # import pdb; pdb.set_trace()
    reference_unet.to(device="cuda",dtype=weight_dtype)
    controlnext.to(dtype=weight_dtype)
    # net = net.to(dtype=weight_dtype)
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    # val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    # sched_kwargs.update(
    #     rescale_betas_zero_snr=True,
    #     timestep_spacing="leading",
    #     # set_alpha_to_one=False,         
    #     # prediction_type="v_prediction",
    # )
    # sched_kwargs.update(
    #     # rescale_betas_zero_snr=True,
    #     # timestep_spacing="leading",
    #     # set_alpha_to_one=False,         
    #     # prediction_type="v_prediction",
    # )
    # sched_kwargs.update({
    #     "beta_schedule": "scaled_linear",
    # })
    # val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    val_noise_scheduler_config = {'num_train_timesteps': 1000, 
                        'beta_start': 0.00085, 
                        'beta_end': 0.012, 
                        'beta_schedule': 'scaled_linear', 
                        'trained_betas': None, 
                        'solver_order': 2, 
                        'prediction_type': 'epsilon', 
                        'thresholding': False,
                        'dynamic_thresholding_ratio': 0.995, 
                        'sample_max_value': 1.0, 
                        'predict_x0': True, 
                        'solver_type': 'bh2', 
                        'lower_order_final': True, 
                        'disable_corrector': [], 
                        'solver_p': None, 
                        'use_karras_sigmas': False, 
                        'use_exponential_sigmas': False, 
                        'use_beta_sigmas': False, 
                        'timestep_spacing': 'linspace', 
                        'steps_offset': 1, 
                        'final_sigmas_type': 'zero', 
                        '_use_default_values': ['prediction_type', 'disable_corrector', 'use_beta_sigmas', 'solver_p', 'sample_max_value', 'use_exponential_sigmas', 'use_karras_sigmas', 'predict_x0', 'timestep_spacing', 'lower_order_final', 'thresholding', 'rescale_betas_zero_snr', 'dynamic_thresholding_ratio', 'solver_order', 'solver_type', 'final_sigmas_type'],
                        'skip_prk_steps': True, 
                        'set_alpha_to_one': False, 
                        '_class_name': 'UniPCMultistepScheduler', 
                        '_diffusers_version': '0.6.0', 
                        'clip_sample': False}
    val_noise_scheduler = DDIMScheduler.from_config(val_noise_scheduler_config)
    # val_noise_scheduler =  UniPCMultistepScheduler.from_config(val_noise_scheduler_config)
    # val_noise_scheduler = UniPCMultistepScheduler(**sched_kwargs) 
    # import pdb; pdb.set_trace()
    
    vae = AutoencoderKL.from_pretrained(
        cfg.vae_model_path, subfolder="vae"
    ).to("cuda", dtype=torch.float16)

    
    return reference_unet, denoising_unet, controlnext, val_noise_scheduler, matcher, vae

def main(cfg):
    reference_unet, denoising_unet, controlnext, scheduler, matcher, vae = load_models(config)  
    denoising_unet.eval()
    reference_unet.eval() 
    controlnext.eval()
    vae.eval()
    matcher.eval()
    sample_dicts = log_validation(
        cfg=cfg,
        vae=vae, 
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        controlnext=controlnext,
        scheduler=scheduler,
        matcher=matcher,
        width=cfg.data.val_width,
        height=cfg.data.val_height,
    )
    # os.makedirs(f"{cfg.output_dir}/{mode}", exist_ok=True)
    # for sample_id, sample_dict in enumerate(sample_dicts):
    #     sample_name = sample_dict["name"]
    #     img = sample_dict["img"]
    #     with TemporaryDirectory() as temp_dir:
    #         os.makedirs(f"{temp_dir}", exist_ok=True)
    #         out_file =f"{cfg.output_dir}/{mode}/{sample_name}.png"

    #         img.save(out_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, default="./configs/inference/stage1_ours.yaml")
    parser.add_argument("--mode")
    args = parser.parse_args() 
    
    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    
    main(config)
    