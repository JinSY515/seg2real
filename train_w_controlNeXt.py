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
from safetensors.torch import load_file, save_file

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
from diffusers import (AutoencoderKL,  DDIMScheduler,
                       StableDiffusionControlNetPipeline)
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from PIL import Image
from src.dataset.real_seg_image_bdd import RealSegDataset
from src.models.mutual_self_attention import ReferenceAttentionControl
# from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.controlnext import ControlNeXtModel
# from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2img import Pose2ImagePipeline
from src.utils.util import (delete_additional_ckpt, import_filename,
                            seed_everything)
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                          CLIPTokenizer, CLIPVisionModelWithProjection)
from src.models.matching_module import OurModel
from src.models.attention import BasicTransformerBlock

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

logger = get_logger(__name__, log_level="INFO")

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

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
        ref_seg_img,
        seg_text_prompt_embeds,
        ref_text_prompt_embeds,
        content_img, #pose_img,
        uncond_fwd: bool = False,
        guess_mode=False,
    ):
        if not uncond_fwd:
            ref_timesteps = torch.ones_like(timesteps)           
            ref_controlnext_output = self.controlnext(ref_seg_img, ref_timesteps)
          
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states= ref_text_prompt_embeds,
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


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    cfg,
    vae,
    net,
    scheduler,
    accelerator,
    width,
    height,
    cond_mode,
):
    logger.info("Running validation... ")

    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    controlnext = ori_net.controlnext
    generator = torch.Generator().manual_seed(42)
    # cast unet dtype
    vae = vae.to(dtype=torch.float32)

    pipe = Pose2ImagePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=controlnext,
        scheduler=scheduler,

    )
    pipe = pipe.to(accelerator.device)
    ref_image_paths = cfg.validation.real_paths 
    pose_image_paths = cfg.validation.seg_paths

    pil_images = []
    for ref_image_path in ref_image_paths:
        for pose_image_path in pose_image_paths:
            pose_name = pose_image_path.split("/")[-1].replace(".png", "")
            ref_name = ref_image_path.split("/")[-1].replace(".jpg", "")
            ref_image_pil = Image.open(ref_image_path).convert("RGB")
            pose_image_pil = Image.open(pose_image_path).convert("RGB")
            ref_seg_path = ref_image_path.replace("/images/10k/val/","/labels/sem_seg/colormaps/val/").replace(".jpg", ".png")
            ref_seg_pil = Image.open(ref_seg_path).convert("RGB")
            cond_image_pil = None
            image = pipe(
                ref_image_pil,
                ref_seg_pil,
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

    del vae
    del pipe
    torch.cuda.empty_cache()

    return pil_images


def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns", device_placement=True, split_batches=True,
        kwargs_handlers=[kwargs],
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device="cuda")
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )#.to(device="cuda")
    denoising_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    )#.to(device="cuda")
    matching_module = OurModel()
    # unet_sd = load_file("/mnt/data4/jeeyoung/checkpoints/controlnext_SD1.5_batch_size_16_10_01/checkpoint-7200/model_1.safetensors")#.to(device="cuda")
    unet_sd = load_file(cfg.pretrained_unet_path)
    reference_unet.load_state_dict(unet_sd)
    denoising_unet.load_state_dict(unet_sd)
    controlnext = ControlNeXtModel(controlnext_scale=cfg.controlnext_scale)
    controlnext.load_state_dict(load_file(cfg.controlnext_path))
    denoising_unet.to(device="cuda")
    reference_unet.to(device="cuda")
    # denoising_unet.load_state_dict(torch.load(f"{cfg.controlnet_path}/diffusion_pytorch_model.bin"), strict=False)
    # Freeze
    vae.requires_grad_(False)
    denoising_unet.requires_grad_(True)
    #  Some top layer parames of reference_unet don't need grad
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
    # reference_unet.requires_grad_(True)
    controlnext.requires_grad_(True)
    controlnext.train()
    matching_module.requires_grad_(True)
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
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        controlnext,
        reference_control_writer,
        reference_control_reader,
    )


    if cfg.solver.enable_xformers_memory_efficient_attention:
        reference_unet.enable_xformers_memory_efficient_attention()
        denoising_unet.enable_xformers_memory_efficient_attention()
        controlnext.enable_xformers_memory_efficient_attention()

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        controlnext.enable_gradient_checkpointing()
    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    _trainable = lambda m: [p for p in m.parameters() if p.requires_grad]
    if cfg.train_matching_module:
        
        trainable_params = _trainable(matching_module) +  _trainable(net.denoising_unet) + _trainable(net.controlnext) + _trainable(net.reference_unet) 
    else:
        trainable_params = _trainable(net.denoising_unet) + _trainable(net.controlnext) + _trainable(net.reference_unet) 
        
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    train_dataset = RealSegDataset(
        real_images_path = cfg.data.real_images_path,
        seg_images_path=cfg.data.seg_images_path,
        img_size=(cfg.data.train_width, cfg.data.train_height),
        seg_size=(cfg.data.train_seg_width, cfg.data.train_seg_height),
        center_crop=False, # random crop
        mode=cfg.data.mode,
        cond_mode=cfg.data.cond_mode,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, pin_memory=False, shuffle=True, num_workers=2, persistent_workers=False
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            cfg.exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )
        # dump config file
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

    # Train!
    total_batch_size = (
        cfg.data.train_bs
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.data.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = save_dir
        # Get the most recent checkpoint
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        matching_module.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(net):
                # Convert videos to latent space
                pixel_values = batch["img"].to(weight_dtype)
                pixel_values_ref = batch["ref_image"].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * 0.18215
                    ref_latents = vae.encode(pixel_values_ref).latent_dist.sample()
                    ref_latents = ref_latents * 0.18215
                noise = torch.randn_like(latents)
                ref_noise = torch.randn_like(ref_latents)
                if cfg.noise_offset > 0.0:
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1),
                        device=noise.device,
                    )
                    ref_noise += cfg.noise_offset * torch.randn(
                        (ref_noise.shape[0], ref_noise.shape[1], 1, 1),
                        device=ref_noise.device,
                    )
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                ref_timesteps = torch.ones_like(timesteps).long() 
                tgt_seg_img = batch["seg_image"] # (bs, 3, 512, 512)
                ref_seg_img = batch["ref_seg_image"] # (bs, 3, 512, 512)
                tgt_seg_img_resized = F.interpolate(tgt_seg_img, size=(32, 32), mode='bilinear', align_corners=False)
                ref_seg_img_resized = F.interpolate(ref_seg_img, size=(32, 32), mode='bilinear', align_corners=False)
                
                
                seg_corr = (tgt_seg_img_resized[:,:,:,:,None,None] == ref_seg_img_resized[:, :, None, None, :, :]).all(dim=1)
                # import pdb; pdb.set_trace()
                refined_seg_corr = matching_module(seg_corr.unsqueeze(1).float().cpu())
                refined_seg_corr = refined_seg_corr.view(refined_seg_corr.shape[0], refined_seg_corr.shape[1], 32 * 32, 32 * 32)  # (bs, 1, 1024, 1024)

                if cfg.data.cond_mode == "multi_canny":
                    canny_cond_img = batch["canny_cond_image"]
                    content_img = torch.cat([tgt_seg_img, canny_cond_img], dim=1)
                elif cfg.data.cond_mode == "multi_panop":
                    panop_cond_img = batch["panop_cond_image"]
                    content_img = torch.cat([tgt_seg_img, panop_cond_img], dim=1)
                elif cfg.data.cond_mode == "single":
                    content_img = tgt_seg_img
                uncond_fwd = random.random() < cfg.uncond_ratio

           
                seg_text_embeds_list = []
                ref_text_embeds_list = []

                seg_json_name = batch["seg_json_name"][0]
                with torch.no_grad():
                    for batch_idx, (seg_image_name, ref_seg_image_name) in enumerate(
                        zip(
                            batch["seg_image_name"],
                            batch["ref_seg_image_name"]
                        )
                    ):
                
                        
                        with open(seg_json_name, "r") as f:
                            seg_json_data = json.load(f)
                            tgt_object_names = []
                            ref_object_names = []
                            tgt_object_names = seg_json_data[seg_image_name.split("/")[-1]]
                            ref_object_names = seg_json_data[ref_seg_image_name.split("/")[-1]]
                          
                        
                        seg_instance_prompt = "A photo of driving scene"
                        for object in tgt_object_names:
                            seg_instance_prompt += f" {object},"
                        seg_text_inputs = tokenizer(
                            seg_instance_prompt,
                            truncation=True,
                            padding="max_length",
                            max_length=77,
                            return_tensors="pt",
                        )
                        seg_input_ids = seg_text_inputs.input_ids
                        seg_text_embeds = text_encoder(seg_input_ids.to(device="cuda"), return_dict=False)[0]
                        seg_text_embeds_list.append(seg_text_embeds)

                        
                        ref_instance_prompt="A photo of driving scene"
                        for object in ref_object_names:
                            ref_instance_prompt += f" {object},"
                        ref_text_inputs = tokenizer(
                            ref_instance_prompt,
                            truncation=True,
                            padding="max_length",
                            max_length=77,
                            return_tensors="pt"
                        )
                        ref_input_ids = ref_text_inputs.input_ids
                        ref_text_embeds = text_encoder(ref_input_ids.to(device="cuda"))[0]
                        ref_text_embeds_list.append(ref_text_embeds)

                with torch.no_grad():
                    seg_text_prompt_embeds = torch.cat(seg_text_embeds_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )
                    ref_text_prompt_embeds = torch.cat(ref_text_embeds_list, dim=0).to(
                        dtype=vae.dtype, device=vae.device
                    )

                # add noise
                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                noisy_ref_latents = train_noise_scheduler.add_noise(
                    ref_latents, ref_noise, ref_timesteps
                )
                # Get the target for loss depending on the prediction type
                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                model_pred = net(
                    noisy_latents,
                    timesteps,
                    noisy_ref_latents,
                    ref_seg_img,
                    seg_text_prompt_embeds,
                    ref_text_prompt_embeds,
                    content_img, 
                    uncond_fwd,
                )

                # attn_score_bank = [m.attn_score for m in torch_dfs(net.denoising_unet) if isinstance(m, BasicTransformerBlock)]
                # for attn_gen, attn_ref in attn_score_bank:
                #     H = W = int(math.sqrt(attn_ref.shape[-1]))
                #     refined_seg_corr_ = F.interpolate(refined_seg_corr, size=(H * W, H * W), mode="nearest").expand(-1, attn_ref.shape[1], -1, -1)
                #     import pdb; pdb.set_trace()
                #     # attn_ref.add(refined_seg_corr_.to(attn_ref.device)) 
                #     attn_ref = attn_ref + refined_seg_corr_.to(attn_ref.device)
                del seg_text_prompt_embeds, ref_text_prompt_embeds
                torch.cuda.empty_cache()
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                accelerator.free_memory()
                torch.cuda.empty_cache()

            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 1)
                        accelerator.save_state(save_path)

                if global_step % cfg.val.validation_steps == 0:
                    if accelerator.is_main_process:
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(cfg.seed)

                        sample_dicts = log_validation(
                            cfg=cfg,
                            vae=vae,
                            net=net,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.train_width,
                            height=cfg.data.train_height,
                            cond_mode=cfg.data.cond_mode,
                        )
                        for sample_id, sample_dict in enumerate(sample_dicts):
                            sample_name = sample_dict["name"]
                            img = sample_dict["img"]
                            with TemporaryDirectory() as temp_dir:
                                os.makedirs(f"{temp_dir}", exist_ok=True)
                                out_file =f"{cfg.output_dir}/stage1/{global_step:06d}-{sample_name}.png"

                                img.save(out_file)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

        # save model after each epoch
        if (
            epoch + 1
        ) % (cfg.save_model_epoch_interval) == 0 and accelerator.is_main_process:
            unwrap_net = accelerator.unwrap_model(net)
            save_checkpoint(
                unwrap_net.reference_unet,
                save_dir,
                "reference_unet",
                global_step,
                total_limit=2,
            )
            save_checkpoint(
                unwrap_net.denoising_unet,
                save_dir,
                "denoising_unet",
                global_step,
                total_limit=2,
            )
            save_checkpoint(
                unwrap_net.controlnext,
                save_dir,
                "controlnext",
                global_step,
                total_limit=2,
            )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage1.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    elif args.config[-3:] == ".py":
        config = import_filename(args.config).cfg
    else:
        raise ValueError("Do not support this format config file")
    main(config)
