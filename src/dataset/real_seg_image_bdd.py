import json
import os
import random
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from retrieval.data_retrieve import retrieve
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


# torch.multiprocessing.set_start_method('spawn')
class RealSegDataset(Dataset):
    def __init__(
        self,
        real_images_path,
        seg_images_path,
        img_size,
        seg_size,
        center_crop,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        mode="default",
        cond_mode="single",
    ):
        super().__init__()

        self.img_size = img_size
        self.center_crop = center_crop
        self.seg_size = seg_size
        self.img_scale = img_scale 
        self.img_ratio = img_ratio

        self.real_images_path = list(glob(f"{real_images_path}/*.jpg", recursive=True))
        self.tgt_images_path = list(glob(f"{seg_images_path}/*.png", recursive=True))
        self.num_real_images = len(self.real_images_path)
        self.seg_images_path = list(glob(f"{seg_images_path}/*.png", recursive=True))
        self.num_seg_images = len(self.seg_images_path)
        self._length = self.num_real_images


        self.clip_image_processor = CLIPImageProcessor()
        self.tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device="cuda")

        self.mode = mode
        assert self.mode in ["default", "panoptic",  "retrieve_adj", "retrieve_dino"]
        self.cond_mode = cond_mode
        # assert self.cond_mode in ["single", "multi_depth", "multi_panop", "multi_canny"]
        # self.transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             self.img_size,
        #             scale=self.img_scale,
        #             ratio=self.img_ratio,
        #             interpolation=transforms.InterpolationMode.BILINEAR,
        #         ),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5], [0.5]),
        #     ]
        # )
        
        self.random_resized_crop = transforms.RandomResizedCrop(
            self.img_size, 
            scale=self.img_scale, 
            ratio=self.img_ratio,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        
        self.transform = transforms.Compose(
            [
                self.random_resized_crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transform = transforms.Compose(
            [
                self.random_resized_crop,
                transforms.ToTensor()
            ]
        )

        # self.cond_transform = transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(
        #             self.img_size,
        #             scale=self.img_scale,
        #             ratio=self.img_ratio,
        #             interpolation=transforms.InterpolationMode.BILINEAR,
        #         ),
        #         transforms.ToTensor(),
        #     ]
        # )


    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        example = {}
        # index == ref_img_idx
        seg_image_name = str(self.seg_images_path[index % self.num_seg_images])
        if self.mode == "panoptic":
            mode_image_name = seg_image_name.replace("/sem_seg/", "/ins_seg/")
        elif self.mode in ["default", "retrieve_dino"]:
            mode_image_name = seg_image_name

        seg_image = Image.open(mode_image_name)
        # /media/dataset1/CityScape/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        tgt_image_name = seg_image_name.replace("/labels/sem_seg/colormaps/train/", "/images/10k/train/").replace(".png", ".jpg")
        tgt_image = Image.open(tgt_image_name) #Image.open(self.tgt_images_path[index % self.num_seg_images])

        json_name = os.path.dirname(seg_image_name).replace("/colormaps/train", "/polygons/sem_seg_train.json")
        example["seg_json_name"] = json_name

        if self.mode in ["default", "panoptic"]:
            ref_img_idx = random.randint(0, self.num_real_images)
            while ref_img_idx == index :
                ref_img_idx = random.randint(0, self.num_real_images)

            ref_real_image_name = str(self.real_images_path[ref_img_idx % self.num_real_images])
            # ref_real_image_name = ref_image_name.replace("/labels/sem_seg/colormaps/", "/images/10k/").replace(".png", ".jpg")

        elif self.mode == "retrieve_dino":
            ref_real_image_name = retrieve(tgt_image_name)
            ref_image_name = ref_real_image_name.replace("/images/10k/","/labels/sem_seg/colormaps/").replace(".jpg", ".png")
       
        ref_image = Image.open(ref_real_image_name)
        # ref_json_name = ref_image_name.replace("/colormaps/train/*.png", "/polygons/sem_seg_train.json")
        # example["ref_json_name"] = ref_json_name

        # ref_seg_image_name = ref_real_image_name.replace("/labels/sem_seg/colormaps/", "/images/10k/").replace(".png", ".jpg")
        # ref_seg_image = Image.open(ref_seg_image_name)

        if not tgt_image.mode == "RGB":
            tgt_image = tgt_image.convert("RGB")
        if not seg_image.mode == "RGB":
            seg_image = seg_image.convert("RGB")
        if not ref_image.mode == "RGB":
            ref_image = ref_image.convert("RGB")
        # if not ref_seg_image.mode == "RGB":
        #     ref_seg_image = ref_seg_image.convert("RGB")


        # tgt_image = self.transform(tgt_image)
        # seg_image = self.cond_transform(seg_image)
        # ref_image = self.transform(ref_image)
        rng_state = torch.get_rng_state()
        tgt_image = self.augmentation(tgt_image, self.transform, rng_state)
        seg_image = self.augmentation(seg_image, self.cond_transform, rng_state)
        ref_image = self.augmentation(ref_image, self.transform, rng_state)
        
        # ref_seg_image = self.cond_transform(ref_seg_image)

        ## multi cond
        if self.cond_mode == "single":
            pass
        elif self.cond_mode == "multi_panop":
            panop_image_name = tgt_image_name.replace("/images/10k/train/", "/labels/ins_seg/colormaps/train/").replace(".jpg", ".png")
            panop_image = Image.open(panop_image_name)
            if not panop_image.mode == "RGB":
                panop_image = panop_image.convert("RGB")
            panop_image = self.cond_transform(panop_image)
            example["panop_cond_image"] = panop_image

        elif self.cond_mode == "multi_canny":
            canny_image_name = tgt_image_name.replace("/leftImg8bit/", "/leftImg8bit_canny/")
            canny_image = Image.open(canny_image_name)
            if not canny_image.mode == "RGB":
                canny_image = canny_image.convert("RGB")
            canny_image = self.cond_transform(canny_image)
            example["canny_cond_image"] = canny_image
        elif self.cond_mode == "multi_depth":
            pass
        
        example["img"] = tgt_image 
        example["seg_image"] = seg_image
        example["ref_image"] = ref_image
        example["seg_image_name"] = seg_image_name
        # example["ref_seg_image"] = ref_seg_image

        ref_image = ((ref_image + 1.0) / 2)
        clip_image = self.clip_image_processor(
            images=ref_image, return_tensors="pt"
        ).pixel_values[0]

        example["clip_image"] = clip_image
        return example

    def __len__(self):
        return self._length
