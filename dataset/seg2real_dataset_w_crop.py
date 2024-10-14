from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
from transformers import AutoTokenizer, PretrainedConfig
from random import shuffle
import random
import os 
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import wandb
import random

class Seg2RealDataset(Dataset):
    def __init__(
        self,
        images_dir,
        conditioning_images_dir,
        height,
        width,
        device,
        text_prompt,
        tokenizer,
        text_encoder,
        classes,
        palette,
        essential=[],
        center_crop=True,
        accelerator=None,
        crop_scale=0.5,
        essential_class_ratio=0.03,
        crop_output_dir='./output/cropped_imgs',
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.center_crop = center_crop
        self.device=device
        self.images_dir = images_dir#list(glob(f"{images_path}/**/", recursive=True))
        self.conditioning_images_dir = conditioning_images_dir
        self.prompt = text_prompt
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.accelerator = accelerator
        self.image_seg_pair = []
        self.crop_scale = crop_scale
        self.crop_output_dir = crop_output_dir
        self.essential_class_ratio = essential_class_ratio
        self.instance_list = classes
        self.essential_classes = essential

        self.color_list = palette


        images_path = [image for image in os.listdir(self.images_dir) if image.endswith('.png') or image.endswith('.jpg')]

        all_conditioning_images = glob(os.path.join(self.conditioning_images_dir, '*'))

        conditioning_images_map = defaultdict(list)
        for img in all_conditioning_images:
            image_base_name = os.path.splitext(os.path.basename(img))[0]
            conditioning_images_map[image_base_name].append(img) 

        for image_path in tqdm(images_path):
            image_name = os.path.splitext(image_path)[0]

            if self.conditioning_images_dir:
                conditioning_image_path = conditioning_images_map.get(image_name, [])

                image_path_all = os.path.join(self.images_dir, image_path)
                if len(conditioning_image_path) == 0:
                    self.image_seg_pair.append((image_path_all, ""))
                elif len(conditioning_image_path) > 1:
                    raise ValueError(f"Multiple conditioning images found for {image_path_all}: {conditioning_image_paths}")
                else:
                    self.image_seg_pair.append((image_path_all, conditioning_image_path[0]))
            else:
                self.image_seg_pair.append((image_path_all, ""))
        
        shuffle(self.image_seg_pair)

        print(f"Total number of images: {len(self.image_seg_pair)}")
        print(random.choice(self.image_seg_pair))
        print(self.image_seg_pair[:10])

        if isinstance(self.prompt, str) and self.prompt is not None:
            print(f"Using the same prompt for all images: {self.prompt}")
            self.input_ids = self.tokenizer(
                self.prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        self.image_transforms = transforms.Compose(
        [
            transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((self.height, self.width)) if center_crop else transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

        self.conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((self.height, self.width)) if center_crop else transforms.RandomCrop(self.seg_size),
            transforms.ToTensor(),
        ]
    )

    def get_class_stacks(self,label_map):
        if isinstance(label_map, str):
            label_map = Image.open(label_map)
        elif isinstance(label_map, np.ndarray):
            label_map = Image.fromarray(label_map)
        elif isinstance(label_map, torch.Tensor):
            label_map = Image.fromarray(label_map.numpy())
        else:
            label_map = label_map
        
        if label_map.mode != 'RGB':
            label_map = label_map.convert('RGB')
        
        pixels = list(label_map.getdata())
        labels_lst = [list(pixel) for pixel in pixels]

        labels = set(tuple(label) for label in labels_lst)
        labels = [list(label) for label in labels]

        if [0,0,0] in labels : labels.remove([0,0,0])

        sentence = [ self.instance_list[self.color_list.index(label)] for label in labels]
        sentence = " ".join(sentence)

        return sentence


    def get_random_crop_rcs(self,label_map, c, rng, crop_thresh=256, max_attempts=10,crop_scale=0.1,essential_class_ratio=0.03):
        try:
            if random.random() > crop_scale:
                if isinstance(label_map, str):
                    label_map = Image.open(label_map).convert('RGB')

                indices = []
                if len(self.essential_classes) > 0:
                    shuffled_classes = self.essential_classes.copy()
                    random.shuffle(shuffled_classes)

                    for c in shuffled_classes:
                        label_map_arr = torch.from_numpy(np.array(label_map)).permute(2, 0, 1) # c h w
                        label_rgb = self.color_list[self.instance_list.index(c)]
                        label_rgb_ = (label_map_arr[0] == label_rgb[0]) & (label_map_arr[1] == label_rgb[1]) & (label_map_arr[2] == label_rgb[2])
                        indices = np.where(label_rgb_)

                        selected_class = c

                        if len(indices[0]) > 0:
                            break
                
                
                if len(indices[0]) == 0:
                    instance_classes =self.instance_list.copy()
                    random.shuffle(instance_classes)

                    for c in instance_classes:
                        label_map_arr = torch.from_numpy(np.array(label_map)).permute(2, 0, 1) # c h w
                        label_rgb = self.color_list[self.instance_list.index(c)]
                        label_rgb_ = (label_map_arr[0] == label_rgb[0]) & (label_map_arr[1] == label_rgb[1]) & (label_map_arr[2] == label_rgb[2])
                        indices = np.where(label_rgb_)

                        if len(indices[0]) > 0:
                            break
                
                if len(indices[0]) <= 0:
                    w, h = label_map.size
                    prompt = self.get_class_stacks(label_map)
                    return {
                    "crop_coords": (0, 0, w, h),
                    "crop_condition_img": label_map, ## PIL image 9mode :RGB)
                    "crop_texts": prompt,
                    "selected_class": "no_crop",
                }  

                
                w, h = label_map.size

                w_crop_size = rng.integers( min(crop_thresh,int(w//2)) , w )

                if crop_thresh >= w : 
                    raise ValueError("cropping threshold (minimum crop size) must be smaller than image width")
                else:
                    step_size = (w - crop_thresh) // max_attempts

                while w_crop_size >= crop_thresh:
                    w_crop_size = rng.integers( min(crop_thresh,int(w//3)) , int(w//2) )
                    h_crop_size = w_crop_size * h // w

                    if h_crop_size <= h:
                        break
                    else:
                        h_crop_size = h
                        
                    w_crop_size -= step_size
                
                for _ in range(max_attempts): 
                    # idx = np.random.randint(0, len(indices[0]) - 1)
                    idx = rng.integers(0, len(indices[0]) - 1)
                    y, x = indices[0][idx], indices[1][idx]
                    x1 = min(max(0, x - w_crop_size // 2), w -  w_crop_size)
                    y1 = min(max(0, y - h_crop_size // 2), h - h_crop_size)
                    x2 = x1 + w_crop_size
                    y2 = y1 + h_crop_size

                    # rare classes
                    label_map_arr_cropped = Image.fromarray(label_map_arr.permute(1,2,0).numpy()).crop((x1,y1,x2,y2))
                    new_condition_img = label_map_arr_cropped.convert('RGB')

                    #label_map_arr_cropped.save("label_map_arr_cropped.png")
                    label_map_arr_cropped = torch.from_numpy(np.array(label_map_arr_cropped.convert('RGB'))).permute(2,0,1)
                    label_rgb_cropped = (label_map_arr_cropped[0] == label_rgb[0]) & (label_map_arr_cropped[1] == label_rgb[1]) & (label_map_arr_cropped[2] == label_rgb[2])
                    
                    if np.sum(label_rgb_cropped.numpy()) > essential_class_ratio * w_crop_size * h_crop_size:
                        break
                #new_condition_img_pil = Image.fromarray(new_condition_img.permute(1,2,0).numpy()).convert('RGB')
                new_texts = self.get_class_stacks(new_condition_img)

                crop_results = {
                    "crop_coords": (x1, y1, x2, y2),
                    "crop_condition_img": new_condition_img, ## PIL image 9mode :RGB)
                    "crop_texts": new_texts,
                    "selected_class": selected_class,
                }   
            else:
                w, h = label_map.size
                prompt = self.get_class_stacks(label_map)
                crop_results = {
                    "crop_coords": (0, 0, w, h),
                    "crop_condition_img": label_map, ## PIL image 9mode :RGB)
                    "crop_texts": prompt,
                    "selected_class": "no_crop",
                } 
                
            return crop_results
        except:
            print("Error in get_random_crop_rcs")
            w, h = label_map.size
            prompt = self.get_class_stacks(label_map)
            return {
                "crop_coords": (0, 0, w, h),
                "crop_condition_img": label_map, ## PIL image 9mode :RGB)
                "crop_texts": prompt,
                "selected_class": "no_crop",
            } 

    def __getitem__(self,index):
        example = {}
        image_path, seg_path = self.image_seg_pair[index]
        image = Image.open(image_path)

        orig_img = image
        orig_seg = Image.open(seg_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')
        if not orig_seg.mode == 'RGB':
            orig_seg = orig_seg.convert('RGB')

        c = random.choice(self.essential_classes)
        gen_seed = 0
        rng = np.random.default_rng(gen_seed)
        crop_results = self.get_random_crop_rcs(label_map=orig_seg, c=c, rng=rng, crop_thresh=512, max_attempts=10,crop_scale=self.crop_scale,essential_class_ratio=self.essential_class_ratio)

        crop_coords = crop_results["crop_coords"]
        crop_image = image.crop(crop_coords)
        crop_condition_image = crop_results["crop_condition_img"]
        instances_prompt = crop_results["crop_texts"]
        selected_class = crop_results["selected_class"]
    
        image = self.image_transforms(crop_image)
        seg = self.conditioning_image_transforms(crop_condition_image)

        if hasattr(self, "input_ids"):
            prompt = self.prompt + " with " + instances_prompt
            #print(f"Current prompt : {prompt}")
            example["input_ids"] = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            #example["input_ids"] = self.input_ids
        else:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            prompt = self.prompt[image_name] + " with " + instances_prompt
            #print(f"Current prompt : {prompt}")
            example["input_ids"] = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        if random.random() < 0.1:
            w, h = crop_image.size

            orig_img_resized = orig_img.resize((w, h))
            orig_seg_resized = orig_seg.resize((w, h))

            canvas_width = w * 4
            canvas_height = h
            canvas = Image.new('RGB', (canvas_width, canvas_height))

            canvas.paste(orig_img_resized, (0, 0))  
            canvas.paste(orig_seg_resized, (w, 0))  
            canvas.paste(crop_image, (w * 2, 0))    
            canvas.paste(crop_condition_image, (w * 3, 0))  

            output_dir = f'{self.crop_output_dir}/{selected_class}'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            canvas.save(f'{output_dir}/{crop_coords}_{instances_prompt}_{index}.png', format='PNG')

        example["pixel_values"] = image
        example["conditioning_pixel_values"] = seg

        return example
    

    def __len__(self):
        return len(self.image_seg_pair)