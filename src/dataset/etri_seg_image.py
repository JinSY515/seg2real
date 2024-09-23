from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
from transformers import AutoTokenizer, PretrainedConfig
import random

class EtriSegDataset(Dataset):
    def __init__(
        self,
        images_path,
        img_size,
        center_crop,
        text_prompt
    ):
        super().__init__()
        self.img_size = img_size
        self.center_crop = center_crop

        self.images_path = list(glob(f"{images_path}/*_frames/**/", recursive=True))
        self.prompt = text_prompt

        self.image_seg_pair = []
        for i in range(len(self.images_path)):
            try:
                image_path = glob(f"{self.images_path[i]}images/*.jpg")
                seg_path = glob(f"{self.images_path[i]}semantic/*.jpg")
                
                assert len(image_path) == len(seg_path)

                for j in range(len(image_path)):
                    self.image_seg_pair.append((image_path[j], seg_path[j]))
            except:
                print(f"Images num and seg num are not same in {self.images_path[i]}")
                continue
                
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
        #self.text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device="cuda")
        
        self.input_ids=self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        self.image_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.img_size) if center_crop else transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

        self.conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.img_size) if center_crop else transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
        ]
    )

    def __getitem__(self,index):
        example = {}
        image_path, seg_path = self.image_seg_pair[index]
        image = Image.open(image_path)
        seg = Image.open(seg_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')
        if not seg.mode == 'RGB':
            seg = seg.convert('RGB')

        image = self.image_transforms(image)
        seg = self.conditioning_image_transforms(seg)

        reference_idx = random.randint(0, len(self.image_seg_pair) - 1)
        while reference_idx == index:
            reference_idx = random.randint(0, len(self.image_seg_pair) - 1)
        
        ref_image_path, _ = self.image_seg_pair[reference_idx]
        ref_image = Image.open(ref_image_path)

        if not ref_image.mode == 'RGB':
            ref_image = ref_image.convert('RGB')
        
        ref_image = self.image_transforms(ref_image)

        example["pixel_values"] = image
        example["seg"] = seg
        example["ref_image_path"] = ref_image
        example["input_ids"] = self.input_ids

        return example
    

    def __len__(self):
        return len(self.image_seg_pair)