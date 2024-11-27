import os 
import cv2 
from torch.nn import functional as F 
import torch 
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from itertools import product
tgt_seg_img_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/10k/sem_seg/colormaps/val/a760566c-e45f7339.png"
ref_seg_img_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/10k/sem_seg/colormaps/val/7dc08598-f42e2015.png"#"/mnt/data4/siyoon/bdd100k/bdd100k/labels/10k/sem_seg/colormaps/val/a8923b1f-00000000.png"
tgt_seg_img = Image.open(tgt_seg_img_path)
ref_seg_img = Image.open(ref_seg_img_path)
ref_seg_img = ref_seg_img.convert("RGB")
tgt_seg_img = tgt_seg_img.convert("RGB")
ref_seg_img = transforms.Compose([transforms.ToTensor()])(ref_seg_img)
tgt_seg_img = transforms.Compose([transforms.ToTensor()])(tgt_seg_img)
tgt_seg_img_resized = F.interpolate(tgt_seg_img.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)
ref_seg_img_resized = F.interpolate(ref_seg_img.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False)

seg_corr = []
seg_corr_rev = []
tgt_seg = transforms.ToPILImage()(tgt_seg_img_resized.squeeze(0))
# import pdb; pdb.set_trace()

for b in range(1):
    seg_corr_ = (tgt_seg_img_resized[b, :, :, :, None, None] == ref_seg_img_resized[b, :, None, None, :, :]).all(dim=0)
    seg_corr.append(seg_corr_)
seg_corr = torch.stack(seg_corr)

os.makedirs("4d_vis", exist_ok=True)
os.makedirs("4d_vis_seg", exist_ok=True)
for h, w in product(range(64, 192), range(0, 192)):
    cv2.imwrite(f"4d_vis/{(h, w)}.png", seg_corr[0, h, w, :, :].unsqueeze(-1).float().cpu().numpy() * 255. )
    point = (w, h)
    radius=4
    draw = ImageDraw.Draw(tgt_seg)
    draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] +radius ), fill=(255, 255, 0))
    save_img = Image.new('RGB', (256, 256))
    # import pdb; pdb.set_trace()
    save_img.paste(tgt_seg)
    tgt_seg.save(f"4d_vis_seg/{(h, w)}.png")
    tgt_seg = transforms.ToPILImage()(tgt_seg_img_resized.squeeze(0))