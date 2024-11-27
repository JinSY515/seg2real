from PIL import Image, ImageDraw
import math
from itertools import product
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
from tqdm import tqdm
ref_seg_img_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/10k/sem_seg/colormaps/val/a760566c-e45f7339.png"
tgt_seg_img_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/10k/sem_seg/colormaps/val/7dc08598-f42e2015.png"#"/mnt/data4/siyoon/bdd100k/bdd100k/labels/10k/sem_seg/colormaps/val/a8923b1f-00000000.png"
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

os.makedirs("4d_vis_star", exist_ok=True)
os.makedirs("4d_vis_seg_star", exist_ok=True)

# 각 꼭지점 좌표 계산 함수
def calculate_star_points(center, radius):
    points = []
    angle = math.pi / 5  # 별의 각 꼭지점 사이 각도 (5각형 별)
    for i in range(10):
        r = radius if i % 2 == 0 else radius / 2  # 큰 꼭지점과 작은 꼭지점의 반지름
        x = center[0] + r * math.cos(i * angle)
        y = center[1] + r * math.sin(i * angle)
        points.append((x, y))
    return points

# Example loop with star drawing
for h, w in tqdm(product(range(64, 192), range(0, 192))):
    # 이미지 저장 (기존 코드)
    cv2.imwrite(f"4d_vis_star/{(h, w)}.png", seg_corr[0, h, w, :, :].unsqueeze(-1).float().cpu().numpy() * 255.)
    
    # 별 그리기
    point = (w, h)
    radius = 6
    draw = ImageDraw.Draw(tgt_seg)
    star_points = calculate_star_points(point, radius)
    draw.polygon(star_points, fill=(255, 255, 0))  # 별을 노란색으로 채움

    # 이미지 저장 (기존 코드)
    save_img = Image.new('RGB', (256, 256))
    save_img.paste(tgt_seg)
    tgt_seg.save(f"4d_vis_seg_star/{(h, w)}.png")
    tgt_seg = transforms.ToPILImage()(tgt_seg_img_resized.squeeze(0))
