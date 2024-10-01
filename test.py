import os
import cv2
import torch
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms

def show_mask_on_image(img, mask, save_path):
    img = np.float32(img) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img) 
    cam = cam / np.max(cam) 
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)

seg2_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/sem_seg/colormaps/val/7d2f7975-e0c1c5a7.png"
seg1_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/sem_seg/colormaps/val/7dc08598-f42e2015.png"

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기를 [256, 256]으로 조정
    transforms.ToTensor()  # 이미지를 텐서로 변환 (크기: [3, 256, 256])
])

# 이미지 로드
image1 = Image.open(seg1_path).convert('RGB')  # 이미지 경로로 불러옴 (RGB로 변환)
image2 = Image.open(seg2_path).convert('RGB')

# 텐서로 변환
image1_tensor = transform(image1).unsqueeze(0)  # 크기: [1, 3, 256, 256]
image2_tensor = transform(image2).unsqueeze(0)  # 크기: [1, 3, 256, 256]

# Outer product 계산
outer_product = torch.einsum('bcij, bckl -> bijkl', image1_tensor, image2_tensor).squeeze(0)

h, w = 128, 200
coords = outer_product[h, w, :, :]
coords = (coords - coords.min()) / (coords.max() - coords.min())
coords = (coords * 255).byte()  # uint8로 변환
coords_np = coords.squeeze().cpu().numpy()  # numpy 배열로 변환

# coords 이미지를 PIL 이미지로 변환
img = Image.fromarray(coords_np)  # mode='L'은 필요 없음, 자동으로 처리됨

# 투명 이미지 생성 및 점 찍기
transparent_img = Image.new('RGBA', img.size, (0, 0, 0, 0))  
draw = ImageDraw.Draw(transparent_img)
point = (w, h)
radius = 2

# 점을 찍기 위해 원을 그리기 (점이 있는 작은 원)
draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(255, 0, 0))

# RGBA 이미지에 원을 그린 투명 이미지를 붙이기
img_with_point = img.convert("RGBA")
img_with_point.paste(transparent_img, (0, 0), transparent_img)

# 최종 이미지 보기 및 저장
img_with_point.show()
img_with_point.save(f"coords_{(h, w)}.png")

# coords2 계산 및 저장
coords2 = outer_product[:, :, h, w]
coords2 = (coords2 - coords2.min()) / (coords2.max() - coords2.min())
coords2 = (coords2 * 255).byte()  # uint8로 변환
coords2_np = coords2.squeeze().cpu().numpy()
# coords 이미지를 PIL 이미지로 변환
img = Image.fromarray(coords2_np)  # mode='L'은 필요 없음, 자동으로 처리됨

# 투명 이미지 생성 및 점 찍기
transparent_img = Image.new('RGBA', img.size, (0, 0, 0, 0))  
draw = ImageDraw.Draw(transparent_img)
point = (w, h)
radius = 2

# 점을 찍기 위해 원을 그리기 (점이 있는 작은 원)
draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(255, 0, 0))

# RGBA 이미지에 원을 그린 투명 이미지를 붙이기
img_with_point = img.convert("RGBA")
img_with_point.paste(transparent_img, (0, 0), transparent_img)

# 최종 이미지 보기 및 저장
img_with_point.show()
img_with_point.save(f"coords2_{(h, w)}.png")
