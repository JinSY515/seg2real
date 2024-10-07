import torch
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms

# Cosine similarity를 계산하기 위해 normalization 함수 정의
def normalize(tensor):
    return tensor / tensor.norm(dim=1, keepdim=True)

# Cosine similarity 계산 함수
def cosine_similarity(tensor1, tensor2):
    tensor1_norm = (tensor1)
    tensor2_norm = (tensor2)
    return torch.einsum('bcij, bckl -> cijkl', tensor1_norm, tensor2_norm)

# 이미지 변환을 위한 transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기를 [256, 256]으로 조정
    transforms.ToTensor()  # 이미지를 텐서로 변환 (크기: [3, 256, 256])
])

# 이미지 로드
seg1_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/sem_seg/colormaps/val/7dc08598-f42e2015.png"
seg2_path = "/mnt/data4/siyoon/bdd100k/bdd100k/labels/sem_seg/colormaps/val/7d2f7975-e0c1c5a7.png"
image1 = Image.open(seg1_path).convert('RGB')  # 이미지 경로로 불러옴 (RGB로 변환)
image2 = Image.open(seg2_path).convert('RGB')

# 텐서로 변환
image1_tensor = transform(image1).unsqueeze(0)  # 크기: [1, 3, 256, 256]
image2_tensor = transform(image2).unsqueeze(0)  # 크기: [1, 3, 256, 256]

# Cosine similarity 계산
similarity = cosine_similarity(image1_tensor, image2_tensor).squeeze(0)
import pdb; pdb.set_trace()

# 특정 좌표에 대한 similarity 맵 시각화
h, w = 128, 120
coords = similarity[:, h, w, :, :]

# NaN을 -inf로 대체하여 최대값 계산에 영향을 주지 않도록 함
coords = torch.nan_to_num(coords, nan=-float('inf'))

# coords 값이 가장 큰 경우만 1로 설정하고 나머지는 0으로 설정
# coords_max = torch.max(coords)  # coords의 최대값
# coords = (coords == coords_max).float()  # 최대값인 위치는 1, 나머지는 0으로 설정
# 시각화를 위해 uint8로 변환
coords_np = (coords * 255).byte().cpu().numpy()  # numpy 배열로 변환
coords_np = np.transpose(coords_np, (1,2,0))
import pdb; pdb.set_trace()

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
img_with_point.save(f"coords_similarity_{(h, w)}_after.png")

# coords2 계산 및 저장
coords2 = similarity[:, :, h, w]

# NaN을 -inf로 대체하여 최대값 계산에 영향을 주지 않도록 함
coords2 = torch.nan_to_num(coords2, nan=-float('inf'))

# coords2 값이 가장 큰 경우만 1로 설정하고 나머지는 0으로 설정
# coords2_max = torch.max(coords2)  # coords2의 최대값
# coords2 = (coords2 == coords2_max).float()  # 최대값인 위치는 1, 나머지는 0으로 설정
coords2_np = np.transpose(coords2_np, (1,2,0))
# 시각화를 위해 uint8로 변환
coords2_np = (coords2 * 255).byte().cpu().numpy()  # numpy 배열로 변환

# coords2 이미지를 PIL 이미지로 변환
img2 = Image.fromarray(coords2_np)  # mode='L'은 필요 없음, 자동으로 처리됨

# 투명 이미지 생성 및 점 찍기
transparent_img = Image.new('RGBA', img2.size, (0, 0, 0, 0))  
draw = ImageDraw.Draw(transparent_img)
point = (w, h)
radius = 2

# 점을 찍기 위해 원을 그리기 (점이 있는 작은 원)
draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=(255, 0, 0))

# RGBA 이미지에 원을 그린 투명 이미지를 붙이기
img_with_point2 = img2.convert("RGBA")
img_with_point2.paste(transparent_img, (0, 0), transparent_img)

# 최종 이미지 보기 및 저장
img_with_point2.show()
img_with_point2.save(f"coords2_similarity_{(h, w)}_after.png")
