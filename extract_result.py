import os 
import cv2
from glob import glob
from tqdm import tqdm
data_root = "/home/cvlab15/project/siyoon/seg2real24/1121_abl/sa_tuned/sa_aug"
data_list = glob(f"{data_root}/*.png")
save_root = data_root.replace("/1121_abl/", "/1121_abl_only/")
os.makedirs(save_root, exist_ok=True)
for img_path in tqdm(data_list):
    img = cv2.imread(img_path)
    save_name = img_path.replace("/1121_abl/", "/1121_abl_only/")
    cv2.imwrite(save_name, img[:, 1024:, :])
    # import pdb; pdb.set_trace()