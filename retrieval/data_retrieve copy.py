import numpy as np
import torch 
import torchvision.transforms as T 
from PIL import Image 
import cv2 
import json 
from tqdm import tqdm 
from matplotlib import pyplot as plt 
import os 
from glob import glob
import faiss

# def load_image(img: str) -> torch.Tensor:
#     """
#     Load an image and return a tensor that can be used as an input to DINOv2.
#     """
#     img = Image.open(img)
#     transformed_img = transform_image(img)[:3].unsqueeze(0)
#     return transformed_img

def create_index(files: list):
    """
    Create an index that contains all of the images in the specified list of files.
    """
    index = faiss.IndexFlatL2(384)
    all_embeddings = {}
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        embeddings = dinov2_vits14(load_image(file).to(device))
        embedding = embeddings[0].cpu().numpy()
        all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()
        
        index.add(np.array(embedding).reshape(1, -1))

    with open("all_embeddings.json", "w") as f:
            f.write(json.dumps(all_embeddings))
    faiss.write_index(index, "data.bin")
    return index, all_embeddings

# def read_index():
#     data_index = "./data.bin"
#     data_dict = faiss.read_index(data_index)
    
#     return data_dict
# def search_index(index, embeddings: list, k: int = 4) -> list:
#     """
#     Search the index for the images that are most similar to the provided image.
#     """
#     D, I = index.search(np.array(embeddings[0].reshape(1, -1)),k)
#     return I[0]

# search_file = "/media/dataset1/CityScape/leftImg8bit/train/ulm/ulm_000010_000019_leftImg8bit.png"
# ROOT_DIR = "/media/dataset1/CityScape/leftImg8bit/train"
# files = glob(ROOT_DIR + "**/**/**.png", recursive=True)
# # data_index, all_embeddings = create_index(files)
# data_index = read_index()
# with torch.no_grad():
#     embedding = dinov2_vits14(load_image(search_file).to(device))
#     # import pdb; pdb.set_trace()
#     indices = search_index(data_index, np.array(embedding[0].cpu()).reshape(1, -1), k=4)
#     for i, index in enumerate(indices):
#         print()
#         print(f"Image {i}: {files[index]}")
#         # img = cv2.resize(cv2.imread(files[index]), (416, 416)) 

def retrieve(search_file):
    # dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # dinov2_vits14.to(device)
    transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
    ROOT_DIR = "/media/dataset2/CityScape/leftImg8bit/train"
    files = glob(ROOT_DIR + "**/**/**.png", recursive=True)
    
    def load_image(img: str) -> torch.Tensor:
        """
        Load an image and return a tensor that can be used as an input to DINOv2.
        """
        img = Image.open(img)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    
    def read_index():
        data_index = "/home/cvlab01/project/siyoon/Moore-AnimateAnyone/retrieval/data.bin"
        data_dict = faiss.read_index(data_index)
        
        return data_dict
    def load_json():
        with open("/home/cvlab01/project/siyoon/Moore-AnimateAnyone/retrieval/all_embeddings.json", "r") as fr:
            data = json.load(fr)
        return data
    def search_index(index, embeddings: list, k: int = 4) -> list:
        """
        Search the index for the images that are most similar to the provided image.
        """
        D, I = index.search(np.array(embeddings[0].reshape(1, -1)),k)
        return I[0]
    data_index = read_index()
    data_dict = load_json() 
    
    with torch.no_grad():
        # embedding = dinov2_vits14(load_image(search_file).to(device))
        embedding = data_dict[search_file]
        indices = search_index(data_index, np.array(embedding[0]).reshape(1, -1), k=4)
        return files[indices[1]]
    
if __name__ == "__main__":
    search_file = ""
    retrieve(search_file)