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
import torch.nn.functional as F
from sklearn.decomposition import PCA


def load_image(img: str) -> torch.Tensor:
    transform_image = T.Compose([T.ToTensor(), T.Resize(224), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img

def load_seg(img_name: str):
    transform_image = T.Compose([T.ToTensor(), T.Resize((256,512))])#,  T.Normalize([0.5], [0.5])])
    seg_name = img_name.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit", "_gtFine_color")
    seg = Image.open(seg_name)
    transform_seg= transform_image(seg)[:3].unsqueeze(0)
    return transform_seg
# def create_index(files: list):
#     """
#     Create an index that contains all of the images in the specified list of files.
#     """
#     index = faiss.IndexFlatL2(384)
#     all_embeddings = {}
#     device=torch.device("cuda:0")
#     dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#     dinov2_vits14.to(device)
#     with torch.no_grad():
#       for i, file in enumerate(tqdm(files)):
#         embeddings = dinov2_vits14(load_image(file).to(device))
#         embedding = embeddings[0].cpu().numpy()
#         all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()
#         import pdb; pdb.set_trace()
#         index.add(np.array(embedding).reshape(1, -1))
#     with open("all_embeddings_w_seg.json", "w") as f:
#             f.write(json.dumps(all_embeddings))
#     faiss.write_index(index, "data_w_seg.bin")
    # return index, all_embeddings
def create_index(files: list):
    """
    Create an index that contains all of the images in the specified list of files.
    """
    index = faiss.IndexFlatL2(384)
    all_embeddings = {}
    device=torch.device("cuda:0")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino.to(device)
    with torch.no_grad():
      for i, file in enumerate(tqdm(files)):
        features = dino.forward_features(load_image(file).to(device))['x_norm_patchtokens'].to(device)
        # features [1, 1, 256, 384]

        sampled_Seg = load_seg(file)
        sampled_Seg = F.interpolate(sampled_Seg.detach().cpu(), (256, 384), mode="bilinear", align_corners=False).to(device)
        # sampled_Seg [1, 3, 256, 384]
        denorm = sampled_Seg.sum(dim=(-1, -2), keepdim=True) + 1e-8 
        sampled_Seg = sampled_Seg.to(device) #* 255.0 #.numpy()
        embedding = features * sampled_Seg / denorm

        embedding = embedding.mean(0).mean(0).mean(0).cpu().numpy()
        all_embeddings[file] = np.array(embedding).reshape(1, -1).tolist()
        index.add(np.array(embedding).reshape(1, -1))
    with open("all_embeddings_w_seg_nopca.json", "w") as f:
            f.write(json.dumps(all_embeddings))
    faiss.write_index(index, "data_w_seg_nopca.bin")
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
        indices = search_index(data_index, np.array(embedding[0]).reshape(1, -1), k=6) #
        
        
        return indices #files[indices[1]]
    
def retrieve_with_seg(search_file):
    # dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # dinov2_vits14.to(device)
    transform_image = T.Compose([T.ToTensor(), T.Resize(224), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
    ROOT_DIR = "/media/dataset2/CityScape/leftImg8bit/train"
    files = glob(ROOT_DIR + "**/**/**.png", recursive=True)
    
    def load_seg(img_name: str):
        seg_name = img_name.replace("/leftImg8bit/", "/gtFine/").replace("_leftImg8bit", "_gtFine_color")
        seg = Image.open(seg_name)
        # transform_seg= transform_image(seg)[:3].unsqueeze(0)
        return seg
    
    def load_image(img: str) -> torch.Tensor:
        """
        Load an image and return a tensor that can be used as an input to DINOv2.
        """
        img = Image.open(img)
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    
    def read_index():
        data_index = "/home/cvlab01/project/siyoon/Moore-AnimateAnyone/retrieval/data_w_seg_nopca.bin"
        data_dict = faiss.read_index(data_index)
        
        return data_dict
    def load_json():
        with open("/home/cvlab01/project/siyoon/Moore-AnimateAnyone/retrieval/all_embeddings_w_seg_nopca.json", "r") as fr:
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
        indices = search_index(data_index, np.array(embedding[0]).reshape(1, -1), k=6) #
        
        return indices
    
if __name__ == "__main__":
    # search_file = ""
    # retrieve(search_file)
    # transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
    ROOT_DIR = "/media/dataset2/CityScape/leftImg8bit/train"
    files = glob(ROOT_DIR + "**/**/**.png", recursive=True)
    # filename = "/media/dataset2/CityScape/leftImg8bit/train/jena/jena_000007_000019_leftImg8bit.png"
    # filename = "/media/dataset2/CityScape/leftImg8bit/train/hanover/hanover_000000_001620_leftImg8bit.png"
    # filename = "/media/dataset2/CityScape/leftImg8bit/train/bochum/bochum_000000_000313_leftImg8bit.png"
    # filename = "/media/dataset2/CityScape/leftImg8bit/train/hamburg/hamburg_000000_006322_leftImg8bit.png"
    # filename="/media/dataset2/CityScape/leftImg8bit/train/hamburg/hamburg_000000_011641_leftImg8bit.png"
    # filename = "/media/dataset2/CityScape/leftImg8bit/train/hamburg/hamburg_000000_014030_leftImg8bit.png"
    filename = "/media/dataset2/CityScape/leftImg8bit/train/hamburg/hamburg_000000_019892_leftImg8bit.png"
    r = retrieve(filename)
    rs = retrieve_with_seg(filename)

    print("DINOv2")
    for i in r:
        print(files[i])
    
    print("DINOv2 + seg")
    for i in rs:
        print(files[i])
    # create_index(files)
    # ROOT_DIR = "/media/dataset2/CityScape/leftImg8bit/train"
    # files = glob(ROOT_DIR + "**/**/**.png", recursive=True)
    # retrieved_list = []
    
    # import matplotlib.pyplot as plt 
    # from collections import Counter


    # for file in tqdm(files):
    #     retrieved = retrieve(file)
    #     retrieved_list.append(retrieved)
    
    # counter = Counter(retrieved_list)
    # # most_common = counter.most_common(20)
    # labels, values = zip(*counter.items())    # import pdb; pdb.set_trace()
    # plt.bar(labels, values)
    # plt.xticks(rotation=90)
    # plt.savefig("plt_all.png")