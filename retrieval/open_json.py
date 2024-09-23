import json 
import faiss
# json_file = "./all_embeddings.json"
# with open(json_file, "r") as f:
#     data = json.load(f)
# import pdb; pdb.set_trace()
# print(data)

bin_file = "./data.bin"
data = faiss.read_index(bin_file)
import pdb; pdb.set_trace()
