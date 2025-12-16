import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# from torchsummary import summary

from PIL import Image
import os
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import glob
from transformers import AutoFeatureExtractor, AutoModel
from multiprocessing import Pool

model_map= {"vit":{"tokenizer_path":"./GoogleViT/","model_path":"./GoogleViT/"},
            "dit":{"tokenizer_path":"microsoft/dit-base-finetuned-rvlcdip","model_path":"microsoft/dit-base-finetuned-rvlcdip"}}

# feature_extractor = AutoFeatureExtractor.from_pretrained("./GoogleViT/")
# model =AutoModel.from_pretrained("./GoogleViT/")

# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
# model =AutoModel.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")

model_name ="vit" # dit

feature_extractor = AutoFeatureExtractor.from_pretrained(model_map[model_name]['tokenizer_path'])
model =AutoModel.from_pretrained(model_map[model_name]['tokenizer_path'])

model.eval()

# data loading
npz_files = glob.glob('./BP-file-np/*.npz')
data = {}
# for file in tqdm(npz_files):
#     try:
#         filename = os.path.basename(file)
#         data[filename] = np.load(file)['arr_0']
#     except:
#         continue


data = {}
# for file in tqdm(npz_files):
#     filename = os.path.basename(file)
#     data[filename] = np.load(file,allow_pickle=True)['arr_0']
def load_npz(file):
    filename = os.path.basename(file)
    return filename, np.load(file, allow_pickle=True)['arr_0']

# from multiprocessing import Pool
def update_data(result):
    filename, arr = result
    data[filename] = arr
with Pool(processes=6) as pool:
    results = []
    for file in npz_files:
        results.append(pool.apply_async(load_npz, (file,)))
    
    for result in tqdm(results):
        update_data(result.get())

print('loading npz data over!!!')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model.to(device)
def get_visual_features(visual_info, vision_model):
  
    visual_features = []
    
    for img in visual_info:
        inputs = feature_extractor(img, return_tensors="pt")
        # img = transform(img)

        features = vision_model(**inputs.to(device))
        # for a bp, every page will be transformed to an embedding for the last hidden layer (average pooling)
        visual_features.append(features.last_hidden_state.mean(axis=1).squeeze().detach().cpu().numpy())
        
    return visual_features

visual_dict = {}
error_bp = []
for key in tqdm(data.keys()):
    try:
        visual_dict[key] = get_visual_features(data[key],model)
    except:
        print(key)
        error_bp.append([key])
np.savez("./result/"+model_name+"_visual_features.npz", **visual_dict)

pd.DataFrame(error_bp).to_csv('./result/'+model_name+'_error_bp.csv',index=False)