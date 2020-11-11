import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class Binary_Dataset(Dataset):
    def __init__(self, labels_dir, label,image_dir ,transform=None):
    #labels_dir: is directory to the csv file with labels probably 'Data/list_attr_celeba.csv'
    #label: which label in the attribute (list_attr_celeba.csv) csv to use 
    #image_dir: directory to images probably 'Data/img_align_celeba/img_align_celeba'
    #pass label as a string
    #pass composed transform in transform
        self.labels_dir=labels_dir
        self.label=label
        self.image_dir=image_dir
        self.transform=transform
        self.df_attr=pd.read_csv(labels_dir)
        self.df=self.df_attr[label]
        self.img_names=self.df_attr["image_id"]

        #change -1 to 0 in csv, remove this if not needed
        #########################
        self.df.replace(-1,0,inplace=True)
        #########################

    def __len__(self):
        #return the length of the column of df_attr
        size=self.df_attr.shape[0]
        return size
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #return label and image as tensors
        #print(self.img_names)
        
        #load images
        img_name=self.img_names[idx]
        print(self.image_dir+"/"+img_name)
        image=Image.open(self.image_dir+"/"+img_name) #something wrong here when shuffled
        
        #apply transforms to images and make sure you transform them to a tensor
        sample=self.transform(image)

        #load label/label for that image
        target=self.df[idx]

        return sample, target