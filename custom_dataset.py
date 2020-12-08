import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
#from PIL import Image
from skimage import io

class Binary_Dataset(Dataset):
    def __init__(self, labels_dir, label, image_dir ,transform=None):
    #labels_dir: is directory to the csv file with labels probably 'Data/list_attr_celeba.csv'
    #label: which label in the attribute (list_attr_celeba.csv) csv to use 
    #image_dir: directory to images probably 'Data/img_align_celeba/img_align_celeba'
    #pass label as a string
    #pass composed transform in transform
        self.labels_dir=labels_dir
        self.label=label
        self.image_dir=image_dir
        self.transformers=transform #transforms
        self.df_attr=pd.read_csv(labels_dir)
        self.df=self.df_attr[label] #dataframe of the chosen label
        self.img_names=self.df_attr["image_id"] #dataframe of image ids

        #change -1 to 0 in csv, remove this if not needed
        #########################
        self.df.replace(-1,0,inplace=True)
        #########################

    def __len__(self):
        #return the length of the column of df_attr
        #####################################################
        size = 50000 #self.df.shape[0] #100000 #size above about 50,000 has issues no clue why
        #####################################################
        return size
    
    def __getitem__(self, idx):
        #load image
        image_path=os.path.join(self.image_dir,self.img_names[idx+10])
        image = io.imread(image_path)
        #print(self.size)




        tensor_image=image

        #set label
        label=self.df[idx]
        label=torch.tensor(int(label))

        return (tensor_image, label)