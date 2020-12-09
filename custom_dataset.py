import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
#from PIL import Image
from skimage import io

#listevalpartition.csv: Recommended partitioning of images into training, validation, testing sets. 
#Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing

class Binary_Dataset(Dataset):
    def __init__(self, labels_dir, label, image_dir , partition, transform=None):
    #labels_dir: is directory to the csv file with labels probably 'Data/list_attr_celeba.csv'
    #label: which label in the attribute (list_attr_celeba.csv) csv to use 
    #image_dir: directory to images probably 'Data/img_align_celeba/img_align_celeba'
    #pass label as a string
    #pass composed transform in transform
        self.labels_dir=labels_dir
        self.label=label
        self.image_dir=image_dir
        self.transformers=transform #transforms
        self.partition = partition
        
        
        if partition == "Train":
            self.df_attr=pd.read_csv(labels_dir, nrows = 162770) 
        if partition == "Val":
            self.df_attr=pd.read_csv(labels_dir, skiprows = [i for i in range(1,162771)], nrows = 19867)
        if partition == "Test":
            self.df_attr=pd.read_csv(labels_dir, skiprows = [i for i in range(1,182638)], nrows = 19962)
        
        self.df=self.df_attr[self.label] #dataframe of the chosen label
        self.img_names=self.df_attr["image_id"] #dataframe of image ids

        #change -1 to 0 in csv, remove this if not needed
        #########################
        self.df.replace(-1,0,inplace=True)
        #########################

    def __len__(self):
        #return the length of the column of df_attr
        #####################################################
        size = self.df.shape[0]  #100000 #size above about 50,000 has issues no clue why
        #####################################################
        return size
    
    def __getitem__(self, idx):
        #load image
        image_path=os.path.join(self.image_dir,self.img_names[idx])
        image = io.imread(image_path)
        #print(self.size)

        print(self.df_attr.info())
        

        tensor_image=self.transformers(image)

        #set label
        label=self.df[idx]
        label=torch.tensor(int(label))

        return (tensor_image, label)


# class Train_Binary_Dataset(Dataset):
#     def __init__(self, labels_dir, label, image_dir ,transform=None):
#     #labels_dir: is directory to the csv file with labels probably 'Data/list_attr_celeba.csv'
#     #label: which label in the attribute (list_attr_celeba.csv) csv to use 
#     #image_dir: directory to images probably 'Data/img_align_celeba/img_align_celeba'
#     #pass label as a string
#     #pass composed transform in transform
#         self.labels_dir=labels_dir
#         self.label=label
#         self.image_dir=image_dir
#         self.transformers=transform #transforms
#         self.df_attr=pd.read_csv(labels_dir, nrows = 162770)
        
#         self.df=self.df_attr[self.label] #dataframe of the chosen label
#         self.img_names=self.df_attr["image_id"] #dataframe of image ids

#         #change -1 to 0 in csv, remove this if not needed
#         #########################
#         self.df.replace(-1,0,inplace=True)
#         #########################

#     def __len__(self):
#         #return the length of the column of df_attr
#         #####################################################
#         size = self.df.shape[0]  #100000 #size above about 50,000 has issues no clue why
#         #####################################################
#         return size
    
#     def __getitem__(self, idx):
#         #load image
#         image_path=os.path.join(self.image_dir,self.img_names[idx])
#         image = io.imread(image_path)
#         #print(self.size)

#         print(self.df_attr.info())
        

#         tensor_image=image

#         #set label
#         label=self.df[idx]
#         label=torch.tensor(int(label))

#         return (tensor_image, label)

# class Val_Binary_Dataset(Dataset):
#     def __init__(self, labels_dir, label, image_dir ,transform=None):
#     #labels_dir: is directory to the csv file with labels probably 'Data/list_attr_celeba.csv'
#     #label: which label in the attribute (list_attr_celeba.csv) csv to use 
#     #image_dir: directory to images probably 'Data/img_align_celeba/img_align_celeba'
#     #pass label as a string
#     #pass composed transform in transform
#         self.labels_dir=labels_dir
#         self.label=label
#         self.image_dir=image_dir
#         self.transformers=transform #transforms
#         self.df_attr=pd.read_csv(labels_dir, skiprows = [i for i in range(1,162771)], nrows = 19867)
        
#         self.df=self.df_attr[self.label] #dataframe of the chosen label
#         self.img_names=self.df_attr["image_id"] #dataframe of image ids

#         #change -1 to 0 in csv, remove this if not needed
#         #########################
#         self.df.replace(-1,0,inplace=True)
#         #########################

#     def __len__(self):
#         #return the length of the column of df_attr
#         #####################################################
#         size = self.df.shape[0]  #100000 #size above about 50,000 has issues no clue why
#         #####################################################
#         return size
    
#     def __getitem__(self, idx):
#         #load image
#         image_path=os.path.join(self.image_dir,self.img_names[idx])
#         image = io.imread(image_path)
#         #print(self.size)

#         print(self.df_attr.info())
        

#         tensor_image=image

#         #set label
#         label=self.df[idx]
#         label=torch.tensor(int(label))

#         return (tensor_image, label)



# class Test_Binary_Dataset(Dataset):
#     def __init__(self, labels_dir, label, image_dir ,transform=None):
#     #labels_dir: is directory to the csv file with labels probably 'Data/list_attr_celeba.csv'
#     #label: which label in the attribute (list_attr_celeba.csv) csv to use 
#     #image_dir: directory to images probably 'Data/img_align_celeba/img_align_celeba'
#     #pass label as a string
#     #pass composed transform in transform
#         self.labels_dir=labels_dir
#         self.label=label
#         self.image_dir=image_dir
#         self.transformers=transform #transforms
#         self.df_attr=pd.read_csv(labels_dir, skiprows = [i for i in range(1,182638)], nrows = 19962)
        
#         self.df=self.df_attr[self.label] #dataframe of the chosen label
#         self.img_names=self.df_attr["image_id"] #dataframe of image ids

#         #change -1 to 0 in csv, remove this if not needed
#         #########################
#         self.df.replace(-1,0,inplace=True)
#         #########################

#     def __len__(self):
#         #return the length of the column of df_attr
#         #####################################################
#         size = self.df.shape[0]  #100000 #size above about 50,000 has issues no clue why
#         #####################################################
#         return size
    
#     def __getitem__(self, idx):
#         #load image
#         image_path=os.path.join(self.image_dir,self.img_names[idx])
#         image = io.imread(image_path)
#         #print(self.size)

#         print(self.df_attr.info())
        

#         tensor_image=image

#         #set label
#         label=self.df[idx]
#         label=torch.tensor(int(label))

#         return (tensor_image, label)