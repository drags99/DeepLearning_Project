from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class Celeb_Dataset(Dataset):
    def __init__(self, csv_file, labels_dir, transform=None):
    #labels_file is to the list_attr_celeba.csv file
    #root_dir is to the folder with images \img_align_celeba\img_align_celeba
    #transforms to be used

    self.labels=pd.read_csv(labels_dir)
    self.transforms=transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name= os.path.join(self.root_dir, self.labels.iloc[idx,0])
        image = io.imread(img_name)