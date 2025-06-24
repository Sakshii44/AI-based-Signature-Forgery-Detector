import os
import random
from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms

class SiameseSignatureDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.people = os.listdir(root_dir)
        self.image_paths = {}
        for person in self.people:
            person_path = os.path.join(root_dir,person)
            self.image_paths[person]={
                "genuine" : os.listdir(os.path.join(person_path,"genuine")),
                "forged" : os.listdir(os.path.join(person_path,"forged")),
            }
    
    def __len__(self):
        return 10000
    
    def __getitem__(self,idx):
        person = random.choice(self.people)
        genuine_images = self.image_paths[person]["genuine"]
        forged_images = self.image_paths[person]["forged"]

        if random.random()<0.5:
            img1_name,img2_name = random.sample(genuine_images,2)
            label = 1
            img1_path = os.path.join(self.root_dir,person,"genuine",img1_name)
            img2_path = os.path.join(self.root_dir,person,"genuine",img2_name)

        else:
            img1_name = random.choice(genuine_images)
            img2_name = random.choice(forged_images)
            label = 0
            img1_path = os.path.join(self.root_dir,person,"genuine",img1_name)
            img2_path = os.path.join(self.root_dir,person,"forged",img2_name)

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1,img2,label
    
