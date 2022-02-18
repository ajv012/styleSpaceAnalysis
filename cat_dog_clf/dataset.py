from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import sys
sys.path.append("./")
sys.path.append("../")

class afhq_dataset(Dataset):
    r"""
    Take a root dir and return the transformed img and associated label with it
    """
    def __init__(self, root_dir, seed, labels, img_transform=None):

        self.seed = seed
        np.random.seed(self.seed)

        # this dir has two sub dirs cat and dog. Need to combine them
        self.root_dir = root_dir
        self.cat_names = os.listdir(os.path.join(self.root_dir, "cat"))
        self.dog_names = os.listdir(os.path.join(self.root_dir, "dog"))
        self.all_names = np.asarray(self.cat_names + self.dog_names)
        np.random.shuffle(self.all_names)
        self.img_transform = img_transform
        self.labels = {}
        for i in range(len(labels)):
            self.labels[labels[i]] = i
        

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        curr_path = os.path.join(self.root_dir, self.all_names[idx].strip().split("_")[1], self.all_names[idx])
        curr_img = Image.open(curr_path)
        curr_label = self.labels[self.all_names[idx].strip().split("_")[1]]
        
        if self.img_transform:
            curr_img_transformed = self.img_transform(curr_img)
        
        return {"inputs" : curr_img_transformed, "labels" : curr_label} 
    
    def viz_img(self, imgs):
        r"""
        Take a tensor or list of tensors and visualize it
        """
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    def what_labels_mean(self):
        return [label + ": " + str(self.labels[label]) for label in self.labels]
        