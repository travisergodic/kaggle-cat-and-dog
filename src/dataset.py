import os

from PIL import Image
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    def __init__(self, image_path_list, label_list, num_classes, image_transform=None):
        super().__init__()
        self.image_path_list = image_path_list
        self.label_list = label_list
        self.num_classes = num_classes
        self.image_transform = image_transform 
    
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label = self.label_list[index]
 
        data = Image.open(image_path).convert('RGB')
        if self.image_transform is not None:
            data = self.image_transform(data)
        return {"data": data, "label": label, "name": os.path.basename(image_path)}

    def __len__(self):
        return len(self.image_path_list)
