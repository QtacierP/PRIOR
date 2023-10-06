import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import json
from PIL import Image
from torchvision import transforms

class ZeroShotClassificationDataset(Dataset):
    def __init__(self, dataset_path, rate=1.0, transform=[]) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform.copy()
        self.rate = rate
        self._load_dataset()
        self._load_statics()
        self._build_transform()
        

    def _load_statics(self):
        self.mean = [0.4755]
        self.std = [0.3011]
        self.weights = None

    def _load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            self.data_json = json.load(f)
        self.id_list = list(self.data_json.keys())
        if self.rate < 1:
            self.id_list = self.id_list[: int(self.rate * len(self.id_list))]
        
    def _build_transform(self):
        self.transform += [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        self.transform = transforms.Compose(self.transform)

    def _get_image(self, index):
        return self.transform(Image.open(self.data_json[self.id_list[index]]['image_path']))

    def _get_label(self, index):
        return self.data_json[self.id_list[index]]['label']

    def __getitem__(self, index):
        image = self._get_image(index)
        label = self._get_label(index)
        return {'image': image, 
                'label': label}
    
    def __len__(self):
        return len(self.id_list)