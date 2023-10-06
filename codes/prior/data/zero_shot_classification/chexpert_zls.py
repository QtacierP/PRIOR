from re import L
import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from glob import glob
import json
import sys
sys.path.append(os.getcwd())
from PIL import Image
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from prior.data.zero_shot_classification.base import ZeroShotClassificationDataset
from tqdm import tqdm
import random
from transformers import AutoTokenizer
from tqdm import tqdm

CHEXPERT_COMPETITION_TASKS =  [
"Atelectasis",
"Cardiomegaly",
"Consolidation",
"Edema",
"Pleural Effusion",
]

class CheXPertZeroClsDataset(ZeroShotClassificationDataset):
    def __init__(self, root_path, dataset_path, rate=1.0, transform=[], num_colors=1) -> None:
        self.num_colors = num_colors
        self.root_path = root_path
        super().__init__(dataset_path=dataset_path, rate=rate, transform=transform)
        if num_colors == 1:
            self.image_io = lambda x: Image.open(x).convert('L')
        else:
            self.image_io = lambda x: Image.open(x).convert('RGB')
        self.binary = True
        self.roi_task_list =  ["Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion"]
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.max_length = 256
        self._init_prompt()
    
    def _load_statics(self):
        if self.num_colors == 1:
            self.mean = [0.5062731200407803]
            self.std = [0.28946775394688107]
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

    def _init_prompt(self):
        prompt = simple_generate_chexpert_class_prompts()
        prompt_list = []
        for i, task in enumerate(self.roi_task_list):
            prompt_list.append([])
            task_prompt_list = prompt[task]
            for task_prompt in task_prompt_list:
                tokens = self.tokenizer(
                        task_prompt,
                        max_length=self.max_length,
                        add_special_tokens=True,
                        padding='max_length',
                        truncation=False,
                        return_token_type_ids=False,
                        return_attention_mask=True,
                        return_tensors='pt'
                    )
                prompt_list[i].append(tokens)
        self.prompt_tensor = prompt_list
                
                
    def _get_label(self, index):
        return np.asarray(self.data_json[self.id_list[index]]['label'], dtype=float).argmax()

    def _get_image(self, index):
        return self.transform(self.image_io(os.path.join(self.root_path, self.data_json[self.id_list[index]]['image_path'])))

    def _get_prompt(self):
        simple_generate_chexpert_class_prompts(5)
    
    def __getitem__(self, index):
        image = self._get_image(index)
        label = self._get_label(index)

        return {'image': image, 
                'label': label}


def simple_generate_chexpert_class_prompts():
    prompts = {}
    for k in CHEXPERT_COMPETITION_TASKS:
        prompts[k] =[k + ' is observed']
    return prompts



    
    
