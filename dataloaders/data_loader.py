import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
import json
import os
from typing import List, Tuple
import datasets


class data_set(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):
        label = torch.tensor([item[0]["relation"] for item in data])
        tokens = [torch.tensor(item[0]["tokens"]) for item in data]
        attention_mask = [torch.tensor(item[0]["attention_mask"]) for item in data]
        text = [item[0]["tokens"] for item in data]
        ind = [item[1] for item in data]

        
        if 'task' in data[0][0].keys():
            task = [item[0]["task"] for item in data]
            return (tokens, attention_mask, label, task)
        
        return (tokens, attention_mask, label)


def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None):
    
    dataset = data_set(data, config)
    if batch_size == None:
        batch_size = config.batch_size
    batch_size = min(batch_size, len(data))
    
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=True, 
        num_workers=config.num_workers, 
        collate_fn=dataset.collate_fn, 
        drop_last=drop_last
    )
    


