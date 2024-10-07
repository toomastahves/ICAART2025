#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np

from torch.utils.data import DataLoader

from tools.trainer import Trainer
from tools.dataset import Dataset
from utils.helpers import create_config_snapshot

with open('config.json', 'r') as f:
    config = json.load(f)

np.random.seed(config['General']['seed'])
trainer = Trainer(config)

create_config_snapshot()

train_data = Dataset(config, 'train', './waymo_dataset/splits_clft/train_all.txt')
train_dataloader = DataLoader(train_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

valid_data = Dataset(config, 'val', './waymo_dataset/splits_clft/early_stop_valid.txt')
valid_dataloader = DataLoader(valid_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

trainer.train_clft(train_dataloader, valid_dataloader, modal=config['CLI']['mode'])
