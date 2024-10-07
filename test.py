
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np

from torch.utils.data import DataLoader

from tools.tester import Tester
from tools.dataset import Dataset


with open('config.json', 'r') as f:
    config = json.load(f)

np.random.seed(config['General']['seed'])
tester = Tester(config)

test_data = Dataset(config, 'test', config['CLI']['path'])
print(f"Testing with the path {config['CLI']['path']}")

test_dataloader = DataLoader(test_data,
                             batch_size=config['General']['batch_size'],
                             shuffle=False,
                             pin_memory=True,
                             drop_last=True)

tester.test_clft(test_dataloader, config['CLI']['mode'])
print('Testing is completed')
