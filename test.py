
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
import argparse

from torch.utils.data import DataLoader

from tools.tester import Tester
from tools.dataset import Dataset

parser = argparse.ArgumentParser(description='CLFT and CLFCN Testing')
parser.add_argument('-c', '--config', type=str, required=False, default='config.json', help='The path of the config file')
args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as f:
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

tester.test_clft(test_dataloader, config['CLI']['mode'], config['Log']['logdir'])
print('Testing is completed')
