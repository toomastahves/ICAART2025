# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from clfcn.fusion_net import FusionNet
from utils.metrics import find_overlap
from utils.metrics import find_overlap_1
from utils.metrics import auc_ap
from clft.clft import CLFT


class Tester(object):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if config['CLI']['backbone'] == 'clfcn':
            self.model = FusionNet()
            print(f"Using backbone {config['CLI']['backbone']}")
            self.optimizer_fcn = torch.optim.Adam(self.model.parameters(), lr=config['CLFCN']['clfcn_lr'])
            self.scheduler_fcn = ReduceLROnPlateau(self.optimizer_fcn)

        elif config['CLI']['backbone'] == 'clft':
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              read=config['CLFT']['read'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=len(config['Dataset']['classes']),
                              type=config['CLFT']['type'],
                              model_timm=config['CLFT']['model_timm'], )
            print(f"Using backbone {config['CLI']['backbone']}")

            model_path = config['General']['model_path']
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)
        self.nclasses = len(config['Dataset']['classes'])
        self.model.eval()

    def test_clft(self, test_dataloader, modal):
        print('Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            background_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            background_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            cyclist_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            cyclist_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            pedestrian_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            pedestrian_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            sign_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            sign_rec = torch.zeros((len(progress_bar)), dtype=torch.float)

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                _, output_seg = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap_1(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                cyclist_pre[i] = batch_precision[0]
                cyclist_rec[i] = batch_recall[0]
                pedestrian_pre[i] = batch_precision[1]
                pedestrian_rec[i] = batch_recall[1]
                sign_pre[i] = batch_precision[2]
                sign_rec[i] = batch_recall[2]

                progress_bar.set_description(f'CYCLIST:IoU->{batch_IoU[0]:.4f} '
                                             f'PEDESTRIAN:IoU->{batch_IoU[1]:.4f} '
                                             f'SIGN:IoU->{batch_IoU[2]:.4f} ')

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            cyclist_AP = auc_ap(cyclist_pre, cyclist_rec)
            pedestrian_AP = auc_ap(pedestrian_pre, pedestrian_rec)
            sign_AP = auc_ap(sign_pre, sign_rec)
            print('-----------------------------------------')
            print(f'CYCLIST:CUM_IoU->{cum_IoU[0]:.4f} '
                  f'CUM_Precision->{cum_precision[0]:.4f} '
                  f'CUM_Recall->{cum_recall[0]:.4f} '
                  f'Average Precision->{cyclist_AP:.4f} \n')
            print(f'PEDESTRIAN:CUM_IoU->{cum_IoU[1]:.4f} '
                  f'CUM_Precision->{cum_precision[1]:.4f} '
                  f'CUM_Recall->{cum_recall[1]:.4f} '
                  f'Average Precision->{pedestrian_AP:.4f} ')
            print(f'SIGN:CUM_IoU->{cum_IoU[2]:.4f} '
                  f'CUM_Precision->{cum_precision[2]:.4f} '
                  f'CUM_Recall->{cum_recall[2]:.4f} '
                  f'Average Precision->{sign_AP:.4f} ')
            print('-----------------------------------------')
            print('Testing of the subset completed')
