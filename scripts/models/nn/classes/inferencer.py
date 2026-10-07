#!/usr/bin/env python

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Inferencer:

    def __init__(self,model,dataloader,device):
        '''
        Purpose: Initialize Inferencer for running a trained model over a dataloader.
        Args:
        - model (torch.nn.Module): trained model
        - dataloader (torch.utils.data.DataLoader): dataloader of samples to predict
        - device (str): 'cuda' | 'cpu'
        '''
        self.model      = model
        self.dataloader = dataloader
        self.device     = device

    def predict(self,haskernel):
        '''
        Purpose: Predict every sample in the dataloader, and for kernel models also return the kernel-integrated features.
        Args:
        - haskernel (bool): whether the model has an integration kernel
        Returns:
        - tuple[np.ndarray, np.ndarray | None]: standardized predictions with shape (nsamples,), and kernel-integrated
            features with shape (nsamples, nfieldvars) or None
        '''
        self.model.eval()
        predslist = []
        featslist = [] if haskernel else None
        with torch.no_grad():
            for batch in self.dataloader:
                fields = batch['fields'].to(self.device,non_blocking=True)
                local  = batch['local'].to(self.device,non_blocking=True)
                if haskernel:
                    dsig   = batch['dsig'][0].to(self.device,non_blocking=True)
                    output = self.model(fields,dsig,local)
                    featslist.append(self.model.kernel.features.detach().cpu().numpy())
                else:
                    output = self.model(fields,local)
                predslist.append(output.detach().cpu().numpy())
        preds = np.concatenate(predslist,axis=0)
        feats = np.concatenate(featslist,axis=0) if haskernel else None
        return preds,feats