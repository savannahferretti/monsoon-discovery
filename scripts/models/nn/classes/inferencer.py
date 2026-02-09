#!/usr/bin/env python

import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Inferencer:

    def __init__(self,model,dataloader,device):
        '''
        Purpose: Initialize Inferencer for model evaluation on test/validation data.
        Args:
        - model (torch.nn.Module): trained model instance
        - dataloader (torch.utils.data.DataLoader): dataloader for inference
        - device (str): device to use (cuda or cpu)
        '''
        self.model      = model
        self.dataloader = dataloader
        self.device     = device

    def predict(self,haskernel):
        '''
        Purpose: Generate predictions for all samples in the dataloader.
        Args:
        - haskernel (bool): whether model has integration kernel
        Returns:
        - np.ndarray: predictions array with shape (nsamples,)
        '''
        self.model.eval()
        predslist = []
        with torch.no_grad():
            for batch in self.dataloader:
                fields = batch['fields'].to(self.device,non_blocking=True)
                lf     = batch['lf'].to(self.device,non_blocking=True)
                mask   = batch['mask'].to(self.device,non_blocking=True) if 'mask' in batch else None
                if haskernel:
                    dlev   = batch['dlev'].to(self.device,non_blocking=True)
                    output = self.model(fields,dlev,lf,mask=mask)
                else:
                    output = self.model(fields,lf,mask=mask)
                predslist.append(output.detach().cpu().numpy())
        return np.concatenate(predslist,axis=0).astype(np.float32)

    def extract_weights(self,nonparam):
        '''
        Purpose: Extract normalized kernel weights as a list of components.
        Args:
        - nonparam (bool): whether the kernel is non-parametric
        Returns:
        - list[np.ndarray]: list of component weight arrays; each has shape (nfieldvars, nlevs)
        '''
        if self.model.kernel.norm is None:
            raise RuntimeError('`model.kernel.norm` was not populated during forward pass')
        norm = self.model.kernel.norm.detach().cpu().numpy().astype(np.float32)
        if nonparam:
            return [norm]
        if hasattr(self.model.kernel,'get_weights'):
            batch = next(iter(self.dataloader))
            with torch.no_grad():
                dlev = batch['dlev'].to(self.device,non_blocking=True)
                self.model.kernel.get_weights(dlev,self.device,decompose=True)
            if self.model.kernel.components is not None:
                components = self.model.kernel.components.detach().cpu().numpy().astype(np.float32)
                return [components[i] for i in range(components.shape[0])]
        return [norm]
