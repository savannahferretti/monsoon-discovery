#!/usr/bin/env python

import os
import time
import torch
import wandb
import logging
import scripts.models.nn.architectures as architectures
from torch.amp import autocast,GradScaler

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(self,model,trainloader,validloader,device,modeldir,project,seed,lr,patience,criterion,epochs,useamp,accumsteps,compile,criterionkwargs=None):
        '''
        Purpose: Initialize Trainer with model, dataloaders, and training configuration.
        Args:
        - model (torch.nn.Module): initialized model instance
        - trainloader (torch.utils.data.DataLoader): training dataloader
        - validloader (torch.utils.data.DataLoader): validation dataloader
        - device (str): device to use (cuda or cpu)
        - modeldir (str): output directory for checkpoints
        - project (str): project name for Weights & Biases logging
        - seed (int): random seed for reproducibility
        - lr (float): initial learning rate
        - patience (int): early stopping patience
        - criterion (str): loss function name
        - epochs (int): maximum number of epochs
        - useamp (bool): whether to use automatic mixed precision
        - accumsteps (int): gradient accumulation steps for larger effective batch size
        - compile (bool): whether to use torch.compile for faster training
        '''
        self.model       = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.device      = device
        self.modeldir    = modeldir
        self.project     = project
        self.seed        = seed
        self.lr          = lr
        self.patience    = patience
        self.epochs      = epochs
        self.useamp      = useamp and (device=='cuda')
        self.accumsteps  = accumsteps
        self.optimizer   = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5,patience=2,min_lr=1e-6)
        self.scaler      = GradScaler('cuda') if self.useamp else None
        kwargs = criterionkwargs or {}
        if hasattr(torch.nn,criterion):
            self.criterion = getattr(torch.nn,criterion)(**kwargs)
        else:
            self.criterion = getattr(architectures,criterion)(**kwargs)
        if compile and hasattr(torch,'compile'):
            logger.info('   Compiling model with torch.compile...')
            self.model = torch.compile(self.model)

    def save_checkpoint(self,name,state):
        '''
        Purpose: Save best model checkpoint and verify by reopening.
        Args:
        - name (str): model name
        - state (dict): model state_dict to save
        Returns:
        - bool: True if save successful, False otherwise
        '''
        os.makedirs(self.modeldir,exist_ok=True)
        filename = f'{name}_{self.seed}.pth'
        filepath = os.path.join(self.modeldir,filename)
        logger.info(f'      Attempting to save {filename}...')
        try:
            torch.save(state,filepath)
            _ = torch.load(filepath,map_location='cpu')
            logger.info('         File write successful')
            return True
        except Exception:
            logger.exception('         Failed to save or verify')
            return False

    def _forward(self,batch,haskernel):
        '''
        Purpose: Run a forward pass on a batch, dispatching to baseline or kernel model interface.
        Args:
        - batch (dict): batch dictionary with keys 'fields', 'local', 'target', and optionally 'dlev' and 'mask'
        - haskernel (bool): whether model has integration kernel
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: (predictions, targets)
        '''
        fields = batch['fields'].to(self.device,non_blocking=True)
        local  = batch['local'].to(self.device,non_blocking=True)
        target = batch['target'].to(self.device,non_blocking=True)
        mask   = batch['mask'].to(self.device,non_blocking=True) if 'mask' in batch else None
        if haskernel:
            dlev   = batch['dlev'][0].to(self.device,non_blocking=True)
            output = self.model(fields,dlev,local,mask=mask)
        else:
            output = self.model(fields,local,mask=mask)
        return output,target

    def train_epoch(self,haskernel):
        '''
        Purpose: Execute one training epoch with gradient accumulation and mixed precision.
        Args:
        - haskernel (bool): whether model has integration kernel
        Returns:
        - float: average training loss for the epoch
        '''
        self.model.train()
        self.optimizer.zero_grad()
        totalloss = 0.0
        for idx,batch in enumerate(self.trainloader):
            if self.useamp:
                with autocast('cuda',enabled=self.useamp):
                    outputvalues,targetvalues = self._forward(batch,haskernel)
                    loss = self.criterion(outputvalues,targetvalues)
                    loss = loss/self.accumsteps
                self.scaler.scale(loss).backward()
                if (idx+1)%self.accumsteps==0 or (idx+1)==len(self.trainloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputvalues,targetvalues = self._forward(batch,haskernel)
                loss = self.criterion(outputvalues,targetvalues)
                loss = loss/self.accumsteps
                loss.backward()
                if (idx+1)%self.accumsteps==0 or (idx+1)==len(self.trainloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            totalloss += loss.detach()*self.accumsteps*targetvalues.numel()
        avgloss = (totalloss/len(self.trainloader.dataset)).item()
        return avgloss

    def validate_epoch(self,haskernel):
        '''
        Purpose: Execute one validation epoch.
        Args:
        - haskernel (bool): whether model has integration kernel
        Returns:
        - float: average validation loss for the epoch
        '''
        totalloss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in self.validloader:
                if self.useamp:
                    with autocast('cuda',enabled=self.useamp):
                        outputvalues,targetvalues = self._forward(batch,haskernel)
                        loss = self.criterion(outputvalues,targetvalues)
                else:
                    outputvalues,targetvalues = self._forward(batch,haskernel)
                    loss = self.criterion(outputvalues,targetvalues)
                totalloss += loss.detach()*targetvalues.numel()
        return (totalloss/len(self.validloader.dataset)).item()
        
    def fit(self,name):
        '''
        Purpose: Train model with early stopping and learning rate scheduling.
        Args:
        - name (str): model name
        '''
        haskernel = hasattr(self.model,'kernel')
        wandb.init(
            project=self.project,
            name=name,
            config={
                'Seed':self.seed,
                'Epochs':self.epochs,
                'Batch size':self.trainloader.batch_size*self.accumsteps,
                'Initial learning rate':self.lr,
                'Early stopping patience':self.patience,
                'Loss function':self.criterion.__class__.__name__,
                'Number of parameters':self.model.nparams if hasattr(self.model,'nparams') else sum(p.numel() for p in self.model.parameters()),
                'Device':self.device,
                'Mixed precision':self.useamp,
                'Training samples':len(self.trainloader.dataset),
                'Validation samples':len(self.validloader.dataset)})
        beststate = None
        bestloss  = float('inf')
        bestepoch = 0
        noimprove = 0
        starttime = time.time()
        for epoch in range(1,self.epochs+1):
            trainloss = self.train_epoch(haskernel)
            validloss = self.validate_epoch(haskernel)
            self.scheduler.step(validloss)
            if validloss<bestloss:
                beststate = {key:value.detach().cpu().clone() for key,value in self.model.state_dict().items()}
                bestloss  = validloss
                bestepoch = epoch
                noimprove = 0
            else:
                noimprove += 1
            wandb.log({
                'Epoch': epoch,
                'Training loss':trainloss,
                'Validation loss':validloss,
                'Learning rate':self.optimizer.param_groups[0]['lr']})
            logger.info(f'   Epoch {epoch}/{self.epochs} | Training Loss = {trainloss:.4f} | Validation Loss = {validloss:.4f}')
            if noimprove>=self.patience:
                break
        duration = time.time()-starttime
        wandb.run.summary.update({'Best validation loss':bestloss})
        logger.info(f'   Training completed in {duration/60:.1f} minutes!')
        if beststate is not None:
            self.save_checkpoint(name,beststate)
        wandb.finish()