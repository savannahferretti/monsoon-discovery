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
        Purpose: Initialize Trainer with a model, dataloaders, and training settings.
        Args:
        - model (torch.nn.Module): untrained model
        - trainloader (torch.utils.data.DataLoader): training dataloader
        - validloader (torch.utils.data.DataLoader): validation dataloader
        - device (str): 'cuda' | 'cpu'
        - modeldir (str): directory for checkpoints
        - project (str): Weights & Biases project name
        - seed (int): training seed (used in the checkpoint name)
        - lr (float): initial learning rate
        - patience (int): epochs without validation improvement before stopping
        - criterion (str): loss function name (torch.nn or architectures)
        - epochs (int): maximum number of epochs
        - useamp (bool): use mixed precision on GPU
        - accumsteps (int): batches per optimizer step
        - compile (bool): use torch.compile
        - criterionkwargs (dict | None): keyword arguments for the loss function
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
        Purpose: Save a checkpoint as {name}_{seed}.pth and verify by reopening.
        Args:
        - name (str): run name
        - state (dict): model state_dict
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

    def forward_batch(self,batch,haskernel):
        '''
        Purpose: Predict one batch and return the predictions with their targets.
        Args:
        - batch (dict): batch with 'fields', 'local', 'target', and (for kernel models) 'dsig'
        - haskernel (bool): whether the model has an integration kernel
        Returns:
        - tuple[torch.Tensor, torch.Tensor]: predictions and targets
        '''
        fields = batch['fields'].to(self.device,non_blocking=True)
        local  = batch['local'].to(self.device,non_blocking=True)
        target = batch['target'].to(self.device,non_blocking=True)
        if haskernel:
            dsig   = batch['dsig'][0].to(self.device,non_blocking=True)
            output = self.model(fields,dsig,local)
        else:
            output = self.model(fields,local)
        return output,target

    def train_epoch(self,haskernel):
        '''
        Purpose: Run one training epoch.
        Args:
        - haskernel (bool): whether the model has an integration kernel
        Returns:
        - float: mean training loss
        '''
        self.model.train()
        self.optimizer.zero_grad()
        totalloss = 0.0
        for idx,batch in enumerate(self.trainloader):
            if self.useamp:
                with autocast('cuda',enabled=self.useamp):
                    outputvalues,targetvalues = self.forward_batch(batch,haskernel)
                    loss = self.criterion(outputvalues,targetvalues)
                    loss = loss/self.accumsteps
                self.scaler.scale(loss).backward()
                if (idx+1)%self.accumsteps==0 or (idx+1)==len(self.trainloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputvalues,targetvalues = self.forward_batch(batch,haskernel)
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
        Purpose: Run one validation epoch.
        Args:
        - haskernel (bool): whether the model has an integration kernel
        Returns:
        - float: mean validation loss
        '''
        totalloss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in self.validloader:
                if self.useamp:
                    with autocast('cuda',enabled=self.useamp):
                        outputvalues,targetvalues = self.forward_batch(batch,haskernel)
                        loss = self.criterion(outputvalues,targetvalues)
                else:
                    outputvalues,targetvalues = self.forward_batch(batch,haskernel)
                    loss = self.criterion(outputvalues,targetvalues)
                totalloss += loss.detach()*targetvalues.numel()
        return (totalloss/len(self.validloader.dataset)).item()

    def fit(self,name):
        '''
        Purpose: Train with early stopping and learning-rate decay, log to Weights & Biases, and save the best checkpoint.
        Args:
        - name (str): run name
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