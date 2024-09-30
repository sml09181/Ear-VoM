#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader   
from SpeechDataGenerator import SpeechDataGenerator
import os
from pathlib import Path
import numpy as np
import argparse
import time
from tqdm import tqdm
from models.x_vector_Indian_LID import X_vector
from sklearn.metrics import accuracy_score
from utils.utils import speech_collate
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


########## Argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-training_filepath',type=str,default='meta/training.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing.txt')
parser.add_argument('-validation_filepath',type=str, default='meta/validation.txt')

parser.add_argument('-input_dim', action="store_true", default=257)
parser.add_argument('-num_classes', action="store_true", default=8)
parser.add_argument('-lamda_val', action="store_true", default=0.1)
parser.add_argument('-batch_size', action="store_true", default=256)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)
args = parser.parse_args()

### Data related
dataset_train = SpeechDataGenerator(manifest=args.training_filepath,mode='train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

dataset_val = SpeechDataGenerator(manifest=args.validation_filepath,mode='train')
dataloader_val = DataLoader(dataset_train, batch_size=args.batch_size,shuffle=True,collate_fn=speech_collate) 

## Model related
os.environ["CUDA_VISIBLE_DEVICES"] = "3";
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", 0)
#print(args.input_dim, args.num_classes)
model = X_vector(args.input_dim, args.num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.98), eps=1e-9)
loss_fun = nn.CrossEntropyLoss()
model_dir = '/scratch/x-vector/models'
writer = SummaryWriter(Path(model_dir) / "logs" / time.strftime('%y%m%d-%X'))


def train(dataloader_train,epoch):
    train_loss_list=[]
    full_preds=[]
    full_gts=[]
    model.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
    
        features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
        labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
        features, labels = features.to(device),labels.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        pred_logits,x_vec = model(features)
        #### CE loss
        loss = loss_fun(pred_logits,labels)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            full_preds.append(pred)
        for lab in labels.detach().cpu().numpy():
            full_gts.append(lab)
            
    mean_acc = accuracy_score(full_gts,full_preds)
    mean_loss = np.mean(np.asarray(train_loss_list))
    writer.add_scalar("Loss/train", mean_loss, epoch)
    writer.add_scalar("Accuracy/train", mean_acc, epoch)
    print('Total training loss {} and training Accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
    

def validation(dataloader_val,epoch):
    model.eval()
    with torch.no_grad():
        val_loss_list=[]
        full_preds=[]
        full_gts=[]
        for i_batch, sample_batched in enumerate(dataloader_val):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy().T for torch_tensor in sample_batched[0]])).float()
            #print(features.shape)
            labels = torch.from_numpy(np.asarray([torch_tensor[0].numpy() for torch_tensor in sample_batched[1]]))
            features, labels = features.to(device),labels.to(device)
            pred_logits,x_vec = model(features)
            #### CE loss
            loss = loss_fun(pred_logits,labels)
            val_loss_list.append(loss.item())
            predictions = np.argmax(pred_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)
                
        mean_acc = accuracy_score(full_gts,full_preds)
        mean_loss = np.mean(np.asarray(val_loss_list))
        print('Total validation loss {} and Validation accuracy {} after {} epochs'.format(mean_loss,mean_acc,epoch))
        
        writer.add_scalar("Loss/valid", mean_loss, epoch)
        writer.add_scalar("Accuracy/valid", mean_acc, epoch)
        model_save_path = os.path.join(model_dir, 'best_check_point_'+str(epoch)+'_'+str(mean_loss))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
    
if __name__ == '__main__':
    for epoch in tqdm(range(args.num_epochs)):
        train(dataloader_train,epoch)
        validation(dataloader_val,epoch)
