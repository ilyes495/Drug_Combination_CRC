import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys, time, os
import numpy as np
import argparse

from tqdm import tqdm
from copy import deepcopy
import pickle as p
from torchvision.utils import make_grid
from tqdm import tqdm
import model


def train_network(train_loader, val_loader, save_dir, input_size):

    
    net = model.Net(input_size)
    
    print("INPUT SIZE (number of genes): ", input_size)
    # Print Num of Params in Model
    params = 0
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

   

    # Adam optimization (but you can try SGD as well)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.cuda()
    num_epochs = 100000
    best_loss = np.float("inf")
    pbar = tqdm(range(num_epochs))
    for i in pbar:
        pbar.set_description(f"Epoch: {i}")
        # print("Epoch: ", i)
        train_loss = train_step(net, optimizer, train_loader)
        # print("Train Loss: ", train_loss)
        val_loss = val_step(net, val_loader)
        # print("Validation Loss: ", val_loss)
        pbar.set_postfix({'Train Loss': train_loss, 'Validation Loss':val_loss, 'Best':best_loss})

        if train_loss < 1e-15:
            break
        if val_loss < best_loss:
            best_loss = val_loss
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            torch.save(d, os.path.join(save_dir,'trained_model_best.pth'))
            net.cuda()
        # print("Best Validation Loss: ", best_loss)
        


def train_step(net, optimizer, train_loader):
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)
    criterion = torch.nn.MSELoss()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = Variable(batch).cuda()
        output, _ = net(inputs)

        loss = criterion(output, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)

    end = time.time()
    # print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, batch in enumerate(val_loader):
        inputs = Variable(batch).cuda()
        with torch.no_grad():
            output, _ = net(inputs)
        loss = criterion(output, inputs)

        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss