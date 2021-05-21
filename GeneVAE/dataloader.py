
import math, random, sys, time, os
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def get_loaders(path, filter_genes, batch_size):

    GE_df = pd.read_csv(path, index_col=0)
    print('GE_DF LOADED!')
    if filter_genes is not None:
        with open(filter_genes) as f:
            filter_gene_indx = [int(s.strip()) for s in f.readlines()]
    else:
        filter_gene_indx = range(GE_df.shape[1]) # include all the genes

    train_idx, test_idx = train_test_split(range(len(GE_df)), test_size=0.1)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2)
    
    np.save('./data/train_idx', train_idx)
    np.save('./data/test_idx', test_idx)
    np.save('./data/val_idx', val_idx)
    
    train_set = GeneVecs(GE_df.iloc[train_idx, filter_gene_indx].values)
    val_set = GeneVecs(GE_df.iloc[val_idx, filter_gene_indx].values)
    test_set = GeneVecs(GE_df.iloc[test_idx, filter_gene_indx].values)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              num_workers=0, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                              num_workers=0, shuffle=False,
                              pin_memory=True)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             num_workers=0, shuffle=False,
                             pin_memory=True)

    return   train_loader, val_loader, test_loader, len(filter_gene_indx) , train_set                                            


# Stores gene expression vectors as pytorch dataset
class GeneVecs(Dataset):

    # Data comes in as a list
    def __init__(self, cell_lines):
        self.cell_lines = np.log2(cell_lines + 1)

    def __len__(self):
        return len(self.cell_lines)

    def __getitem__(self, idx):
        sample = self.cell_lines[idx]
        return torch.from_numpy(sample).float()
