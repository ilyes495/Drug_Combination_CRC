"""
Dispose of drug data, gene expression data and drug response data
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torch

class LatentDataset(Dataset):
    def __init__(self, response_df, drug_df, cell_df,mutation_df=None, target_col='response', drug_col='broad_id', cell_col='cell_id', include_mutation=False):
        self.response_df = response_df
        self.drug_df = drug_df
        self.cell_df = cell_df
        self.mutation_df = mutation_df
        self.target_col = target_col
        self.drug_col = drug_col
        self.cell_col = cell_col
        self.include_mutation = include_mutation
        

    def __getitem__(self, index):
       
        target = self.response_df.loc[index, self.target_col]
        target = torch.tensor(target).float()

        drug_id = self.response_df.loc[index, self.drug_col]
        try:
            dLatentVec = self.drug_df[self.drug_df[self.drug_col] == drug_id].values[0,2:].astype(np.float32)
        except:
            raise Exception(f'Problem at drug_id={drug_id}, {self.drug_df.shape}')
        dLatentVec = torch.from_numpy(dLatentVec).float()


        cell_id = self.response_df.loc[index, self.cell_col]
        geLatentVec = self.cell_df.loc[cell_id].values[7:-2].astype(np.float32)
        geLatentVec = torch.from_numpy(geLatentVec).float()
        if self.include_mutation:
            mutations = self.mutation_df.loc[cell_id].values.astype(np.float32)
            return geLatentVec, dLatentVec, mutations, target
        
        return geLatentVec, dLatentVec, target

    def __len__(self):
        return len(self.response_df)
