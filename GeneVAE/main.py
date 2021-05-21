import sys, os
# import options_parser as op
import numpy as np
import torch
import random
import train
from torch.utils.data import Dataset, DataLoader
from dataloader import get_loaders
import argparse

SEED = 459
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main(args):
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    train_loader, val_loader, test_loader, input_size = get_loaders(args.data, args.filter_genes, args.batch_size)
    
    train.train_network(train_loader, val_loader, save_dir, input_size)

    net = torch.load(os.path.join(save_dir,'trained_model_best.pth'))
    net.eval()
    val_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, batch in enumerate(test_loader):
        inputs = Variable(batch).cuda()
        with torch.no_grad():
            output = net(inputs)
        loss = criterion(output, inputs)

        test_loss += loss.cpu().data.numpy() * len(inputs)
    test_loss = test_loss / len(test_loader.dataset)
    



if __name__ == "__main__":
    # args = op.setup_options()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--filter_genes', type=str )
    parser.add_argument('--save_dir', type=str, default='./data/model')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_epoch', type=int, default=0)
    
    args = parser.parse_args()
    main(args)