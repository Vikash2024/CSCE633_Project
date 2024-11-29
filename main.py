import argparse
from preprocess import *
from dataloader import *
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSCE633 FINAL PROJECT")
    parser.add_argument('--task', type=str, help="train/test", default = 'train')
    train_dataloader, val_dataloader = read_dataset()
    
    