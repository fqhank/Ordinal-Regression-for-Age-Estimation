import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from dataset import AgeDataset
from network import AgeNet
from utils import *

def make_args():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--batch_size',default=1024)
    parser.add_argument('--test_path',default='E:\PKU\cv_learning\ordinal-regression\dataset\\test_data_dirty')
    parser.add_argument('--trained_model',default='E:\PKU\cv_learning\ordinal-regression\model\\best.pth',help='the path to the saved trained model')
    args = parser.parse_args()
    return args

def test_loop(model,loader,device):
    total = len(loader.dataset)
    mae = 0
    for step,batch in enumerate(loader):
        x,label,age = batch
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        predict = model(x)
        mae += MAE(predict,age)*len(age)
    mae = mae/total
    print('test || MAE:{}'.format(mae))

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using [{device}] for the work')

    model = AgeNet()
    dict = torch.load(args.trained_model)
    model.load_state_dict(dict)
    print('model loaded successfully!')
    model.to(device)

    test_dataset = AgeDataset(args.test_path)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size)

    print('-----testing-----')
    with torch.no_grad():
        model.eval()
        test_loop(model,test_loader,device)

if __name__ == '__main__':
    args = make_args()
    main(args)