import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math

from dglgo.utils.early_stop import EarlyStopping

from utils import *
from model import Transformer
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,0"

# Press the green button in the gu-tter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Demo of argparse")

    parser.add_argument('--dataset', type=str, default="SMD")
    parser.add_argument('--group', type=str, default="1-1")
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--filename', type=str, default="the name of result that save loss and score")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--input_window', type=int, default=100)
    parser.add_argument('--model_path', type=str, default="model name without pkl")

    args = parser.parse_args()
    dataset = args.dataset
    print(args)
    seed = 0
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    torch.multiprocessing.set_start_method('spawn')
    input_window = args.input_window
    output_window = 20
    batch_size = args.batch_size
    group = args.group
    patience = args.patience
    epochs = args.epoch
    num_head = args.num_head
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = 3

    if dataset == "MSL":
        # MSL
        train_data_ = np.load("data/MSL/MSL_train.npy")
        val_data_ = np.load("data/MSL/MSL_test.npy")

        # train_data_ = np.load("data/MSL/C-2_train.npy")
        # val_data_ = np.load("data/MSL/C-2_test.npy")
        feature_size = 55
        filename = "msl"
        # state_dict = torch.load('./msllstm.pth')
        early_stopping = EarlyStopping(patience, checkpoint_path="msllstm.pth")
    elif dataset == "PSM":
        # PSM
        train_data_ = pd.read_csv("data/PSM/train.csv")
        train_data_ = train_data_.fillna(method="bfill")
        train_data_ = train_data_.to_numpy()
        # train_data_ = train_data_[:len(train_data_)//2]
        train_data_ = train_data_[:, 1:]
        val_data_ = pd.read_csv("data/PSM/test.csv")
        val_data_ = val_data_.fillna(method="bfill")
        val_data_ = val_data_.to_numpy()
        # val_data_ = val_data_[:len(val_data_)//2]
        val_data_ = val_data_[:, 1:]
        feature_size = 25
        filename = "psm"
        # state_dict = torch.load('./'+args.filename+'.pth')
        early_stopping = EarlyStopping(patience, checkpoint_path="psmlstminconv.pth")
    elif dataset == "SMAP":
        # SMAP
        train_data_ = np.load("data/SMAP/SMAP_train.npy")
        # train_data_ = train_data_[:len(train_data_) // 10]
        val_data_ = np.load("data/SMAP/SMAP_test.npy")
        val_data_ = val_data_[:len(val_data_)]
        feature_size = 25
        filename = "smap"
        # state_dict = torch.load('./'+args.filename+'.pth')
        early_stopping = EarlyStopping(patience, checkpoint_path="smaplstm.pth")
    elif dataset == "SMD":
        # SMD
        if group == "0-0":
            train_data_ = np.load("data/SMD/SMD_train.npy")
            val_data_ = np.load("data/SMD/SMD_test.npy")
            filename = "smd"
        else:
            train_data_ = np.load("data/SMD/machine-" + group + "_train.npy")
            val_data_ = np.load("data/SMD/machine-" + group + "_test.npy")
            filename = "smd" + group
            model_path = "smd" + group + "HC"
        feature_size = 38
        # state_dict = torch.load('./'+args.filename+'.pth')
        early_stopping = EarlyStopping(patience, checkpoint_path="smdlstm.pth")
    elif dataset == "MSDS":
        # SMAP
        train_data_ = np.load("data/MSDS/train.npy")
        val_data_ = np.load("data/MSDS/test.npy")
        val_data_ = val_data_[:len(val_data_)]
        feature_size = 10
        model_path = args.model_path
        filename = args.filename
    elif dataset == "WADI":
        # SMAP
        train_data_ = np.load("data/WADI/train.npy")
        val_data_ = np.load("data/WADI/test.npy")
        feature_size = 127
        filename = "wadi"
        # need_stand = False
    elif dataset == "SWAT":
        # SMAP

        train_data_ = pd.read_csv("data/SWAT/SWaT_train.csv")
        train_data_ = train_data_.fillna(method="bfill")
        train_data_ = train_data_.to_numpy()
        train_data_ = train_data_[:, 1:-1]
        val_data_ = pd.read_csv("data/SWAT/SWaT_test.csv")
        val_data_ = val_data_.fillna(method="bfill")
        val_data_ = val_data_.to_numpy()
        val_data_ = val_data_[:, 1:-2]

        feature_size = 51
        filename = "swat"
        # need_stand = False
    else:
        print("THE DATA IS NOT INCLUDE IN THIS PROJECT.")
        RuntimeError

    model_path = args.model_path
    filename = args.filename
    state_dict = torch.load('./' + model_path + '.pth')
    train_data, train_scaler = get_data(data_=train_data_)
    val_data, val_scaler = get_data(data_=val_data_)
    print("load data finished")
    adj = load_adj(train_data)
    print("load adj finished")
    model = Transformer(feature_size=feature_size, num_layers=5, input_window=input_window, adj=adj,
                        batch_size=batch_size, num_head=num_head).cuda()

    model.load_state_dict(state_dict)
    print("load model parameters finished")
    total_num = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %d" % total_num)
    criterion = nn.MSELoss()

    test_model(eval_model=model, data_source=val_data, scaler=val_scaler, file_name=filename,
                    input_window=input_window, output_window=input_window, epoch=epochs,
                    feature_size=feature_size, batch_size=batch_size)

