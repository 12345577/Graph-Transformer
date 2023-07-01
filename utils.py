import heapq
import os

import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
from datetime import date
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn.functional as F


class TimeData(Dataset):
    def __init__(self, data, input_window, output_window, mode="train", device="cuda:2"):
        # re = len(data) % input_window
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.len_ = data.shape[0]
        self.feature_size = data.shape[len(data.shape) - 1]
        self.device = device
        self.mode = mode

    def __getitem__(self, index) -> T_co:
        if self.mode == "train":
            i = index
            tw = self.input_window
            output_window = self.output_window
            # 实现重构
            train_seq = self.data[i:i + tw, :]
            # 实现预测
            # train_seq = np.append(self.data[i:i + tw, :][:-output_window, :],
            #                       np.zeros((output_window, self.feature_size)), axis=0)
            train_label = self.data[i:i + tw, :]
            return torch.from_numpy(train_seq).type(torch.float32).cuda(), \
                   torch.from_numpy(train_label).type(torch.float32).cuda()
        elif self.mode == "test":
            i = index * self.input_window
            tw = self.input_window
            # 实现重构
            train_seq = self.data[i:i + tw, :]
            # 实现预测
            # train_seq = np.append(self.data[i:i + tw, :][:-output_window, :],
            #                       np.zeros((output_window, self.feature_size)), axis=0)
            train_label = self.data[i:i + tw, :]
            return torch.from_numpy(train_seq).type(torch.float32).cuda(), \
                   torch.from_numpy(train_label).type(torch.float32).cuda()

    def __len__(self):
        if self.mode == "train":
            return self.len_ - self.input_window + 1
        elif self.mode == "test":
            return (self.len_ - self.input_window) // self.input_window + 1


def get_data(data_, need_stand=True):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    if need_stand == True:
        data = data_
        series = data
        amplitude = scaler.fit_transform(series)
        return amplitude, scaler
    else:
        return data_, scaler


def normalize(mx):
    """行归一化"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_mx_adj(adj):
    # adj = adj.to_dense().cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return torch.FloatTensor(adj.todense()).cuda()


def normalize_adj_torch(mx):
    # mx = mx.to_dense()           #构建张量
    rowsum = mx.sum(1)  # 每行的数加在一起
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()  # 输出rowsum ** -1/2
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.  # 溢出部分赋值为0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)  # 对角化
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)  # 转置
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def load_adj(x):
    x = torch.from_numpy(x)
    a = x.T
    a = a / torch.norm(a, dim=-1, keepdim=True)
    adj = torch.mm(a, a.T)
    adj = np.nan_to_num(adj)
    where_are_nan = np.isnan(adj)
    where_are_inf = np.isinf(adj)
    # nan替换成0,inf替换成nan
    adj[where_are_nan] = 0
    adj[where_are_inf] = 0
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将一个scipy sparse matrix转化为torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def kl(series_s, prior_s):
    items = F.kl_div(F.log_softmax(series_s), F.softmax(prior_s), reduce=False) \
            + F.kl_div(F.log_softmax(prior_s), F.softmax(series_s), reduce=False)
    item = torch.mean(torch.mean(items, dim=-1), dim=-1)
    return item


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def test_model(eval_model, data_source, epoch, scaler, input_window, output_window, feature_size, file_name,
                    batch_size):
    eval_model.eval()
    batch_size = 512
    dataset = TimeData(data_source, input_window, output_window=input_window, mode="train")
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    # truth = torch.zeros(len(val_loader) * batch_size, input_window, feature_size)
    # pre = torch.zeros(len(val_loader) * batch_size, input_window, feature_size)
    truth = torch.zeros(len(val_loader) * batch_size, 1, feature_size)
    pre = torch.zeros(len(val_loader) * batch_size, 1, feature_size)
    adj = []
    series = []
    prior = []
    print("======================start test======================")
    with torch.no_grad():
        index = 0
        for batch, (data, targets) in enumerate(val_loader):
            output, atte, adj_s, series_s, prior_s, _ = eval_model(data)
            adj.append(adj_s)
            for i in range(batch_size):
                truth[index] = targets[i][-1, :].clone()
                pre[index] = output[i][-1, :].clone()
                index = index + 1
                if index % 10000 == 0:
                    print(index, "/", len(val_loader) * batch_size)

            torch.cuda.empty_cache()

    adj = torch.cat(adj)
    np.save("adj"+file_name+".npy", adj.cpu().numpy())
    np.save("pre" + file_name + "_" + str(epoch) + ".npy", pre.cpu())
    np.save("truth" + file_name + "_" + str(epoch) + ".npy", truth.cpu())


def predict_future2(eval_model, data_source, epoch, scaler, input_window, output_window, feature_size, file_name,
                    batch_size):
    eval_model.eval()
    batch_size = 1
    dataset = TimeData(data_source, input_window, output_window=input_window, mode="train")
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    truth = torch.zeros(len(val_loader) * batch_size, input_window, feature_size)
    pre = torch.zeros(len(val_loader) * batch_size, input_window, feature_size)
    adj = []
    adj1 = []
    print("start")
    with torch.no_grad():
        index = 0
        for batch, (data, targets) in enumerate(val_loader):
            for i in range(batch_size):
                truth[index] = targets[i][-input_window:, :]
                output, atte, adjs, adj1s = eval_model(data)
                adj.extend(adjs)
                adj1.extend(adj1s)
                # 1/0
                pre[index] = output[i][-input_window:, :]
                index = index + 1
                if index % 10000 == 0:
                    print(index, "/", len(val_loader) * batch_size)
    # adj = torch.cat(adj)
    # adj1 = torch.cat(adj1)
    # np.save("adj.npy", adj.cpu().numpy())
    # np.save("adj1.npy", adj1.cpu().numpy())
    np.save("pre_train_" + file_name + "_" + str(epoch) + ".npy", pre.cpu())
    np.save("truth_train_" + file_name + "_" + str(epoch) + ".npy", truth.cpu())


def evaluate(eval_model, data_source, calculate_loss_over_all_values, output_window, criterion, batch_size,
             input_window, scaler, epoch):
    eval_model.eval()  # Turn on the evaluation mode
    all_loss = []
    dataset = TimeData(data_source, input_window, output_window)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=1)
    with torch.no_grad():
        for batch, (data, targets) in enumerate(val_loader):
            output, atte_s, adj_s, series_s, prior_s, _ = eval_model(data)

            loss = criterion(output, targets)

            all_loss.append(loss.cpu().item())
            # print(all_loss)
    all_loss = np.array(all_loss)
    all_loss[np.isnan(all_loss)] = 0
    all_loss = np.mean(all_loss)
    return all_loss / len(val_loader)


def train(train_data, epoch, scheduler, optimizer, criterion, model, batch_size, output_window, input_window, scaler, cos_adj,
          calculate_loss_over_all_values=False):
    model.train()
    total_loss = 0.
    all_loss = 0.
    start_time = time.time()
    alpha = F.tanh(torch.range(0, 2, 0.1))[epoch-1]

    dataset = TimeData(train_data, input_window, output_window)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        output, atte_s, adj_s, series_s, prior_s, out = model(data)

        kl_loss = .0
        for i in range(len(adj_s)):
            kl_loss += kl(cos_adj, adj_s[i])
        kl_loss = kl_loss / len(adj_s)

        rec_loss = criterion(out, targets)

        loss = criterion(output, targets)



        # TODO FULL
        loss1 = rec_loss + (20 * kl_loss) + (.1 * loss)
        loss2 = rec_loss - (10 * kl_loss) - (.05 * loss)
        loss1.backward(retain_graph=True)
        loss2.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        all_loss += loss.item()
        log_interval = 100
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.10f}'.format(epoch, batch, len(train_loader), scheduler.get_lr()[0],
                                                        elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
    return all_loss / len(train_loader)


def test(model, train_data, val_data, input_window, test_labels, anomaly_ratio=10):
    model.eval()
    batch_size = 128

    print("======================TEST MODE======================")

    criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    dataset = TimeData(train_data, input_window, 1)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    attens_energy = []
    train_out = []
    train_in = []
    for batch, (data, targets) in enumerate(train_loader):
        output, attes, adjs, _, _ = model(data)
        # ouput shape (batch, window, feature)
        # 计算损失的方法 只用窗口的最后一个元素，或者是用全部元素
        if batch % (len(train_loader) // 10) == 0:
            print("stastic on the train set:", batch // (len(train_loader) // 10) * 10, "%")
        for i in range(batch_size):
            # loss = torch.mean(criterion(data[i, -1, :], output[i, -1, :]), dim=-1)
            train_out.append(output[i].unsqueeze(0).detach().cpu())
            train_in.append(data[i].unsqueeze(0).detach().cpu())
            loss = torch.mean(torch.mean(criterion(data[i, :, :], output[i, :, :]), dim=-1), dim=-1)

            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)
    train_out = torch.cat(train_out)
    np.save("train_out.npy", train_out.numpy())
    train_in = torch.cat(train_in)
    np.save("train_in.npy", train_in.numpy())
    train_energy = np.array(attens_energy).reshape(-1)

    # (2) find the threshold 阈值选择的时候用不用测试集的数据
    attens_energy = []
    dataset = TimeData(val_data, input_window, 1)
    val_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
    test_out = []
    test_in = []
    for batch, (data, targets) in enumerate(val_loader):
        output, attes, adjs, _, _ = model(data)
        # ouput shape (batch, window, feature)
        # 计算损失的方法 只用窗口的最后一个元素，或者是用全部元素
        if batch % (len(val_loader) // 10) == 0:
            print("find the threshold:", batch // (len(val_loader) // 10) * 10, "%")
        for i in range(batch_size):
            # loss = torch.mean(criterion(data[i, -1, :], output[i, -1, :]), dim=-1)
            test_out.append(output[i].unsqueeze(0).detach().cpu())
            test_in.append(data[i].unsqueeze(0).detach().cpu())
            loss = torch.mean(torch.mean(criterion(data[i, :, :], output[i, :, :]), dim=-1), dim=-1)

            loss = loss.detach().cpu().numpy()
            attens_energy.append(loss)
    test_out = torch.cat(test_out)
    np.save("test_out.npy", test_out.numpy())
    test_in = torch.cat(test_in)
    np.save("test_in.npy", test_in.numpy())
    test_energy = np.array(attens_energy).reshape(-1)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
    np.save("threshold.npy", combined_energy)
    print("Threshold :", thresh)

    # (3) evaluation on the test set
    test_labels = np.array(test_labels)
    np.save("test_energy.npy", test_energy)
    np.save("test_labels.npy", test_labels)

    pred = (test_energy > thresh).astype(int)

    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    gt = gt[:len(pred)]

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    np.save("pred.npy", pred)
    np.save("gt.npy", gt)

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    return accuracy, precision, recall, f_score


def threshold_test(test_energy, test_labels, thresh=1):
    # (3) evaluation on the test set
    test_labels = np.array(test_labels)
    print("thresh", thresh)
    np.save("test_energy.npy", test_energy)
    np.save("test_labels.npy", test_labels)

    pred = (test_energy > thresh).astype(int)

    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    gt = gt[:len(pred)]

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    np.save("pred.npy", pred)
    np.save("gt.npy", gt)

    np.save("adjust.npy", pred)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    return accuracy, precision, recall, f_score


def percentage_test(combined_energy, test_energy, test_labels, anomaly_ratio=10):
    thresh = np.percentile(combined_energy, 100 - anomaly_ratio)
    np.save("threshold.npy", combined_energy)
    print("Threshold :", thresh)

    # (3) evaluation on the test set
    test_labels = np.array(test_labels)

    pred = (test_energy > thresh).astype(int)

    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    gt = gt[:len(pred)]

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred = np.array(pred)
    gt = np.array(gt)
    np.save("pred.npy", pred)
    np.save("gt.npy", gt)

    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    np.save("adjust.npy", pred)
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    return accuracy, precision, recall, f_score


