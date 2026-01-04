import sys, os, re

pPath = os.path.split(os.path.realpath(sys.argv[0]))[0]
sys.path.append(pPath)
pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(sys.argv[0]))[0])
sys.path.append(pPath)
import pickle
import math
from collections import Counter
import os
import numpy as np
import chardet
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from torch.nn import AdaptiveAvgPool1d

device = torch.device("cuda:0")



def load_bench_data(file):
    with open(file, 'rb') as f:
        raw_data = f.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
    tmp = pd.read_csv(file, encoding=detected_encoding, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, file, train=True)
    data_iter = list(data_iter)
    return data_iter


def load_Ind_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, file, train=True)
    data_iter = list(data_iter)
    return data_iter


def data_construct(seqs, labels, file, train):
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    longest_num = len(max(seqs, key=len))
    sequences = [i.ljust(longest_num, 'X') for i in seqs]
    pep_codes = []
    for pep in seqs:
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep).to(device))
    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True).to(device)
    pos_embed = np.array(position_encoding(labels, sequences))
    HF_feature = HF_encoding(seqs, sequences, labels).detach().numpy()
    dataset = Data.TensorDataset(embed_data, torch.FloatTensor(pos_embed), torch.FloatTensor(HF_feature),
                                 torch.LongTensor(labels))
    batch_size = 128
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return data_iter


def BLOSUM62(seqs):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    encodings = []
    for seq in seqs:
        code = []
        for aa in seq:
            code += blosum62[aa]
        encodings.append(code)
    return encodings


def position_encoding(labels, seqs):
    d = 128
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return np.array(res)


def HF_encoding(seqs, sequences, labels):
    embed_dim = 256
    num_heads = 8
    dropout = 0.1
    d_model = 256
    multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
    result = BLOSUM62(sequences)
    result = torch.tensor(result)
    linear_layer1 = nn.Linear(result.size(-1), d_model)
    result = linear_layer1(result.float())
    return result

class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.conv_hidden_dim = 64
        self.lstm_hidden_dim = 32
        self.MLP_embed_dim = 64
        self.MLP_hidden_dim = 64
        self.dropout = 0.2
        self.batch_size = 64
        self.emb_dim = 128
        self.embedding_seq = nn.Embedding(24, self.emb_dim, padding_idx=0, device=device)
        self.encoder_layer_seq = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=8)
        self.transformer_encoder_seq = nn.TransformerEncoder(self.encoder_layer_seq, num_layers=1)
        self.conv_seq = nn.Sequential(
            nn.Conv1d(128, 20, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(20),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )
        self.conv_HF = nn.Sequential(
            nn.Conv1d(128, 20, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(20),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.MaxPool1d(3, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )
        self.linearlayer = nn.Linear(1, 128, device=device)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(20, self.lstm_hidden_dim, num_layers=2, bidirectional=True, dropout=self.dropout,
                            batch_first=False, device=device)

        self.block1 = self._make_res_block(self.MLP_embed_dim, self.MLP_hidden_dim).to(device)
        self.block2 = self._make_res_block(self.MLP_hidden_dim, self.MLP_hidden_dim).to(device)

        self.classifier = nn.Sequential(
            nn.Linear(self.MLP_hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, num_classes)
        )

        self.branch_attn = nn.MultiheadAttention(
            embed_dim=20,
            num_heads=5,
            batch_first=True,
            dropout=0.0
        ).to(device)
        self.attn_scale = nn.Parameter(torch.ones(1) * 0.1)

        self.adaptive_pool = AdaptiveAvgPool1d(1)
    def _make_res_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, x, pos_embed, HF):

        output1 = self.embedding_seq(x) + pos_embed.to(device)
        output1 = self.transformer_encoder_seq(output1)
        output1 = output1.permute(0, 2, 1)
        output1 = self.conv_seq(output1)  # (batch, 20, seq1)

        output2 = HF.unsqueeze(2)
        output2 = output2.to(device)
        output2 = self.relu(self.linearlayer(output2))
        output2 = output2.permute(0, 2, 1)
        output2 = self.conv_HF(output2)  # (batch, 20, seq2)

        seq_len1 = output1.size(2)
        seq_len2 = output2.size(2)
        target_len = min(seq_len1, seq_len2)

        if seq_len1 > target_len:

            self.adaptive_pool.output_size = target_len
            output1 = self.adaptive_pool(output1)
        else:

            output1 = output1[:, :, :target_len]

        if seq_len2 > target_len:
            self.adaptive_pool.output_size = target_len
            output2 = self.adaptive_pool(output2)
        else:
            output2 = output2[:, :, :target_len]

        q = output1.permute(0, 2, 1)
        k = v = output2.permute(0, 2, 1)

        attn_out, attn_weights = self.branch_attn(q, k, v)
        attn_out = attn_out * self.attn_scale
        fused = q + attn_out
        fused = fused.permute(0, 2, 1)

        output = torch.cat([output1, fused], 2)
        output = output.permute(2, 0, 1)
        output, (h, c) = self.lstm(output)
        output = output.permute(1, 0, 2)
        identity = output.mean(dim=1)
        out = self.block1(output.mean(dim=1)) + identity
        out = self.block2(out) + out
        out = self.classifier(out)

        return out, attn_weights

def evaluate(data_iter, net, epoch):
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    true_labels = []
    for x, pos, hf, y in data_iter:
        outputs, rep = net(x, pos, hf)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y.tolist()
        label = y.cpu()
        if rep is not None and rep.numel() > 0:
            rep = rep.cpu()
            rep_list.append(rep.detach().numpy())
        else:
            rep_list.append(None)
        true_labels.append(label.detach().numpy())
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real, epoch)
    return performance, roc_data, prc_data, rep_list, label_real, true_labels


def caculate_metric(pred_prob, label_pred, label_real, epoch):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    ACC = float(tp + tn) / test_num
    Sensitivity = float(tp) / (tp + fn) if (tp + fn) != 0 else 0
    Specificity = float(tn) / (tn + fp) if (tn + fp) != 0 else 0
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) if (tp + fp) * (
                tp + fn) * (tn + fp) * (tn + fn) != 0 else 0
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)
    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    l2_lambda = 0.0
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)
    total_regularization_loss = l2_lambda * regularization_loss
    total_loss = criterion(output.to(device), label.to(device)) + total_regularization_loss.to(device)
    return total_loss


def train_test(train_iter, test_iter):
    net = Net()
    net = net.to(device)
    lr = 0.002
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-2)
    best_acc = 0
    best_ind_acc = 0
    EPOCH = 200

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()
        for seq, pos, hf, label in train_iter:
            output, _ = net(seq, pos, hf)
            loss = reg_loss(net, output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, train_rep_list, train_label_real, train_true_labels = evaluate(
                train_iter, net, epoch)
            test_performance, test_roc_data, test_prc_data, test_rep_list, test_label_real, test_true_labels = evaluate(
                test_iter, net, epoch)
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[2], test_performance[1], test_performance[3],
            test_performance[4])
        print(results)
        test_acc = test_performance[0]
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_performance
            filename = '{}, {}[{:.4f}].pt'.format('H_A_Model' + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
            save_path_pt = os.path.join('file', filename)
            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[2], best_performance[1], best_performance[3],
                best_performance[4]) + '\n' + '=' * 60
            best_ROC = test_roc_data
            best_PRC = test_prc_data

    return best_performance, best_results, best_ROC, best_PRC


if __name__ == '__main__':
    train_iter = load_bench_data("Train.csv")
    ind_iter = load_Ind_data('Ind.csv')
    performance, result_bench, roc_data, prc_data = train_test(train_iter, ind_iter)