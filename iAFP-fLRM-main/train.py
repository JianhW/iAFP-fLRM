
import os
import numpy as np
import torch
import torch.nn as nn
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from model import Net
device = torch.device("cuda:0")



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
