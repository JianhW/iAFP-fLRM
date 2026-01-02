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
device=torch.device("cuda:0")
def load_bench_data(file):#训练集
    with open(file, 'rb') as f:
        raw_data=f.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
    tmp = pd.read_csv(file,encoding=detected_encoding, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, file,train=True)
    data_iter = list(data_iter)
    return data_iter
def load_Ind_data(file):#独立测试集
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, file,train=True)
    data_iter = list(data_iter)
    return data_iter
def data_construct(seqs, labels,file, train):#将氨基酸序列转化为模型所需要的输入
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    longest_num = len(max(seqs, key=len))
    sequences = [i.ljust(longest_num, 'X') for i in seqs]#以最长的氨基酸序列为基准，对其他长度不足的氨基酸序列进行填充
    pep_codes = []
    for pep in seqs:
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep).to(device))
    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True).to(device)  # Fill the sequence to the same length
    pos_embed = np.array(position_encoding(labels,sequences))  # 确保返回的是NumPy数组
    #pos_embed=SVM_RFE(pos_embed,labels)
    HF_feature = HF_encoding(seqs, sequences,labels).detach().numpy()  # 确保返回的是NumPy数组
    dataset = Data.TensorDataset(embed_data, torch.FloatTensor(pos_embed),torch.FloatTensor(HF_feature), torch.LongTensor(labels))
    #print(dataset.shape)
    batch_size = 128
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return data_iter



def BLOSUM62(seqs):
    """
    BLOSUM 是“blocks substitution matrix”的缩写。它是目前常用的一种氨基酸替换打分矩阵
    需要等长的氨基酸序列
    """
    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # X
    }
    encodings = []
    for seq in seqs:
        code = []
        for aa in seq:
            code += blosum62[aa]
        encodings.append(code)
    return encodings

def position_encoding(labels,seqs):#位置编码，transformer模型的步骤之一
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


def HF_encoding(seqs, sequences,labels):
    embed_dim = 256
    num_heads = 8
    dropout = 0.1
    d_model=256
    multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
    result = BLOSUM62(sequences)
    result= torch.tensor(result)
    #print(result.shape)
    linear_layer1=nn.Linear(result.size(-1),d_model)
    result=linear_layer1(result.float())
    #result,_=multihead_attn(result,result,result)
    #print(result.shape)
    #pca_TSNE(result,labels)
    #print(result.shape)
    return result

class Net(nn.Module):
    def __init__(self,num_classes=2):
        super(Net, self).__init__()
        self.conv_hidden_dim = 64
        self.lstm_hidden_dim=32
        self.MLP_embed_dim = 64
        self.MLP_hidden_dim = 64
        self.dropout=0.2
        self.batch_size = 64
        self.emb_dim = 128
        self.embedding_seq = nn.Embedding(24, self.emb_dim, padding_idx=0,device=device)
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
        self.linearlayer=nn.Linear(1,128,device=device)
        self.relu=nn.ReLU()
        self.lstm = nn.LSTM(20, self.lstm_hidden_dim, num_layers=2, bidirectional=True, dropout=self.dropout, batch_first=False,device=device)

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
        self.classifier1=nn.Sequential(
            nn.Linear(self.MLP_hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, num_classes)
        )
        self.cnn_classifier = nn.Sequential(
            # 添加通道维度: [batch, hidden_dim] -> [batch, 1, hidden_dim]
            nn.Unflatten(1, (1, self.MLP_hidden_dim)),

            # 1D卷积层
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化

            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_classes)
        )
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
        #print(output1.size())

        output1 = output1.permute(0, 2, 1)  # 调整为 (batch, 128, seq_len)
        output1 = self.conv_seq(output1)  # 输入形状 (batch, 20, new_seq_len)
        #print(output1.size())
        #print(HF.size())
        output2 = HF.unsqueeze(2)

        output2=output2.to(device)
        output2=self.relu(self.linearlayer(output2))
        output2 = output2.permute(0, 2, 1)
        output2 = self.conv_HF(output2)#(batch,channels,seq)
        output = torch.cat([output1, output2], 2)#(batch,channels,seq)
        output = output.permute(2, 0, 1)  # (seq,batch,channels)
        output,(h,c)= self.lstm(output)
        # print(output.size())
        output = output.permute(1, 0, 2)  # (batch, seq_len, channels)
        #output=self.relu(output)
        #print(output.size())
        identity = output.mean(dim=1)  # (batch, embed_dim)
        #print(identity.size())
        out = self.block1(output.mean(dim=1)) + identity  # (batch, hidden_dim)
        out = self.block2(out) + out  # (batch, hidden_dim)
        #out=self.classifier(out)
        out = self.classifier(out)
        return out, output


def draw_AUC(fpr, tpr,roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def pca_TSNE(features,labels,n_components=50):



    # 中心化
    mean = torch.mean(features, dim=0)
    centered = features - mean
    # 计算协方差矩阵
    cov = torch.matmul(centered.T, centered) / (centered.shape[0]-1)
    # 特征分解
    eigvals, eigvecs = torch.linalg.eigh(cov)
    # 取前n_components个主成分
    components = eigvecs[:, -n_components:]
    features_pca = torch.matmul(centered, components).cpu().detach().numpy()


    labels = torch.tensor(labels)
    # Step 2: t-SNE（需转CPU）
    features = TruncatedSVD(n_components=200).fit_transform(features.detach().numpy())
    tsne = TSNE(
        n_components=2,
        perplexity=45,
        early_exaggeration=48,
        learning_rate='auto',
        metric='cosine',
        n_iter=1000,
        random_state=42
    )
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1],
                c=labels, cmap='tab10', alpha=0.6, s=15)
    plt.colorbar()
    plt.title("pca_t-SNE Visualization (Direct on High-Dim Data)")
    plt.show()
def t_SNE(features,labels):

    labels = torch.tensor(labels)
    features=torch.tensor(features)
    #print(features.shape)
    features = torch.mean(features, dim=1).detach().numpy()
    #features = TruncatedSVD(n_components=200).fit_transform(features.detach().numpy())

    tsne = TSNE(
        n_components=2,
        perplexity=45,
        early_exaggeration=48,
        learning_rate='auto',
        metric='cosine',
        n_iter=1000,
        random_state=42
    )
    features_2d = tsne.fit_transform(features)

    # 可视化
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1],
                c=labels, cmap='tab10', alpha=0.6, s=15)
    plt.colorbar()
    plt.title("t-SNE Visualization (Direct on High-Dim Data)")
    plt.show()
def evaluate(data_iter, net,epoch):
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    true_labels = []
    for x, pos, hf, y in data_iter:
        outputs,rep = net(x, pos, hf)
        #print(rep.shape)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y.tolist()
        label=y.cpu()
        rep=rep.cpu()
        rep_list.append(rep.detach().numpy())
        true_labels.append(label.detach().numpy())
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real,epoch)
    return performance, roc_data, prc_data,rep_list,label_real,true_labels
def caculate_metric(pred_prob, label_pred, label_real,epoch):
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
    # Accuracy
    ACC = float(tp + tn) / test_num
    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)
    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)
    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)
    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data
def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion=criterion.to(device)
    l2_lambda = 0.0
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)
    total_regularization_loss=l2_lambda * regularization_loss
    total_loss = criterion(output.to(device), label.to(device)) + total_regularization_loss.to(device)
    return total_loss



def train_test(train_iter, test_iter):
    net = Net()
    net=net.to(device)
    lr = 0.002
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=1e-2)
    best_acc = 0
    best_ind_acc = 0
    EPOCH = 200
    #writer=SummaryWriter("../logs_train")

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()
        net.train()
        for seq, pos, hf, label in train_iter:
            output,_ = net(seq, pos, hf)
            loss = reg_loss(net, output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
        #writer.add_scalar("loss",np.mean(loss_ls),epoch)
        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data,train_rep_list,train_label_real,train_true_labels = evaluate(train_iter, net,epoch)
            test_performance, test_roc_data, test_prc_data,test_rep_list,test_label_real,test_true_labels = evaluate(test_iter, net,epoch)
        #writer.add_scalar("test_loss", np.mean(test_loss_ls), epoch)
        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Test Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[2], test_performance[1], test_performance[3],
            test_performance[4])
        print(results)
        #print(ind_performance)
        test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        if test_acc > best_acc:
            best_acc = test_acc
            best_performance = test_performance
            filename = '{}, {}[{:.4f}].pt'.format('H_A_Model' + ', epoch[{}]'.format(epoch + 1), 'ACC', best_acc)
            save_path_pt = os.path.join('file', filename)
            #torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[2], best_performance[1], best_performance[3],
                best_performance[4]) + '\n' + '=' * 60
            # print(best_results)
            best_ROC = test_roc_data
            best_PRC = test_prc_data
        if (epoch+1)%50==0:
            torch.save(net,"BLO_mlp.pth".format(epoch+1))
    #writer.close()
    return best_performance, best_results, best_ROC, best_PRC



def K_CV(file, ind_iter, k):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = np.array(tmp[0].values.tolist()), np.array(tmp[1].values.tolist())
    data_iter = data_construct(seqs, labels, train=True)
    data_iter = list(data_iter)
    CV_perform = []

    for iter_k in range(k):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)
        train_iter = [x for i, x in enumerate(data_iter) if i % k != iter_k]
        test_iter = [x for i, x in enumerate(data_iter) if i % k == iter_k]
        performance, _, ROC, PRC = train_test(train_iter, test_iter, ind_iter)
        print(performance)
        CV_perform.append(performance)
    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ',
                                    'red') + '=' * 16 + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n')
    for out in np.array(CV_perform):
        print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(out[0], out[2], out[1], out[3], out[4]))
    mean_out = np.array(CV_perform).mean(axis=0)
    print('\n' + '=' * 16 + "Mean out" + '=' * 16)
    print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(mean_out[0], mean_out[2], mean_out[1], mean_out[3],
                                                              mean_out[4]))
    print('\n' + '=' * 60)
def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model



if __name__ == '__main__':
    train_iter = load_bench_data("train9.csv")
    ind_iter = load_Ind_data('ds1.csv')
    performance, result_bench,roc_data,prc_data = train_test(train_iter, ind_iter)

    #print(train_iter.shape)
    #test_iter = load_ind_data("test.csv")
    #print(test_iter.shape)
