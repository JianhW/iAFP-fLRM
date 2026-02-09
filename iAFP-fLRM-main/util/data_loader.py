import sys, os, re
pPath = os.path.split(os.path.realpath(sys.argv[0]))[0]
sys.path.append(pPath)
pPath = re.sub(r'codes$', '', os.path.split(os.path.realpath(sys.argv[0]))[0])
sys.path.append(pPath)
import numpy as np
import chardet
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
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
    d_model = 256
    result = BLOSUM62(sequences)
    result = torch.tensor(result)
    linear_layer1 = nn.Linear(result.size(-1), d_model)
    result = linear_layer1(result.float())
    return result