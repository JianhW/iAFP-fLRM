from model import load_ind_data, Net, load_model, evaluate
import torch
from termcolor import colored
import numpy as np
def predict(file):
    data_iter = load_ind_data(file)
    model = Net()
    model=model.cuda()
    path_pretrain_model = "net_100.pth"
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        ind_performance, ind_roc_data, ind_prc_data,_,_ = evaluate(data_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Performance', 'red') + '=' * 16 \
                   + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            ind_performance[0], ind_performance[2], ind_performance[1], ind_performance[3],
            ind_performance[4]) + '\n' + '=' * 60
    # np.save("实验结果/ROC-PRC/M_M/M_M_Fusion_roc1.npy", ind_roc_data)
    # np.save("实验结果/ROC-PRC/M_M/M_M_Fusion_prc1.npy", ind_prc_data)
    return ind_results
ind_result = predict('Ind.csv')
print(ind_result)