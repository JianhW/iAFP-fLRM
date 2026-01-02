import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import Net
from egaac import Net1
from Paac import Net2
from trans_model import Net3
from pos import Net4



def plot_pr_curve(model_name,model, test_loader, device):
    """
    绘制PR曲线

    参数:
        model: 训练好的PyTorch模型
        test_loader: 测试数据加载器
        device: 计算设备(cpu/cuda)
        model_name: 模型名称(用于图例)
    """
    y_true = []
    y_scores = []

    with torch.no_grad():
        for x, pos, hf, y in test_loader:
            outputs, rep = model(x, pos, hf)
            y=y.to(device)
            # 假设是二分类问题，获取正类的概率
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(y.cpu().numpy())
            y_scores.extend(probabilities.cpu().numpy())

    # 计算PR曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    # 绘制曲线
    plt.plot(recall, precision, lw=2,
             label=f'{model_name}_MLP (AP = {average_precision:.4f})')


    # 添加随机模型的参考线
    #no_skill = sum(y_true) / len(y_true)  # 正样本比例
    #plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='red',label='Random Classifier')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
egaac_model = torch.load('BLO_cnn.pth')
blou_model = torch.load('egaac_mlp.pth')
paac_model = torch.load('paac_resmlp.pth')
trans_model= torch.load('trans.pth')
pos_model = torch.load('pos_cnn.pth')
# 准备模型字典
models_dict = {
    "EGAAC": egaac_model,
    "BLOSUM62": blou_model,
    "PAAC": paac_model,
    "Transformers":trans_model,
    "word_embedding":pos_model
}

# 准备测试数据加载器
with open('test_loader_config.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

loaded_dataset = loaded_data['dataset']
loaded_loader = DataLoader(loaded_dataset, batch_size=64,shuffle=True)
with open('trans_test_loader_config.pkl', 'rb') as f1:
    loaded_data1 = pickle.load(f1)

loaded_dataset1 = loaded_data1['dataset']
loaded_loader1 = DataLoader(loaded_dataset1, batch_size=64,shuffle=True)

with open('egaac_test_loader_config.pkl', 'rb') as f2:
    loaded_data2 = pickle.load(f2)

loaded_dataset2 = loaded_data2['dataset']
loaded_loader2 = DataLoader(loaded_dataset2, batch_size=64,shuffle=True)
with open('pos_test_loader_config.pkl', 'rb') as f3:
    loaded_data3 = pickle.load(f3)

loaded_dataset3 = loaded_data3['dataset']
loaded_loader3 = DataLoader(loaded_dataset3, batch_size=64,shuffle=True)
with open('paac_test_loader_config.pkl', 'rb') as f4:
    loaded_data4 = pickle.load(f4)

loaded_dataset4 = loaded_data4['dataset']
loaded_loader4 = DataLoader(loaded_dataset4, batch_size=64,shuffle=True)

plt.figure(figsize=(8, 6))
plot_pr_curve("EGAAC", egaac_model, loaded_loader, device)
plot_pr_curve("BLOSUM62",blou_model, loaded_loader2, device)
plot_pr_curve("PAAC", paac_model, loaded_loader4, device)
plot_pr_curve("word_embedding", pos_model, loaded_loader3, device)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

# 设置坐标轴范围
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')



plt.legend(loc="lower left")
plt.grid(True)
plt.show()
