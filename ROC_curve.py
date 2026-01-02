from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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








def plot_multiple_roc_curves(model_name,model, test_loader, device):
    """
    在同一坐标系中绘制多条ROC曲线

    参数:
        models_dict: 字典 {模型名称: 模型实例}
        test_loader: 测试数据加载器
        device: 计算设备(cpu/cuda)
    """
    #plt.figure(figsize=(8, 6))

    # 为每个模型计算并绘制ROC曲线
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    true_labels = []
    with torch.no_grad():
        for x, pos, hf, y in test_loader:
            outputs, rep = model(x, pos, hf)
            pred_prob_positive = outputs[:, 1]
            pred_prob = pred_prob + pred_prob_positive.tolist()
            label_pred = label_pred + outputs.argmax(dim=1).tolist()
            label_real = label_real + y.tolist()

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(label_real, pred_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # 绘制当前模型的ROC曲线
        plt.plot(fpr, tpr, lw=2,
                 label=f'{model_name} _ResMLP(_AUC = {roc_auc:.4f})')



# 假设你已经有多个训练好的模型
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
plot_multiple_roc_curves("EGAAC", egaac_model, loaded_loader, device)
plot_multiple_roc_curves("BLOSUM62",blou_model, loaded_loader2, device)
plot_multiple_roc_curves("PAAC", paac_model, loaded_loader4, device)
plot_multiple_roc_curves("word_embedding", pos_model, loaded_loader3, device)
# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

