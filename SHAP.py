
from model import load_bench_data, Net, load_Ind_data
import shap
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shap
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class SafeShapWrapper(torch.nn.Module):
    """完全安全的模型包装器，处理所有维度问题"""

    def __init__(self, model, seq_len=128, hf_dim=256):
        super().__init__()
        self.model = model
        self.seq_len = seq_len
        self.hf_dim = hf_dim

    def _safe_adjust(self, x, expected_dim):
        """安全调整输入维度"""
        x = np.array(x)

        # 处理序列输入
        if expected_dim == "seq":
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.shape[1] > self.seq_len:
                x = x[:, :self.seq_len]
            elif x.shape[1] < self.seq_len:
                x = np.pad(x, ((0, 0), (0, self.seq_len - x.shape[1])))
            return x

        # 处理HF特征输入
        elif expected_dim == "hf":
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.shape[1] > self.hf_dim:
                x = x[:, :self.hf_dim]
            elif x.shape[1] < self.hf_dim:
                x = np.pad(x, ((0, 0), (0, self.hf_dim - x.shape[1])))
            return x

        # 处理位置编码
        elif expected_dim == "pos":
            if x.ndim == 1:
                x = x.reshape(1, -1, 1)
            elif x.ndim == 2:
                x = x.reshape(1, x.shape[0], x.shape[1])
            return x

    def forward(self, *args):
        try:
            # 统一输入处理
            inputs = args[0] if isinstance(args[0], list) else args

            # 安全转换各输入
            seq = torch.tensor(
                self._safe_adjust(inputs[0], "seq"),
                device=self.model.device
            ).long()

            pos = torch.tensor(
                self._safe_adjust(inputs[1], "pos"),
                device=self.model.device
            ).float()

            hf = torch.tensor(
                self._safe_adjust(inputs[2], "hf"),
                device=self.model.device
            ).float()

            # 维度验证
            print(f"Input shapes - seq: {seq.shape}, pos: {pos.shape}, hf: {hf.shape}")  # 调试

            with torch.no_grad():
                outputs, _ = self.model(seq, pos, hf)
                return torch.softmax(outputs, dim=1).cpu().numpy()

        except Exception as e:
            print(f"Forward error: {str(e)}")
            raise


def safe_shap_analysis(model, data_iter, device):
    """完全安全的SHAP分析流程"""
    try:
        model.eval()
        model.to(device)

        # 1. 确定模型参数
        SEQ_LEN = 128  # 必须与data_construct()中的填充长度一致
        HF_DIM = 256  # 必须与HF_encoding的输出维度一致
        POS_DIM = 128  # 必须与position_encoding的输出维度一致

        # 2. 准备背景数据 (单个样本)
        background = []
        for x, pos, hf, _ in data_iter:
            background = (
                x[:1].cpu().numpy(),  # 取第一个样本
                pos[:1].cpu().numpy(),
                hf[:1].cpu().numpy()
            )
            break

        # 3. 准备解释数据 (3个样本)
        explanation_data = []
        labels = []
        for i, (x, pos, hf, y) in enumerate(data_iter):
            if i >= 3:
                break
            explanation_data.append((
                x.cpu().numpy(),
                pos.cpu().numpy(),
                hf.cpu().numpy()
            ))
            labels.append(y.cpu().numpy())

        # 4. 创建安全包装器
        wrapped_model = SafeShapWrapper(model, SEQ_LEN, HF_DIM).to(device)

        # 5. 创建解释器
        explainer = shap.DeepExplainer(
            model=wrapped_model,
            data=[background[0][0], background[1][0], background[2][0]]  # 使用第一个样本
        )

        # 6. 安全计算SHAP值
        shap_values = []
        for i in range(3):  # 只计算3个样本
            try:
                sample = [
                    explanation_data[i][0],
                    explanation_data[i][1],
                    explanation_data[i][2]
                ]
                current_shap = explainer.shap_values(sample, check_additivity=False)
                shap_values.append(current_shap)
                print(f"Sample {i} SHAP computed successfully")
            except Exception as e:
                print(f"Error on sample {i}: {str(e)}")

        # 7. 安全可视化
        if shap_values:
            plot_results(shap_values, explanation_data, SEQ_LEN)
        else:
            print("No valid SHAP values to plot")

    except Exception as e:
        print(f"Fatal error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_results(shap_values, data, seq_len):
    """安全可视化函数"""
    try:
        # 提取正类SHAP值
        positive_shap = []
        for sv in shap_values:
            if len(sv) > 1:  # 检查多分类
                shap_array = sv[1]  # 正类SHAP值
                if isinstance(shap_array, list):
                    shap_array = shap_array[0]  # 取序列部分

                # 调整形状
                if shap_array.shape[1] < seq_len:
                    shap_array = np.pad(shap_array, ((0, 0), (0, seq_len - shap_array.shape[1])))
                elif shap_array.shape[1] > seq_len:
                    shap_array = shap_array[:, :seq_len]

                positive_shap.append(shap_array)

        if not positive_shap:
            raise ValueError("No valid SHAP values")

        positive_shap = np.concatenate(positive_shap)
        seq_data = np.concatenate([d[0] for d in data[:len(positive_shap)]])

        print(f"Final SHAP shape: {positive_shap.shape}, Data shape: {seq_data.shape}")

        # 绘制结果
        plt.figure(figsize=(15, 8))
        shap.summary_plot(
            positive_shap,
            features=seq_data,
            feature_names=[f"Pos_{i + 1}" for i in range(seq_len)],
            show=False
        )
        plt.title("SHAP Values (Safe Version)", pad=20)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Plotting error: {str(e)}")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 加载模型
        model = Net().to(device)
        model.load_state_dict(torch.load('net_200.pth', map_location=device))

        # 加载数据
        train_iter = load_bench_data("Train.csv")
        ind_iter = load_Ind_data('Ind.csv')

        # 执行安全分析
        print("Running ultra-safe SHAP analysis...")
        safe_shap_analysis(model, ind_iter, device)

    except Exception as e:
        print(f"Initialization failed: {str(e)}")