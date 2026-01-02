import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from typing import List
import logging

# 设置日志（可选）



class ProtT5FeatureExtractor:
    def __init__(self):
        """
        初始化 ProtT5-Base 模型和分词器
        Args:
            device: 指定设备 ("cuda" 或 "cpu")，若为 None 则自动选择
        """
        self.device=torch.device("cuda:0")
        self.model_name = "Rostlab/prot_t5_base"

        # 加载模型和分词器

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()  # 设置为评估模式
    def embed_sequences(self, sequences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量提取蛋白质序列的特征
        Args:
            sequences: 蛋白质序列列表 (e.g., ["ACDEF", "GHIKL..."])
            batch_size: 批处理大小（根据显存调整）
        Returns:
            np.ndarray: 形状为 [num_sequences, 768] 的特征矩阵
        """
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]

            # 预处理批次数据
            batch_sequences = [" ".join(list(seq)) for seq in batch]
            inputs = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding="longest",  # 动态填充到批次内最长序列
                truncation=True,  # 截断超过512的序列
                max_length=512,
            ).to(self.device)

            # 特征提取
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 全局平均池化 (得到序列级表示)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.append(batch_embeddings)

            logger.info(f"Processed batch {i // batch_size + 1}/{(len(sequences) // batch_size) + 1}")

        return np.concatenate(all_embeddings, axis=0)


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 示例蛋白质序列
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",  # 示例1
        "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # 示例2 (AMP)
        "ARNDCEQGHILKMFPSTWYV",  # 示例3
    ]

    # 初始化特征提取器
    extractor = ProtT5FeatureExtractor()

    # 批量提取特征
    embeddings = extractor.embed_sequences(sequences, batch_size=2)

    # 输出结果
    print(f"Embeddings shape: {embeddings.shape}")  # 应为 (3, 768)
    print(f"First sequence embedding (first 5 dims): {embeddings[0][:5]}")