
import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool1d
device = torch.device("cuda:0")

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
