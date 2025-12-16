import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetCentricAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 定义 Q, K, V 的线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
    def forward(self, x, scale=True):
        # x shape: [batch_size, seq_len, d_model]
        
        # 1. 生成 Q, K, V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        d_k = k.size(-1)
        
        # 2. 计算相似度得分 (QK^T)
        scores = torch.matmul(q, k.transpose(-2, -1))
        if scale:
            scores = scores / (d_k ** 0.5)
            
        # 3. 投票归一化 (Row-wise Softmax)
        # dim=-1 代表对每一行进行归一化，确保每个 q 发出的票数和为 1
        # weights[b, i, j] = Token i 投给 Token j 的权重
        weights = F.softmax(scores, dim=-1)
        
        # 4. 聚合 (Aggregation) -> W^T * V
        # 我们希望 Output[j] = sum_i (weights[i, j] * v[i])
        # 也就是对于目标 j，它收集了所有 i 投过来的 v[i]
        # 公式: weights^T * v
        # weights.transpose(-2, -1) shape: [batch_size, seq_len(j), seq_len(i)]
        # v shape:                         [batch_size, seq_len(i), dim]
        output = torch.matmul(weights.transpose(-2, -1), v)
        
        return output

# 使用示例
# model = TargetCentricAttention(d_model=512)
# out = model(input_tensor)