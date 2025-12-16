import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadTargetCentricAttention(nn.Module):
    """
    多头版本的 Target-Centric Attention。

    与标准 MultiHeadAttention 的接口尽量保持一致：
    - 输入: x [batch_size, seq_len, d_model]
    - 可选 attention_mask: [batch_size, 1, seq_len, seq_len] 或 [batch_size, seq_len, seq_len]
      mask 位置为 True / 1 表示要屏蔽（不可见），将对应 logits 置为 -inf。
    - 输出: [batch_size, seq_len, d_model]

    注意聚合方式使用 W^T V（被投票者聚合所有投票者的 V）。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: [batch_size, seq_len_q, d_model]，作为投票者序列
        :param attention_mask:
            - None，或者
            - [batch_size, seq_len_q, seq_len_k]
            - [batch_size, 1, seq_len_q, seq_len_k]
          值为 True / 1 的位置会被 mask 掉。
        :param kv: 作为候选人序列的表示（用于 K/V），若为 None，则退化为自注意力，使用 x。
        :return: [batch_size, seq_len_k, d_model]（被投票者 / 候选人新的表示）

        说明:
        - 自注意力: x 既是投票者也是候选人，此时输出长度与输入相同。
        - Cross-Attention: x 为投票者，kv 为候选人；但聚合依然是 W^T V，
          即每个候选人聚合同一批投票者的 V。
        """
        if kv is None:
            kv = x

        bsz, seq_len_q, _ = x.size()
        seq_len_k = kv.size(1)

        # 1. 生成 Q, K, V 并拆分为多头
        q = self.w_q(x)  # [B, L_q, D]
        k = self.w_k(kv)  # [B, L_k, D]
        v = self.w_v(kv)  # [B, L_k, D]

        # 拆成多头
        def split_heads(t, seq_len):
            return t.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        q = split_heads(q, seq_len_q)  # [B, H, L_q, d_head]
        k = split_heads(k, seq_len_k)  # [B, H, L_k, d_head]
        v = split_heads(v, seq_len_k)  # [B, H, L_k, d_head]

        # 2. 计算相似度得分 (QK^T)  -> [B, H, L_q, L_k]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_head)

        # 3. 应用 attention_mask（行 softmax 前）
        if attention_mask is not None:
            # 将 mask 形状扩展到 [B, 1, L_q, L_k] 再广播到多头
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask.bool(), float("-inf"))

        # 4. 投票归一化 (Row-wise Softmax)
        # 对每个投票者（行）进行归一化
        weights = F.softmax(scores, dim=-1)  # [B, H, L_q(i), L_k(j)]
        weights = self.dropout(weights)

        # 5. 聚合: W^T * V
        # 我们希望 Output[j] = sum_i (weights[i, j] * v[i])
        # 对应矩阵形式: weights^T * V
        # weights.transpose(-2, -1): [B, H, L_k(j), L_q(i)]
        # v: [B, H, L_k(i), d_head]
        # 为了与自注意力保持一致，这里仍然按 "候选人维" L_k 来输出。
        output = torch.matmul(weights.transpose(-2, -1), v)  # [B, H, L_k, d_head]

        # 6. 合并多头
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len_k, self.d_model)
        output = self.w_o(output)  # [B, L_k, D]
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TargetCentricTransformerBlock(nn.Module):
    """
    与标准 Transformer Encoder Block 类似，但将自注意力替换为 Target-Centric Attention。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadTargetCentricAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.linear1 = PositionwiseFeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param src: [batch_size, seq_len, d_model]
        :param src_mask: [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        """
        # 自注意力子层（投票者 = 候选人 = src）
        attn_output = self.self_attn(src, attention_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # 前馈子层
        ff_output = self.linear1(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class PositionalEncoding(nn.Module):
    """
    标准正余弦位置编码实现，接口与 PyTorch 官方教程类似。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TargetCentricEncoder(nn.Module):
    """
    多层 Target-Centric Transformer Encoder。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TargetCentricTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param src: [batch_size, seq_len, d_model]
        :param src_mask: [batch_size, seq_len, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask)
        output = self.norm(output)
        return output


class TargetCentricDecoder(nn.Module):
    """
    可选的 Decoder 骨架，示意如何在 cross-attention 中复用 Target-Centric 注意力。
    你可以根据具体任务补充/修改此结构。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": MultiHeadTargetCentricAttention(
                            d_model=d_model,
                            num_heads=num_heads,
                            dropout=dropout,
                        ),
                        "cross_attn": MultiHeadTargetCentricAttention(
                            d_model=d_model,
                            num_heads=num_heads,
                            dropout=dropout,
                        ),
                        "ffn": PositionwiseFeedForward(
                            d_model=d_model,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                        ),
                        "norm1": nn.LayerNorm(d_model),
                        "norm2": nn.LayerNorm(d_model),
                        "norm3": nn.LayerNorm(d_model),
                        "drop1": nn.Dropout(dropout),
                        "drop2": nn.Dropout(dropout),
                        "drop3": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param tgt: [batch_size, tgt_len, d_model]
        :param memory: [batch_size, src_len, d_model] 编码器输出
        """
        output = tgt
        for layer in self.layers:
            # 1) 自注意力
            sa = layer["self_attn"](output, attention_mask=tgt_mask)
            output = output + layer["drop1"](sa)
            output = layer["norm1"](output)

            # 2) Cross-Attention: 投票者 = tgt，候选人 = memory
            ca = layer["cross_attn"](output, attention_mask=memory_mask, kv=memory)
            output = output + layer["drop2"](ca)
            output = layer["norm2"](output)

            # 3) FFN
            ff = layer["ffn"](output)
            output = output + layer["drop3"](ff)
            output = layer["norm3"](output)

        output = self.norm(output)
        return output


class TargetCentricTransformerModel(nn.Module):
    """
    通用的 Target-Centric Transformer 模型封装。

    - 如果提供 vocab_size，则内部包含 Embedding + PositionalEncoding；
    - 否则假定输入已经是 [batch_size, seq_len, d_model] 的连续特征。

    该模型只返回编码后的序列特征，具体任务头和 loss 由外部决定。
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        vocab_size: Optional[int] = None,
        max_len: int = 5000,
        use_decoder: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_decoder = use_decoder

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = None

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        self.encoder = TargetCentricEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        if use_decoder:
            self.decoder = TargetCentricDecoder(
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            self.decoder = None

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.embedding is not None and src.dtype == torch.long:
            src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.encoder(src, src_mask=src_mask)
        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("当前模型未启用 Decoder（use_decoder=False）")
        if self.embedding is not None and tgt.dtype == torch.long:
            tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return output

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        通用前向接口：
        - Encoder-only: 只传入 src（以及可选 src_mask），返回 encoder 输出；
        - Encoder-Decoder: 传入 src 和 tgt，返回 decoder 输出。
        """
        memory = self.encode(src, src_mask=src_mask)
        if not self.use_decoder:
            return memory
        if tgt is None:
            raise ValueError("use_decoder=True 时必须提供 tgt")
        output = self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return output



