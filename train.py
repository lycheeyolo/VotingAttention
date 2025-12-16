import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models_target_transformer import TargetCentricTransformerModel


class RandomSequenceDataset(Dataset):
    """
    一个简单的玩具数据集，用随机张量演示训练流程。
    你可以用自己的任务数据集替换这里：
    - 将 __getitem__ 返回的 inputs/targets 改成真实数据。
    """

    def __init__(self, num_samples: int, seq_len: int, d_model: int):
        super().__init__()
        self.inputs = torch.randn(num_samples, seq_len, d_model)
        # 这里构造一个简单的回归目标：对序列做平均池化后回归到一个标量
        self.targets = self.inputs.mean(dim=(1, 2), keepdim=True)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SimpleRegressionHead(nn.Module):
    """
    示例任务头：将编码后的序列特征做池化并回归到一个标量。
    你可以按需改成分类头、token 级别预测头等。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # encoded: [batch_size, seq_len, d_model]
        pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        out = self.proj(pooled)  # [batch_size, 1]
        return out


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    通用损失函数占位示例。
    根据你的任务修改这里，比如：
    - 分类: nn.CrossEntropyLoss
    - 回归: nn.MSELoss
    """
    return nn.MSELoss()(pred, target)


def main():
    # 一些示例超参数，你可以根据任务调整
    d_model = 64
    num_layers = 2
    num_heads = 4
    dim_feedforward = 128
    dropout = 0.1
    seq_len = 16
    batch_size = 8
    num_epochs = 30
    num_samples = 128

    # 单卡 / 多卡设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using CUDA, num_gpus = {num_gpus}")
    else:
        device = torch.device("cpu")
        num_gpus = 0

    # 1. 构造数据集和 DataLoader（这里使用随机数据）
    dataset = RandomSequenceDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        d_model=d_model,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 构造模型（Encoder-only 示例）
    model = TargetCentricTransformerModel(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        vocab_size=None,  # 这里输入已经是连续特征，不需要 Embedding
        use_decoder=False,
    )
    head = SimpleRegressionHead(d_model=d_model)

    # 多卡：使用 DataParallel 包裹模型和任务头
    if num_gpus > 1:
        model = nn.DataParallel(model)
        head = nn.DataParallel(head)

    model.to(device)
    head.to(device)

    # 3. 优化器
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=1e-3,
    )

    # 4. 训练循环
    model.train()
    head.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Encoder-only 前向：返回编码后的序列特征
            encoded = model(src=inputs)
            preds = head(encoded)
            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, loss = {avg_loss:.4f}")


if __name__ == "__main__":
    main()


