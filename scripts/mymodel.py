import torch.nn as nn
import os,torch
class MLPModel(nn.Module):
    """简单的多层感知机模型"""
    
    def __init__(self, input_size, hidden_size, output_size,dropout=0.1, device=torch.device('cpu')):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device=device
        self.to(device)  # 关键：初始化时就把模型移到目标设备
    def forward(self, x:torch.Tensor)->torch.Tensor:
        assert isinstance(x, torch.Tensor),  "输入必须是torch.Tensor类型"
        x = x.to(self.device)  # 确保输入数据在正确的设备上
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
# 修改 mymodel.py 中的 MLPAttention 类
class MLPresidual(nn.Module):
    """带有残差连接的多层感知机模型"""
    
    def __init__(self, input_size, hidden_size, output_size,dropout=0.1, device=torch.device('cpu')):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 升维到hidden_size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)   # 降维回input_size，确保与残差相加
        )
        # 输出层：处理残差后的特征
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.device = device
        self.to(device)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor), "输入必须是torch.Tensor类型"
       
        x = x.to(self.device)
        residual = x  # 保存原始输入作为残差
        # 残差块计算
        x = self.residual_block(x)
        x = x + residual  # 残差连接（核心：当前输出+原始输入）
        # 输出层
        x = self.output_layer(x)
        
        return x