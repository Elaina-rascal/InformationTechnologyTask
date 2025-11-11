import torch.nn as nn
import os,torch
class MLPModel(nn.Module):
    """简单的多层感知机模型"""
    
    def __init__(self, input_size, hidden_size, output_size, device=torch.device('cpu')):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.device=device
        self.to(device)  # 关键：初始化时就把模型移到目标设备
    def forward(self, x:torch.Tensor):
        assert isinstance(x, torch.Tensor),  "输入必须是torch.Tensor类型"
        x = x.to(self.device)  # 确保输入数据在正确的设备上
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x