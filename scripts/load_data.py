import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import os
def LoadTask2Data(file_path)->tuple[torch.Tensor, torch.Tensor]:
    '''
    加载任务2的数据集
    '''
    #读取原始xlsx文件
    data = pd.read_excel(file_path, sheet_name=None)
    #将其中X1-X8提取成输入
    inputs = data["sheet1"][['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
    #将Y1-Y2提取成输出
    outputs = data["sheet1"][['Y1', 'Y2']].values
    #转换成torch的tensor格式
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    outputs_tensor = torch.tensor(outputs, dtype=torch.float32)
    return inputs_tensor, outputs_tensor
class DataBatcher:
    def __init__(self, file_path: str, val_ratio: float = 0.2,shuffle: bool = True,device: torch.device = torch.device('cpu')):
        """
        数据批处理工具类，支持划分验证集和生成批次数据
        """
        # self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        # 加载原始数据
        self.inputs, self.outputs = LoadTask2Data(file_path)
        
        # 划分训练集和验证集并转换为批次张量
        train_data, val_data = self._splitAndCreateBatches(val_ratio)
        self.train_inputs, self.train_outputs = train_data
        self.val_inputs, self.val_outputs = val_data
    def _splitAndCreateBatches(self, val_ratio: float):   
        dataset = TensorDataset(self.inputs, self.outputs)
        total_size = len(dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        # 固定随机种子确保划分一致
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=generator
        )
        
        # 生成训练集批次（直接返回二维张量 [总样本数, 特征数]）
        # 注意：这里不再按批次拆分，而是返回完整数据集（训练时用DataLoader动态批处理）
        train_inputs = torch.stack([x[0] for x in train_dataset])
        train_outputs = torch.stack([x[1] for x in train_dataset])
        
        # 生成验证集批次
        val_inputs = torch.stack([x[0] for x in val_dataset])
        val_outputs = torch.stack([x[1] for x in val_dataset])
        
        return (train_inputs, train_outputs), (val_inputs, val_outputs)
    
    def getTrainBatches(self):
        """获取训练集数据 (shape: [样本数, 特征数])"""
        return self.train_inputs.to(self.device), self.train_outputs.to(self.device)
    
    def getValBatches(self):
        """获取验证集数据 (shape: [样本数, 特征数])"""
        return self.val_inputs.to(self.device), self.val_outputs.to(self.device)

if __name__ == "__main__":
    # 测试代码
    file_path = os.path.join("/pytorch/data", "task2.xlsx")
    
    # 初始化批处理工具
    batcher = DataBatcher(file_path, val_ratio=0.2)    
    # 获取批次张量
    train_inputs, train_outputs = batcher.getTrainBatches()
    val_inputs, val_outputs = batcher.getValBatches()
    print("训练集输入批次张量形状:", train_inputs.shape)
    print("训练集输出批次张量形状:", train_outputs.shape)