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
    def __init__(self, file_path: str, val_ratio: float = 0.2, batch_size: int = 32, shuffle: bool = True,device: torch.device = torch.device('cpu')):
        """
        数据批处理工具类，支持划分验证集和生成批次数据
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        # 加载原始数据
        self.inputs, self.outputs = LoadTask2Data(file_path)
        
        # 划分训练集和验证集并转换为批次张量
        train_data, val_data = self._splitAndCreateBatches(val_ratio)
        self.train_inputs, self.train_outputs = train_data
        self.val_inputs, self.val_outputs = val_data
        
    def _splitAndCreateBatches(self, val_ratio: float):
        """划分训练集和验证集并转换为批次张量"""
        # 创建数据集并划分
        dataset = TensorDataset(self.inputs, self.outputs)
        total_size = len(dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        
        # 划分训练集和验证集
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=generator
        )
        
        # 生成训练集批次张量（过滤掉最后一个不足batch_size的批次）
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        train_input_batches = []
        train_output_batches = []
        for batch in train_loader:
            # 只保留大小等于batch_size的批次
            if batch[0].shape[0] == self.batch_size:
                train_input_batches.append(batch[0])
                train_output_batches.append(batch[1])
        # 如果所有批次都被过滤了，至少保留一个
        if not train_input_batches:
            train_input_batches.append(train_loader.dataset[:self.batch_size][0])
            train_output_batches.append(train_loader.dataset[:self.batch_size][1])
        train_input_batches = torch.stack(train_input_batches)
        train_output_batches = torch.stack(train_output_batches)
        
        # 生成验证集批次张量（同样处理）
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        val_input_batches = []
        val_output_batches = []
        for batch in val_loader:
            if batch[0].shape[0] == self.batch_size:
                val_input_batches.append(batch[0])
                val_output_batches.append(batch[1])
        if not val_input_batches:
            val_input_batches.append(val_loader.dataset[:self.batch_size][0])
            val_output_batches.append(val_loader.dataset[:self.batch_size][1])
        val_input_batches = torch.stack(val_input_batches)
        val_output_batches = torch.stack(val_output_batches)
        
        return (train_input_batches, train_output_batches), (val_input_batches, val_output_batches)
    
    def getTrainBatches(self):
        """获取训练集批次张量 (输入, 输出)"""
        return self.train_inputs.to(self.device), self.train_outputs.to(self.device)
    
    def getValBatches(self):
        """获取验证集批次张量 (输入, 输出)"""
        return self.val_inputs.to(self.device), self.val_outputs.to(self.device)
    

if __name__ == "__main__":
    # 测试代码
    file_path = os.path.join("/pytorch/data", "task2.xlsx")
    
    # 初始化批处理工具
    batcher = DataBatcher(file_path, val_ratio=0.2, batch_size=16)    
    # 获取批次张量
    train_inputs, train_outputs = batcher.getTrainBatches()
    val_inputs, val_outputs = batcher.getValBatches()
    print("训练集输入批次张量形状:", train_inputs.shape)
    print("训练集输出批次张量形状:", train_outputs.shape)