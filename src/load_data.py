import pandas as pd
import torch
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
if __name__ == "__main__":
    #测试代码
    file_path = os.path.join("/home/Elaina/pytorch/data", "task2.xlsx")
    inputs, outputs = LoadTask2Data(file_path)
    print("Inputs shape:", inputs.shape)
    print("Outputs shape:", outputs.shape)