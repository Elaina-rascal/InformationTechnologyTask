from visual import *
from model import *
from load_data import *
def infer():
    #初始化参数
    model_path='/pytorch/model/best_transformer.pth'
    device=torch.device('cuda' if torch.cuda.is_available() else'cpu')
    #加载模型
    model=MLPModel(input_size=8,hidden_size=24,output_size=2,device=device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    #加载数据
    data_path='/pytorch/data/task2.xlsx'
    batcher = DataBatcher(file_path=data_path, val_ratio=0.2, batch_size=16,device=device)
    val_inputs, val_outputs = batcher.getValBatches()
    #进行推理
    output=model(val_inputs)
    #取第1batch的结果比较
    print("真实值:", val_outputs[0])
    print("预测值:", output[0])
    # 计算均方误差
    loss_fn=torch.nn.MSELoss()
    loss=loss_fn(output,val_outputs)
    print(f"验证集均方误差: {loss.item():.4f}")
if __name__ == "__main__":
    infer()