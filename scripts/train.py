from visual import *
from mymodel import *
from load_data import *
def train():
    device=torch.device('cuda' if torch.cuda.is_available() else'cpu')
    model=MLPresidual(input_size=8,hidden_size=24,output_size=2,dropout=0.1,device=device)
    # model=MLPAttention(input_size=8,attention_hidden_size=8,hidden_size=32,output_size=2,device=device)
    data_path='/pytorch/data/task2.xlsx'
    model_path='/pytorch/model/'
    batcher = DataBatcher(file_path=data_path, val_ratio=0.2, batch_size=16,device=device)
    train_inputs, train_outputs = batcher.getTrainBatches()
    val_inputs, val_outputs = batcher.getValBatches()
    #平方损失
    loss_fn=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

    visual=SaveAndVisual(model_dir=model_path, loss_img_path=model_path+'loss_curve.png')
    num_epoch=600
    visual.loadModel(model,optimizer,device)
    for epoch in range(num_epoch):
        model.train()
        inputs=train_inputs
        targets=train_outputs
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        visual.updateVisualization(epoch,loss.item())
    visual.finalizeVisualization()

if __name__ == "__main__":
    train()