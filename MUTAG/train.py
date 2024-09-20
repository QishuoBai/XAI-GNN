import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import MutagGCN
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

# 自定义数据集类
class MutagDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

def collate_fn(batch):
    graphs, labels = zip(*batch)
    labels = th.tensor(labels).long()
    return dgl.batch(graphs), labels

if __name__ == '__main__':
    graphs = dgl.load_graphs('./data/graphs.dgl')[0]
    graph_labels = th.load('./data/graph_labels.pth')
    # 创建自定义数据集
    dataset = MutagDataset(graphs, graph_labels)
    train_ratio = 0.8
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MutagGCN()
    # train
    loss_function = nn.CrossEntropyLoss(weight=th.tensor([1.3, 1.0]))
    # loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    Epoch = 100
    loss_list = []
    for epoch in tqdm(range(Epoch)):
        model.train()
        cur_epoch_loss = 0
        for x, y in train_loader:
            pred = model(x, train_batch_size)
            # print(pred)
            # print(y)
            loss = loss_function(pred, y)
            cur_epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        total_correct = 0
        val_data_1_ratio = 0
        for x, y in val_loader:
            pred = model(x, 1).squeeze()
            pred = th.argmax(pred).item()
            y = y.item()
            val_data_1_ratio += (y == 1)
            total_correct += (pred == y)
        loss_list.append(cur_epoch_loss)
        print(f'Epoch {epoch}| loss: {cur_epoch_loss: .4f}')
        print(f'Epoch {epoch}| Acc: {total_correct / val_size: .2f}')
        val_data_1_ratio /= val_size
    print(f'val_dataset class1 ratio: {val_data_1_ratio: .2f}')
    th.save(model.state_dict(), './checkpoint/MutagGCN_' + str(time.time()) + '.pth')
    
    # 绘制训练loss下降曲线图
    plt_x = np.arange(1, len(loss_list) + 1)
    # 绘制折线图
    plt.plot(plt_x, loss_list)

    # 添加标题和标签
    plt.ylabel('loss')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()