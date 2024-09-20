import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import MutagGCN
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import time

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

    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = MutagGCN()
    model.load_state_dict(th.load('./weight/MutagGCN.pth'))
    # train
    loss_function = nn.CrossEntropyLoss(weight=th.tensor([1.0, 1.0]))
    model.eval()
    total_correct = 0
    val_data_1_ratio = 0
    for x, y in data_loader:
        pred = model(x, 1).squeeze()
        pred = th.argmax(pred).item()
        y = y.item()
        val_data_1_ratio += (y == 1)
        total_correct += (pred == y)
    print(f'Acc: {total_correct / len(dataset): .2f}')
    val_data_1_ratio /= len(dataset)
    print(f'val_dataset class1 ratio: {val_data_1_ratio: .2f}')