import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from explainer import MutagGCN_Explainer
from model import MutagGCN, MutagGCN_Masked
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import time
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

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

def loss_function(ys, y0, edge_mask):
    pred_loss = - th.log(ys[th.argmax(y0).item()])
    edge_mask_sigmoid = edge_mask
    mask_size_loss = F.relu(th.sum(edge_mask_sigmoid) - 5)
    mask_entropy_loss = - th.sum(edge_mask_sigmoid * th.log(edge_mask_sigmoid) + (1 - edge_mask_sigmoid) * th.log(1 - edge_mask_sigmoid))
    return 2.0 * pred_loss + 0.02 * mask_size_loss + 0.02 * mask_entropy_loss


if __name__ == '__main__':
    graphs = dgl.load_graphs('../MUTAG/data/graphs.dgl')[0]
    graph_labels = th.load('../MUTAG/data/graph_labels.pth')
    graphs = graphs[0:1]
    graph_labels = graph_labels[0:1]

    dataset = MutagDataset(graphs, graph_labels)
    data_size = len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    explainer = MutagGCN_Explainer()
    explainer.train()

    model = MutagGCN()
    model.load_state_dict(th.load('../MUTAG/weight/MutagGCN.pth'))
    model.train()
    model_masked = MutagGCN_Masked()
    model_masked.load_state_dict(th.load('../MUTAG/weight/MutagGCN.pth'))
    model_masked.train()

    # train
    optimizer = optim.Adam(explainer.parameters(), 1e-3)

    Epoch = 1000
    for epoch in tqdm(range(Epoch)):
        loss = th.tensor((0)).float()
        for x, y in loader:
            y0, edge_embedding = model(x, 1)
            y0 = y0.squeeze()
            edge_mask = explainer(edge_embedding, training=True)
            # print(edge_mask)
            ys = model_masked(x, edge_mask, 1).squeeze()
            loss += loss_function(ys, y0, edge_mask)
        cur_explainer = explainer.state_dict()
        loss /= data_size
        # print(loss.item())
        if th.isnan(loss):
            print('Nan!!!')
            cur_explainer = last_explainer
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_explainer = explainer.state_dict()
        # print(edge_mask.sigmoid())
    
    th.save(cur_explainer, './checkpoint/MUTAG_PGExplainer_' + str(time.time()) + '.pth')