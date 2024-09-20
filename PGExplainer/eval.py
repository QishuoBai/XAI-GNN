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
import os
import sys

# 自定义数据集类
# class MutagDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         sample = self.data[index]
#         label = self.labels[index]
#         return sample, label

# def collate_fn(batch):
#     graphs, labels = zip(*batch)
#     labels = th.tensor(labels).long()
#     return dgl.batch(graphs), labels

# def loss_function(ys, y0, edge_mask):
#     pred_loss = - th.log(ys[th.argmax(y0).item()])
#     edge_mask_sigmoid = edge_mask
#     mask_size_loss = th.sum(edge_mask_sigmoid)
#     mask_entropy_loss = - th.sum(edge_mask_sigmoid * th.log(edge_mask_sigmoid) + (1 - edge_mask_sigmoid) * th.log(1 - edge_mask_sigmoid))
#     return 2.0 * pred_loss + 0.01 * mask_size_loss + 0.05 * mask_entropy_loss


if __name__ == '__main__':
    graphs = dgl.load_graphs('../MUTAG/data/graphs.dgl')[0]
    graph_labels = th.load('../MUTAG/data/graph_labels.pth')

    explainer = MutagGCN_Explainer()
    explainer_weight = sorted(os.listdir('./checkpoint'))[-1]
    print(explainer_weight)
    explainer.load_state_dict(th.load('./checkpoint/' + explainer_weight)) 
    explainer.eval()

    model = MutagGCN()
    model.load_state_dict(th.load('../MUTAG/weight/MutagGCN.pth'))
    model.eval()
    model_masked = MutagGCN_Masked()
    model_masked.load_state_dict(th.load('../MUTAG/weight/MutagGCN.pth'))
    model_masked.eval()
    
    # 待解释对象
    graph_id = int(sys.argv[1])
    x = graphs[graph_id]
    y = graph_labels[graph_id]
    y0, edge_embedding = model(x, 1)
    # print(edge_embedding[0])
    y0 = y0.squeeze()
    edge_mask = explainer(edge_embedding, training=False)
    print(edge_mask)
    ys = model_masked(x, edge_mask, 1).squeeze()
    print(ys[th.argmax(y0).item()])
    x.edata['mask'] = edge_mask.detach().squeeze()
    print(f'True Label: {y}')
    print(f'Origin Pred: {y0} | Masked Pred: {ys}')
    
    # 将图画出来
    # 将 DGL 图转换为 NetworkX 图
    nxg = x.to_networkx().to_undirected()

    # node color
    node_camp = ['green', 'red', 'blue', 'yellow', 'gray', 'purple']
    node_colors = [node_camp[i.item()] for i in th.argmax(x.ndata['feature'], 1)]
    
    # edge color
    edge_cmap = cm.get_cmap('coolwarm')
    src = x.edges()[0]
    dst = x.edges()[1]
    edges_tuple_list = [(src[i].item(), dst[i].item()) for i in range(src.shape[0])]
    edge_colos = []
    edge_labels = {}
    for edge in nxg.edges():
        eid1 = edges_tuple_list.index(edge)
        eid2 = eid1 + 1 if eid1 %2 == 0 else eid1 - 1
        # print((x.edata['mask'][eid1].item() + x.edata['mask'][eid2].item())/2)
        edge_labels[edge] = str(format((x.edata['mask'][eid1].item() + x.edata['mask'][eid2].item())/2, '.2f'))
        edge_colos.append(edge_cmap((x.edata['mask'][eid1].item() + x.edata['mask'][eid2].item())/2))
    # print(edge_labels)

    # 绘制图形
    pos = nx.spring_layout(nxg)  # 定义节点布局
    # pos = nx.kamada_kawai_layout(nxg)  # 定义节点布局
    nx.draw_networkx_nodes(nxg, pos, node_color=node_colors, node_size=500, alpha=0.8)  # 绘制节点，根据节点特征选择颜色
    nx.draw_networkx_edges(nxg, pos, edge_color=edge_colos, width=2, alpha=1)  # 绘制边，根据边特征选择颜色
    nx.draw_networkx_edge_labels(nxg, pos, edge_labels=edge_labels, font_size=6, font_color='black')
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图形
    