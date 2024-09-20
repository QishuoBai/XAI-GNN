import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class MutagGCN_Explainer(nn.Module):
    def __init__(self) -> None:
        super(MutagGCN_Explainer, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def message_func_l1(self, edges):
        return {'m':  self.edge_mask.sigmoid() * th.cat((edges.data['h'], edges.src['h']), 1)}
    
    def message_func_l2(self, edges):
        return {'m': self.edge_mask.sigmoid() * edges.src['h']} 
    
    def forward(self, graph, edge_mask, batch_size = 1):
        with graph.local_scope():
            # init
            graph.ndata['h'] = graph.ndata['feature']
            graph.edata['h'] = graph.edata['feature']
            # 获取张量的长度
            n = edge_mask.shape[0]

            # 扩展为大小为(2n, 1)的张量
            self.edge_mask = th.empty((2 * n, 1))
            self.edge_mask[0::2] = edge_mask
            self.edge_mask[1::2] = edge_mask
            # l1
            graph.update_all(self.message_func_l1, fn.mean('m', 'h_neigh'))
            graph.ndata['h'] = self.l1(graph.ndata['h_neigh'])
            # l2
            graph.update_all(self.message_func_l2, fn.mean('m', 'h_neigh'))
            graph.ndata['h'] = self.l2(graph.ndata['h_neigh'])
            # print(graph.ndata['h'].shape)
            if batch_size > 1:
                graphs = dgl.unbatch(graph)
                pred = th.zeros((batch_size, 2))
                for i, g in enumerate(graphs):
                    with g.local_scope():
                        # MLP
                        p = F.softmax(self.mlp(th.max(g.ndata['h'], 0)[0]), 0)
                        pred[i] = p
                return pred
            else:
                pred = F.softmax(self.mlp(th.max(graph.ndata['h'], 0)[0]), 0)
                return pred.unsqueeze(0)

class MutagGCN(nn.Module):
    def __init__(self) -> None:
        super(MutagGCN, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def message_func_l1(self, edges):
        return {'m': th.cat((edges.data['h'], edges.src['h']), 1)}
    
    def message_func_l2(self, edges):
        return {'m': edges.src['h']} 
    
    def forward(self, graph, batch_size = 5):
        with graph.local_scope():
            # init
            graph.ndata['h'] = graph.ndata['feature']
            graph.edata['h'] = graph.edata['feature']
            # l1
            graph.update_all(self.message_func_l1, fn.mean('m', 'h_neigh'))
            graph.ndata['h'] = self.l1(graph.ndata['h_neigh'])
            # l2
            graph.update_all(self.message_func_l2, fn.mean('m', 'h_neigh'))
            graph.ndata['h'] = self.l2(graph.ndata['h_neigh'])
            # print(graph.ndata['h'].shape)
            if batch_size > 1:
                graphs = dgl.unbatch(graph)
                pred = th.zeros((batch_size, 2))
                for i, g in enumerate(graphs):
                    with g.local_scope():
                        # MLP
                        p = F.softmax(self.mlp(th.max(g.ndata['h'], 0)[0]), 0)
                        pred[i] = p
                return pred
            else:
                pred = F.softmax(self.mlp(th.max(graph.ndata['h'], 0)[0]), 0)
                return pred.unsqueeze(0)
