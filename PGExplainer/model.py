import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

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

    def get_edge_embedding(self, edges):
        return {'embedding': th.cat((edges.src['h'], edges.dst['h']), 1)}
    
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
                graph.apply_edges(self.get_edge_embedding)
                return pred.unsqueeze(0), graph.edata['embedding']

class MutagGCN_Masked(nn.Module):
    def __init__(self) -> None:
        super(MutagGCN_Masked, self).__init__()
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
        return {'m':  self.edge_mask * th.cat((edges.data['h'], edges.src['h']), 1)}
    
    def message_func_l2(self, edges):
        return {'m': self.edge_mask * edges.src['h']} 
    
    def forward(self, graph, edge_mask, batch_size = 1):
        with graph.local_scope():
            # init
            graph.ndata['h'] = graph.ndata['feature']
            graph.edata['h'] = graph.edata['feature']
            self.edge_mask = edge_mask
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