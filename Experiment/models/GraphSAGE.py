import torch.nn as nn
import torch as th
import torch.nn.functional as F
import dgl.function as fn

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edim, ndim_out, activation):
        super(SAGELayer, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edim, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        # 聚合源节点和边信息到当前节点
        return {'m': self.W_msg(th.cat([edges.src['h'], edges.data['h']], 1))}
    
    def reduce_func(self, nodes):
         # 等价于fn.mean('m', 'h_neigh')
        return {'h_neigh': th.mean(nodes.mailbox['m'], dim=1)}
    
    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            # g.update_all(self.message_func, self.reduce_func)
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            g.ndata['h'] = self.activation(self.W_apply(th.cat([g.ndata['h'], g.ndata['h_neigh']], 1)))
            return g.ndata['h']


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats
        