import torch.nn as nn
import torch as th

class MLPPredictor(nn.Module):
    def __init__(self, in_features, edge_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2 + edge_features, out_classes)
        self.softmax = nn.Softmax(dim=1)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h_e = edges.data['h']
        score = self.W(th.cat([h_u, h_e, h_v], 1))
        # score = self.softmax(score)
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
               
class MLPPredictorEmbed(nn.Module):
    def __init__(self, in_features, edge_features, edge_embed, out_classes):
        super().__init__()
        self.W1 = nn.Linear(in_features*2, edge_features)
        self.W2 = nn.Linear(edge_features+edge_embed, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        h_e = self.W1(th.cat([h_u, h_v], 1)) # 聚合节点信息到边
        score = self.W2(th.cat([h_e, edges.data['e'].squeeze()], 1))
        return {'score': score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)  # 更新特征到边上
            return g.edata['score']