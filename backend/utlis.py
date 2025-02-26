import pandas as pd
import numpy as np
from models.GraphSAGE import SAGE
from models.MLP import MLPPredictor
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
from dgl import from_networkx
import torch as th

X_train = pd.read_csv('../Experiment/datasets/TON-IoT/ton_iot_train.csv')
X_test = pd.read_csv('../Experiment/datasets/TON-IoT/ton_iot_test.csv')
X = pd.concat([X_train, X_test])
X.sort_values(by='ID', ascending=True, inplace=True)

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout, bin=False):
        super(Model, self).__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        if bin:
            self.pred = MLPPredictor(ndim_out, edim, 2)
        else:
            self.pred = MLPPredictor(ndim_out, edim, 10)
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
# 加载模型
model = Model(30, 128, 30, F.relu, 0.2, False)
model.load_state_dict(th.load('../Experiment/pts/TON-IoT-p-ndim_out_128dropout_0.2epochs_500_multi(0.8653).pt'))
model = model.to(device)
model.eval()
coeffs = {
        "feat_pred": 3.0,
        "feat_size": 1.0,
        "feat_ent": 0.1,
        "edge_pred": 3.0,
        "edge_size": 50,
        "edge_ent": 0.1,
    }

def feature_mask_loss(g, pred_feature_masked, feature_mask, eid, user_coeffs):
    # feature_mask 是sigmod过的

    eid_selector = g.edata['ID'] == eid
    # MI loss
    type = g.edata['type'][eid_selector][0].item()
    pred_loss = -th.log(pred_feature_masked[eid_selector][:,type].mean())
    pred_loss = coeffs['feat_pred'] * pred_loss

    # size
    feature_mask_size_loss = coeffs['feat_size'] * th.mean(feature_mask)

    # entropy
    feature_mask_ent = -feature_mask * th.log(feature_mask) - (1 - feature_mask) * th.log(1 - feature_mask)
    feature_mask_ent_loss = coeffs['feat_ent'] * th.mean(feature_mask_ent)

    loss = pred_loss * user_coeffs['feat_pred'] + feature_mask_size_loss * user_coeffs['feat_size'] + feature_mask_ent_loss * user_coeffs['feat_ent']

    return loss

def edge_mask_loss(g, pred_edge_masked, edge_mask, eid, user_coeffs):
    # edge_mask 是sigmod过的

    eid_selector = g.edata['ID'] == eid
    # MI loss
    type = g.edata['type'][eid_selector][0].item()
    pred_loss = -th.log(pred_edge_masked[eid_selector][:,type].mean())
    pred_loss = coeffs['edge_pred'] * pred_loss

    # size
    edge_mask_size_loss = coeffs['edge_size'] * th.mean(edge_mask)

    # entropy
    edge_mask_ent = -edge_mask * th.log(edge_mask) - (1 - edge_mask) * th.log(1 - edge_mask)
    edge_mask_ent_loss = coeffs['edge_ent'] * th.mean(edge_mask_ent)

    loss = pred_loss * user_coeffs['edge_pred'] + edge_mask_size_loss * user_coeffs['edge_size'] + edge_mask_ent_loss * user_coeffs['edge_ent']
    return loss

def cal_gnnexplainer_result(explain_id, user_coeffs):
    # 找到id对应两跳的数据
    expand_round = 2
    ids = np.array([explain_id])
    for _ in range(expand_round):
        cur_nodes = np.unique(np.concatenate([X[X['ID'].isin(ids)]['src_ip'].values, X[X['ID'].isin(ids)]['dst_ip'].values]))
        not_in_ids = X[~X['ID'].isin(ids)]
        not_in_ids = not_in_ids[(not_in_ids['src_ip'].isin(cur_nodes)) | (not_in_ids['dst_ip'].isin(cur_nodes))]
        if not_in_ids.shape[0] == 0:
            break
        ids = np.concatenate([ids, not_in_ids['ID'].values])
    sub_X = X[X['ID'].isin(ids)].copy()
    norm_cols = X.columns.to_list()[3:-2]
    sub_X['h'] = sub_X[norm_cols].values.tolist()
    eattrs = ['ID', 'h', 'label', 'type']
    G = nx.from_pandas_edgelist(sub_X, 'src_ip', 'dst_ip', eattrs, create_using=nx.MultiGraph)
    G = G.to_directed()
    G = from_networkx(G, edge_attrs=eattrs)
    G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])

    feature_mask = nn.Parameter(th.zeros(G.edata['h'].shape[1]).unsqueeze(0).to(device), requires_grad=True)
    edge_mask = nn.Parameter(th.zeros(G.edata['h'].shape[0]).unsqueeze(1).to(device), requires_grad=True)
    G = G.to(device)

    eid = explain_id

    # 训练优化 feature mask
    print("Training Feature Mask...")
    optimizer = th.optim.Adam([feature_mask], lr=0.01)
    Epochs_feat = user_coeffs['feat_epochs']
    for epoch in range(Epochs_feat):
        optimizer.zero_grad()
        pred_feature_masked = model(G, G.ndata['h'], G.edata['h'] * feature_mask.sigmoid())
        loss = feature_mask_loss(G, pred_feature_masked, feature_mask.sigmoid(), eid, user_coeffs)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(loss.item())

    # 训练优化 edge mask
    print("Training Edge Mask...")
    optimizer = th.optim.Adam([edge_mask], lr=0.01)
    Epochs_edge = user_coeffs['edge_epochs']
    for epoch in range(Epochs_edge):
        optimizer.zero_grad()
        pred_edge_masked = model(G, G.ndata['h'], G.edata['h'] * edge_mask.sigmoid())
        loss = edge_mask_loss(G, pred_edge_masked, edge_mask.sigmoid(), eid, user_coeffs)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(loss.item())
    
    pred_origin = model(G, G.ndata['h'], G.edata['h'])
    pred_feature_masked = model(G, G.ndata['h'], G.edata['h'] * feature_mask.sigmoid())
    pred_edge_masked = model(G, G.ndata['h'], G.edata['h'] * edge_mask.sigmoid())
    
    explainer_result = {}
    explainer_result['feature_importance'] = feature_mask.sigmoid().cpu().detach().numpy().squeeze().tolist()
    explainer_result['edge_importance'] = []
    edge_mask_list = edge_mask.sigmoid().cpu().detach().numpy().squeeze().tolist()
    IDs_list = G.edata['ID'].cpu().detach().numpy().tolist()
    types_list = G.edata['type'].cpu().detach().numpy().tolist()
    pred_list = pred_origin.argmax(dim=1).cpu().detach().numpy().tolist()
    for i in range(len(edge_mask_list)):
        explainer_result['edge_importance'].append({'ID': IDs_list[i], 'importance': edge_mask_list[i], 'type': types_list[i], 'pred': pred_list[i]})
    explainer_result['pred_origin'] = pred_origin[G.edata['ID'] == eid].argmax(dim=1).cpu().detach().numpy().tolist()[0]
    explainer_result['pred_feature_masked'] = pred_feature_masked[G.edata['ID'] == eid].argmax(dim=1).cpu().detach().numpy().tolist()[0]
    explainer_result['pred_edge_masked'] = pred_edge_masked[G.edata['ID'] == eid].argmax(dim=1).cpu().detach().numpy().tolist()[0]
    explainer_result['pred_origin_vec'] = F.softmax(pred_origin[G.edata['ID'] == eid].cpu().detach()[0], dim=0).numpy().tolist()
    explainer_result['pred_feature_masked_vec'] = F.softmax(pred_feature_masked[G.edata['ID'] == eid].cpu().detach()[0], dim=0).numpy().tolist()
    explainer_result['pred_edge_masked_vec'] = F.softmax(pred_edge_masked[G.edata['ID'] == eid].cpu().detach()[0], dim=0).numpy().tolist()
    return explainer_result


def get_edge_feature_values(data, id, feature_keys):
    row = data[data['ID'] == id]
    values = row[feature_keys].values[0].tolist()
    return values