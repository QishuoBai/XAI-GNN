import networkx as nx
from dgl import from_networkx
import torch as th
import pandas as pd

def load_ton_iot_train_test(train_file, test_file, bin=True):
    # X_train = pd.read_parquet(train_file)
    # X_test = pd.read_parquet(test_file)
    X_train = pd.read_csv(train_file)
    X_test = pd.read_csv(test_file)

    if bin:
        y_train, y_test = X_train.label, X_test.label
    else:
        y_train, y_test = X_train.type, X_test.type

    # norm_cols = ['spkts', 'dload', 'dloss', 'dpkts', 'service', 'sbytes', 'sloss', 'sintpkt', 'sload', 'sjit', 'dintpkt', 'synack', 'djit', 'tcprtt', 'dtcpb', 'dmeansz', 'is_sm_ips_ports', 'src_ip', 'ct_dst_sport_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'dbytes', 'trans_depth', 'proto', 'stcpb', 'dwin', 'ackdat', 'is_ftp_login', 'smeansz', 'dst_ip', 'swin', 'dur', 'state', 'ct_src_dport_ltm']
    norm_cols = X_train.columns.tolist()[3:-2]
    # norm_cols = ['dns_rcode', 'http_status_code', 'ssl_version', 'http_resp_mime_types', 'duration', 'dns_qclass', 'dns_rejected', 'dns_qtype', 'proto', 'src_bytes', 'dst_pkts', 'src_pkts', 'src_ip_bytes', 'missed_bytes', 'conn_state', 'http_method', 'ssl_resumed', 'dst_ip_bytes', 'dns_RD', 'dst_bytes', 'ssl_established', 'dns_RA', 'http_orig_mime_types', 'http_response_body_len', 'http_version', 'service', 'http_request_body_len', 'dns_AA', 'ssl_cipher', 'http_trans_depth']
    X_train['h'] = X_train[norm_cols].values.tolist()
    X_test['h'] = X_test[norm_cols].values.tolist()

    return X_train, X_test, y_train, y_test

# 从处理好的train和test里构图
def generate_ton_iot_graph(X, is_embed=False):
    eattrs = ['ID', 'h', 'e', 'label', 'type'] if is_embed else ['ID', 'h', 'label', 'type']
    G = nx.from_pandas_edgelist(X, 'src_ip', 'dst_ip', eattrs, create_using=nx.MultiGraph)
    G = G.to_directed()
    G = from_networkx(G, edge_attrs=eattrs)
    G.ndata['h'] = th.ones(G.num_nodes(), G.edata['h'].shape[1])

    return G

# 计算准确率
def compute_accuracy(pred, labels):
    return (pred.argmax(1) == labels).float().mean().item()

def save_model(model, params, acc, dataset, bin=True):
    pt_name = 'pts/' + dataset + '-p-'
    for key in params:
        pt_name += (key+'_'+str(params[key]))
    if bin:
        pt_name += '_bin'
    else:
        pt_name += '_multi'

    pt_name+='('+str(round(acc, 4))+').pt'

    th.save(model.state_dict(), pt_name)
    print('Model saved as ' + pt_name)

def save_coder(model, params, acc, dataset):
    pt_name='pts/coder-'+dataset+'-p-'
    for key in params:
        pt_name+=(key+'_'+str(params[key]))

    pt_name+='('+str(round(acc, 4))+').pt'

    th.save(model.state_dict(), pt_name)