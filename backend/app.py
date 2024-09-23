# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)

CORS(app)

edgelist = pd.read_csv('../Experiment/datasets/TON-IoT/edgelist_ton_iot.csv')
type_map = {'normal': 0, 'backdoor': 1, 'ddos': 2, 'dos': 3, 'injection': 4, 'password': 5, 'ransomware': 6, 'scanning': 7, 'xss': 8, 'mitm': 9}
predictdetail = pd.read_csv('../Experiment/datasets/TON-IoT/predictdetail_ton_iot.csv')

# 主页路由
@app.route('/hello')
def home():
    return "Welcome to the Flask API Demo!"

@app.route('/api/data_loader/cm', methods=['POST'])
def data_loader_cm():
    data = request.get_json()
    print(data)
    type = type_map[data['row_type']]
    pred = type_map[data['col_type']]
    edge_num = data['edge_num']
    edge_id_range = data['edge_id_range']
    filtered = edgelist[(edgelist['type'] == type) & (edgelist['pred'] == pred)]
    filtered = filtered[edge_id_range[0]:edge_id_range[1]]

    target_ids = filtered['ID'].values
    all_ids = target_ids.copy()

    # 向外拓展target数组
    while len(all_ids) < edge_num:
        cur_nodes = np.unique(np.concatenate([edgelist[edgelist['ID'].isin(all_ids)]['src_ip'].values, edgelist[edgelist['ID'].isin(all_ids)]['dst_ip'].values]))
        not_in_all_ids = edgelist[~edgelist['ID'].isin(all_ids)]
        not_in_all_ids = not_in_all_ids[(not_in_all_ids['src_ip'].isin(cur_nodes)) | (not_in_all_ids['dst_ip'].isin(cur_nodes))]
        if not_in_all_ids.shape[0] == 0:
            break
        if all_ids.shape[0] + not_in_all_ids.shape[0] > edge_num:
            not_in_all_ids = not_in_all_ids[:edge_num - all_ids.shape[0]]
        all_ids = np.concatenate([all_ids, not_in_all_ids['ID'].values])

    all_ids.sort()
    # nodes
    nodes = np.unique(np.concatenate([edgelist[edgelist['ID'].isin(all_ids)]['src_ip'].values, edgelist[edgelist['ID'].isin(all_ids)]['dst_ip'].values]))
    nodes = [{'ip': node} for node in nodes]
    links = edgelist[edgelist['ID'].isin(all_ids)]
    links.rename(columns={'src_ip': 'source'}, inplace=True)
    links.rename(columns={'dst_ip': 'target'}, inplace=True)
    links = links.to_dict(orient='records')
    return {'target_ids': target_ids.tolist(), 'all_ids': all_ids.tolist(), 'nodes': nodes, 'links': links}

@app.route('/api/get_predict_detail', methods=['POST'])
def get_predict_detail():
    data = request.get_json()
    print(data)
    id = data['ID']
    return predictdetail[predictdetail['ID'] == id]['pred_origin'].values[0]

# 启动应用
if __name__ == '__main__':
    app.run(debug=True)