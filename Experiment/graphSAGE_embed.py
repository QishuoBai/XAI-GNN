import torch.nn as nn
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from models.GraphSAGE import SAGE
from models.MLP import MLPPredictorEmbed
from utils.tools import generate_ton_iot_graph, load_ton_iot_train_test, compute_accuracy, save_model
from sklearn.metrics import accuracy_score
import time

params={
    'ndim_out': 128,
    'dropout': 0.2,
    'epochs': 2000,
}
bin = True

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
print(params)

class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, embedim, activation, dropout, bin=True):
        super(Model, self).__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        if bin:
            self.pred = MLPPredictorEmbed(ndim_out, edim, embedim, 2)
        else:
            self.pred = MLPPredictorEmbed(ndim_out, edim, embedim, 10)
    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()  
        )

    def forward(self, x):
        e = self.encoder(x)
        x_ = self.decoder(e)
        return x_, e

# 获取训练测试集
X_train, X_test, y_train, y_test = load_ton_iot_train_test('datasets/TON-IoT/ton_iot_train.parquet', 'datasets/TON-IoT/ton_iot_test.parquet', bin=True)

# 获取边的embedding
input_dim, latent_dim=len(X_train['h'].iloc[0]), 10
coder = AutoEncoder(input_dim, latent_dim)
coder_criterion = nn.MSELoss()
coder_optimizer = th.optim.Adam(coder.parameters(), lr=1e-2)

coder_input=th.tensor(X_train['h'].to_list()).float()
# 训练自编码器
for epoch in range(1200):
    coder_optimizer.zero_grad()
    outputs, embed = coder(coder_input)
    loss = coder_criterion(outputs, coder_input)
    loss.backward()
    coder_optimizer.step()
    if epoch%100==0:
        print(f'Epoch [{epoch+1}/1200], Loss: {loss.item():.4f}')
X_train['e']=embed.tolist()

coder_test_input=th.tensor(X_test['h'].to_list()).float()
test_outputs, test_embed=coder(coder_test_input)
X_test['e']=test_embed.tolist()

# 构图
G = generate_ton_iot_graph(X_train, True)
G_test = generate_ton_iot_graph(X_test, True)

# 模型
class_weights = class_weight.compute_class_weight('balanced', np.unique(G.edata['label'].cpu().numpy()), G.edata['label'].cpu().numpy())
criterion = nn.CrossEntropyLoss(weight = th.FloatTensor(class_weights))

model = Model(G.ndata['h'].shape[1], params['ndim_out'], G.edata['h'].shape[1], latent_dim, F.relu, params['dropout'], bin)
optim = th.optim.Adam(model.parameters())

# 训练
for epoch in range(1, params['epochs']+1):
    pred = model(G, G.ndata['h'], G.edata['h'])
    loss = criterion(pred, G.edata['label'])
    optim.zero_grad()
    loss.backward()
    optim.step()
    if epoch % 100 == 0:
        print('Epoch:', epoch ,' Training acc:', compute_accuracy(pred, G.edata['label']))

# 测试
test_pred = model(G_test, G_test.ndata['h'], G_test.edata['h'])
test_pred = test_pred.argmax(1)
test_pred = th.Tensor.cpu(test_pred).detach().numpy()
print(classification_report(G_test.edata['label'], test_pred, digits=4))

acc=accuracy_score(G_test.edata['label'], test_pred)

# 保存模型
save_model(model, params, acc)