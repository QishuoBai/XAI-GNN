import torch.nn as nn
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from models.GraphSAGE import SAGE
from models.MLP import MLPPredictor
from utils.tools import generate_ton_iot_graph, load_ton_iot_train_test, compute_accuracy, save_model
from sklearn.metrics import accuracy_score
import time

device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print('Using device:', device)

params={
    'ndim_out': 128,
    'dropout': 0.2,
    'epochs': 500, # 2000
}
bin = False

print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
print(params)

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


# 获取训练测试集
print('Loading data...')
X_train, X_test, y_train, y_test = load_ton_iot_train_test('datasets/TON-IoT/ton_iot_train.csv', 'datasets/TON-IoT/ton_iot_test.csv', bin)

# 构图
print('Generating graph...')
G = generate_ton_iot_graph(X_train, False)
G_test = generate_ton_iot_graph(X_test, False)
G = G.to(device)
G_test = G_test.to(device)

if bin:
    label_key = 'label'
else:
    label_key = 'type'

# 模型
print('Initializing model...')
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(G.edata[label_key].cpu().numpy()), y=G.edata[label_key].cpu().numpy())
class_weights = th.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)

model = Model(G.ndata['h'].shape[1], params['ndim_out'], G.edata['h'].shape[1], F.relu, params['dropout'], bin)
model = model.to(device)
optim = th.optim.Adam(model.parameters())



# 训练
print('Training...')
model.train()
for epoch in range(1, params['epochs']+1):
    pred = model(G, G.ndata['h'], G.edata['h'])
    loss = criterion(pred, G.edata[label_key])
    optim.zero_grad()
    loss.backward()
    optim.step()
    if epoch % 10 == 0:
        print('Epoch:', epoch ,' Training acc:', compute_accuracy(pred, G.edata[label_key]))

# 测试
print('Testing...')
model.eval()
test_pred = model(G_test, G_test.ndata['h'], G_test.edata['h'])
test_pred = test_pred.argmax(1)
test_pred = th.Tensor.cpu(test_pred).detach().numpy()
test_label = th.Tensor.cpu(G_test.edata[label_key]).detach().numpy()
print(classification_report(test_label, test_pred, digits=4))

acc=accuracy_score(test_label, test_pred)

# 保存模型
print('Saving model...')
model = model.to('cpu')
save_model(model, params, acc, 'TON-IoT', bin)