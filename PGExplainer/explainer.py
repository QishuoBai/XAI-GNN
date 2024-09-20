import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class MutagGCN_Explainer(nn.Module):
    def __init__(self) -> None:
        super(MutagGCN_Explainer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, embedding, training = False, tau = 1.0):
        if training:
            w = self.mlp(embedding)
            random_noise = th.rand(w.shape)
            random_noise = th.log(random_noise) - th.log(1-random_noise)
            e = F.sigmoid((random_noise + w) / tau)
        else:
            e = F.sigmoid(self.mlp(embedding))
        return e