# tody_net.py

import torch
import torch.nn as nn
from todynet_model.net import GNNStack  # Import GNNStack from TodyNet's net.py
import torch.nn.functional as F

class TodyNetClassifier(nn.Module):
    def __init__(self, input_dim, seq_length, num_nodes, num_classes,
                 num_layers=3, groups=4, pool_ratio=0.2, kern_size=[9,5,3],
                 in_dim=64, hidden_dim=128, out_dim=256, dropout=0.5, gnn_model_type='dyGIN2d'):
        super().__init__() #TodyNetClassifier, self
        self.model = GNNStack(
            gnn_model_type=gnn_model_type,
            num_layers=num_layers,
            groups=groups,
            pool_ratio=pool_ratio,
            kern_size=kern_size,
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            seq_len=seq_length,
            num_nodes=num_nodes,
            num_classes=num_classes,
            dropout=dropout,
            activation=nn.ReLU()
        )

    def forward(self, x):
        # x should be of shape [batch_size, 1, num_nodes, seq_length]
        # print("Shape of x in TodyNetClassifier:", x.shape)
        logits, intermediate_outputs = self.model(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs, intermediate_outputs  # Return logits, probabilities, and intermediate outputs
