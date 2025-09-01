# net.py

from math import ceil

from todynet_model.layer import *
import torch.nn as nn

"""
https://github.com/liuxz1011/TodyNet

@article{Liu_2024,
   title={TodyNet: Temporal dynamic graph neural network for multivariate time series classification},
   volume={677},
   ISSN={0020-0255},
   url={http://dx.doi.org/10.1016/j.ins.2024.120914},
   DOI={10.1016/j.ins.2024.120914},
   journal={Information Sciences},
   publisher={Elsevier BV},
   author={Liu, Huaiyuan and Yang, Donghua and Liu, Xianzhang and Chen, Xinglei and Liang, Zhiyu and Wang, Hongzhi and Cui, Yong and Gu, Jun},
   year={2024},
   month=aug, pages={120914} }

"""


class GNNStack(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU()):

        super().__init__()
        
        # TODO: Sparsity Analysis
        k_neighs = self.num_nodes = num_nodes
        
        self.num_graphs = groups
        print("num_graphs: ", self.num_graphs)
        self.num_feats = seq_len
        print("num_feats: ", self.num_feats)

        if seq_len % groups:
            self.num_feats += ( groups - seq_len % groups )
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)
        
        gnn_model, heads = self.build_gnn_model(gnn_model_type)
        print("heads: ", heads)
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [ (k - 1) // 2 for k in kern_size ]
        
        self.tconvs = nn.ModuleList(
            [nn.Conv2d(1, in_dim, (1, kern_size[0]), padding=(0, paddings[0]))] + 
            [nn.Conv2d(heads * in_dim, hidden_dim, (1, kern_size[layer+1]), padding=(0, paddings[layer+1])) for layer in range(num_layers - 2)] + 
            [nn.Conv2d(heads * hidden_dim, out_dim, (1, kern_size[-1]), padding=(0, paddings[-1]))]
        )
        

        print("in_dim: ", in_dim)
        print("groups: ", groups)
        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] + 
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 2)] + 
            [gnn_model(out_dim, heads * out_dim, groups)]
        )
        
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(heads * in_dim)] + 
            [nn.BatchNorm2d(heads * hidden_dim) for _ in range(num_layers - 2)] + 
            [nn.BatchNorm2d(heads * out_dim)]
        )
        
        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round( num_nodes * (1 - (pool_ratio*layer)) )
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        self.diffpool = nn.ModuleList(
            [Dense_TimeDiffPool2d(self.left_num_nodes[layer], self.left_num_nodes[layer+1], kern_size[layer], paddings[layer]) for layer in range(num_layers - 1)] + 
            [Dense_TimeDiffPool2d(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.activation = activation
        # Use a separate activation function for each layer
        #self.activations = nn.ModuleList([activation for _ in range(num_layers)])

        #print("num_layers: ", self.num_layers)
        #print("dropout:", self.dropout)
        #print("activation:", self.activation)

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.linear = nn.Linear(heads * out_dim, num_classes)


        self.reset_parameters()
        
        
    def reset_parameters(self):
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            tconv.reset_parameters()
            gconv.reset_parameters()
            bn.reset_parameters()
            pool.reset_parameters()
        
        self.linear.reset_parameters()
        
        
    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        

    def forward(self, inputs: Tensor):
        intermediate_outputs = []
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs

        adj = self.g_constr(x.device)
        #print(f"Shape before first Conv2d: {x.shape}")
        for tconv, gconv, bn, pool in zip(self.tconvs, self.gconvs, self.bns, self.diffpool):
            x_tconv = tconv(x)
            #print("Shape of x_tconv in first convolution step", x_tconv.shape)
            #print("Shape of adj in first convolution step", adj.shape)
            x_gconv = gconv(x_tconv, adj)
            #print("Shape of x_gconv in second convolution step", x_gconv.shape)

            x_pooled, adj = pool(x_gconv, adj)
            #x = self.activation(bn(x_pooled))
            #x = F.dropout(x, p=self.dropout, training=self.training)

            # changed self.activation(bn(x_pooled))to F.relu due to compatibility issues with captum DeepLift
            x = F.relu(bn(x_pooled))  # Use functional activation
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Store intermediate outputs for interpretability
            intermediate_outputs.append(x)
        
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #print(f"Shape of output in forward net.py: {out.shape}")

        return out, intermediate_outputs  # Return logits and intermediate outputs