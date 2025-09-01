# layer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

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

class multi_shallow_embedding(nn.Module):
    
    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))
        self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)
        
    def forward(self, device):
        # adj: [G, N, N]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        
        # Remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
        
        # Top-k-edge adj
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)
        
        idx = torch.tensor([ i // self.k for i in range(indices.size(0)) ], device=device)
        
        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.0
        adj = adj_flat.reshape_as(adj)
        
        return adj

class Group_Linear(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()
             
        self.out_channels = out_channels
        self.groups = groups
        #print("in_channels in layer.py: ", in_channels)
        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.group_mlp.reset_parameters()
        
    def forward(self, x: Tensor, is_reshape=False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups
        
        if not is_reshape:
            # x: [B, C, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C, N, F//G]
        x = x.transpose(1, 2).reshape(B, G * C, N, -1)
        
        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)
        # print("Shape of layer output: ", out)
        # out: [B, C_out, G, N, F//G]
        return out

class DenseGCNConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        print("in_channels in layer.py DenseGCNConv2d: ", in_channels)
        print("out_channels in layer.py DenseGCNConv2d: ", out_channels)
        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            init.zeros_(self.bias)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj_norm
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        # Normalize adjacency matrix
        adj_norm = self.norm(adj, add_loop)
        self.attention_weights = adj_norm  # Store normalized adjacency matrix as attention weights

        # Reshape adjacency matrix for multiplication
        adj = adj_norm.unsqueeze(1)

        # x: [B, C, G, N, F//G]
        x = self.lin(x, is_reshape=False)

        # Perform graph convolution
        out = torch.matmul(adj, x)

        # Reshape output
        # out: [B, C, N, F]
        B, C, _, N, _ = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

class DenseGINConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()
        
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        # print("in_channels in layer.py DenseGINConv2d: ", in_channels)
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
        self.reset_parameters()
            
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj_norm
            
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, F = x.size()
        G = self.groups
        
        # Normalize adjacency matrix
        adj_norm = self.norm(adj, add_loop)
        self.attention_weights = adj_norm  # Store normalized adjacency matrix as attention weights

        # Reshape adjacency matrix
        adj = adj_norm.unsqueeze(0).expand(B, -1, -1, -1)  # [B, G, N, N]

        # Reshape x
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)  # [B, C, G, N, F//G]

        # Perform message passing
        neighbor_info = torch.matmul(adj, x)  # [B, C, G, N, F//G]

        if add_loop:
            out = (1 + self.eps) * x + neighbor_info
        else:
            out = neighbor_info

        # Apply MLP
        out = self.mlp(out, is_reshape=True)

        # Reshape output
        out = out.transpose(2, 3).reshape(B, self.out_channels, N, -1)  # [B, C_out, N, F]

        return out

class Dense_TimeDiffPool2d(nn.Module):
    
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()
        
        # Time convolution for pooling
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))
        
        self.re_param = Parameter(Tensor(kern_size, 1))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')
        
    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        # Apply time convolution for pooling
        x = x.transpose(1, 2)  # [B, N, C, F]
        x_pooled = self.time_conv(x)  # [B, N_pooled, C, F_pooled]
        x_pooled = x_pooled.transpose(1, 2)  # [B, C, N_pooled, F_pooled]
        
        # Generate new adjacency matrix
        s = torch.matmul(self.time_conv.weight, self.re_param).view(x_pooled.size(-2), -1)  # [N_pooled, N]
        
        # Update adjacency matrix
        adj_pooled = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))  # [G, N_pooled, N_pooled]
        
        return x_pooled, adj_pooled
