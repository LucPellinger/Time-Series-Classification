import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossChannelAttention(nn.Module):
    def __init__(self, D, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = D // num_heads
        assert D % num_heads == 0, "D must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(D, D * 3, bias=False)
        self.fc_out = nn.Linear(D, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, D)
        b, n, D = x.shape
        x = x.reshape(b * n, D)  # Merge batch and seq_len for processing

        # Generate queries, keys, and values
        qkv = self.qkv(x)  # (b * n, D * 3)
        qkv = qkv.reshape(b * n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # (3, num_heads, b * n, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (num_heads, b * n, head_dim)

        # Compute attention scores
        attn_scores = torch.einsum('hbd,hcd->hbc', q, k) * self.scale  # (num_heads, b * n, b * n)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('hbc,hcd->hbd', attn_weights, v)  # (num_heads, b * n, head_dim)

        # Concatenate heads and reshape back to (b, n, D)
        attn_output = attn_output.permute(1, 0, 2).reshape(b * n, D)
        attn_output = self.fc_out(attn_output)
        attn_output = attn_output.view(b, n, D)

        return attn_output
