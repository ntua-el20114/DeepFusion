import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention for final fusion"""
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_modalities, hidden_dim]
        Returns:
            attended_x: [batch_size, num_modalities, hidden_dim]
            attention_weights: [batch_size, num_heads, num_modalities, num_modalities]
        """
        batch_size, num_modalities, _ = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, num_modalities, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, num_modalities, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, num_modalities, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_modalities, self.hidden_dim)
        output = self.output_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output, attention_weights
