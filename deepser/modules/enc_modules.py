import torch
from torch import nn

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=1)
    

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Create a learnable query vector for pooling
        global_query = x.mean(dim=1, keepdim=True)  # (batch_size, 1, embed_dim)
        
        Q = self.query(global_query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        pooled = torch.matmul(attn_weights, V)  # (batch_size, num_heads, 1, head_dim)
        pooled = pooled.transpose(1, 2).contiguous().view(batch_size, embed_dim)
        
        return self.out_proj(pooled)


class LinearEncoder(nn.Module): 
    def __init__(self, dim, num_layers, dropout):
        super(LinearEncoder, self).__init__()  

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),   #
                nn.Linear(dim, dim), 
                nn.ReLU(),
                nn.Dropout(dropout)) # 
            for _ in range(num_layers)
        ])
    
    def __iter__(self):  
        return iter(self.layers)
    
class BottleneckMLP(nn.Module): 
    def __init__(self, dim, num_layers, factor=4, dropout=0.1):
        super(BottleneckMLP, self).__init__()  
         
        bottleneck_dim = dim // factor

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, bottleneck_dim),  # Compress
                nn.LayerNorm(bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, dim * factor),  # Expand
                nn.LayerNorm(dim * factor),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim * factor, dim),  # Back to original
                nn.Dropout(dropout))  
            for _ in range(num_layers)
        ])
    
    def __iter__(self):  
        return iter([lambda x, layer=layer: x + layer(x) for layer in self.layers]) # Residual connection
    

class MultiScaleEncoder(nn.Module):
    """Processes representations at multiple scales simultaneously"""
    def __init__(self, dim, num_layers, scales=[1, 2, 4], dropout=0.1):
        super().__init__()
        self.scales = scales
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                f'scale_{scale}': nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim // scale),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim // scale, dim),
                    nn.Dropout(dropout)
                ) for scale in scales
            }) for _ in range(num_layers)
        ])
        
        # Fusion layers to combine multi-scale outputs
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * len(scales), dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
    
    def __iter__(self):
        def make_layer_func(layer_dict, fusion):
            def layer_func(x):
                # Process at each scale
                scale_outputs = []
                for scale in self.scales:
                    scale_outputs.append(layer_dict[f'scale_{scale}'](x))
                
                # Concatenate and fuse
                fused = torch.cat(scale_outputs, dim=-1)
                return x + fusion(fused)
            return layer_func
        
        return iter([make_layer_func(layer, fusion) 
                    for layer, fusion in zip(self.layers, self.fusion_layers)])


class ConvolutionalEncoder(nn.Module):
    """1D Convolutions treating sentence vector as a 1D signal"""
    def __init__(self, dim, num_layers, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                f'conv_{k}': nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=k, padding=k//2),  # Treat dim as sequence
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Conv1d(8, 1, kernel_size=1),  # Back to single channel
                    nn.Dropout(dropout)
                ) for k in kernel_sizes
            }) for _ in range(num_layers)
        ])
        
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * len(kernel_sizes), dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
    
    def __iter__(self):
        def make_layer_func(layer_dict, fusion):
            def layer_func(x):
                # x: [batch, dim] -> [batch, 1, dim] for 1D conv
                x_conv = x.unsqueeze(1)
                
                conv_outputs = []
                for k in self.kernel_sizes:
                    conv_out = layer_dict[f'conv_{k}'](x_conv)  # [batch, 1, dim]
                    conv_outputs.append(conv_out.squeeze(1))  # [batch, dim]
                
                # Fuse multi-kernel outputs
                fused = torch.cat(conv_outputs, dim=-1)
                return x + fusion(fused)
            return layer_func
        
        return iter([make_layer_func(layer, fusion) 
                    for layer, fusion in zip(self.layers, self.fusion_layers)])
    
