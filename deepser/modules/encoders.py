import torch
from torch import nn


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=1)


class UnimodalEncoder(nn.Module):
    """Three-layer encoder for each modality following DeepSER structure"""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(MultimodalEncoder, self).__init__()
        
        # Layer 1
        self.layer1_linear = nn.Linear(embed_dim, hidden_dim)
        self.layer1_norm = nn.LayerNorm(hidden_dim)
        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2, 
            dropout=dropout, batch_first=True
        )
        self.layer1_transformer = nn.TransformerEncoder(encoder_layer1, num_layers=1)
        
        # Layer 2  
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2,
            dropout=dropout, batch_first=True
        )
        self.layer2_transformer = nn.TransformerEncoder(encoder_layer2, num_layers=1)
        self.layer2_norm = nn.LayerNorm(hidden_dim)
        
        # Layer 3
        encoder_layer3 = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*2,
            dropout=dropout, batch_first=True
        )
        self.layer3_transformer = nn.TransformerEncoder(encoder_layer3, num_layers=1)
        self.layer3_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.pooling = MeanPooling()

    def forward(self, x):
        """
        Returns three layer outputs as required by DeepSER algorithm
        Returns:
            h1, h2, h3: Sequential representations from each layer
        """
        # Layer 1: h1
        x = self.layer1_linear(x)
        x = self.layer1_norm(x)
        h1 = self.layer1_transformer(x)
        h1 = self.dropout(h1)
        
        # Layer 2: h2  
        h2 = self.layer2_transformer(h1)
        h2 = self.layer2_norm(h2)
        h2 = self.dropout(h2)
        
        # Layer 3: h3
        h3 = self.layer3_transformer(h2)
        h3 = self.layer3_norm(h3)
        h3 = self.dropout(h3)
        
        return h1, h2, h3


class FusionEncoder(nn.Module):
    """Encoder for fusion representations"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(FusionEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=8, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.pooling = MeanPooling()

    def forward(self, x):
        x = self.transformer(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x
