from torch import nn
from modules.enc_modules import *



class DeepLeg(nn.Module):
    def __init__(self, embed_dim, mlp_out, dropout=0.1):
        """
        The original DeepSER Encoder, as implemented in the original repo.
        Note that this implementation seems to disagree with the paper description.

        Instead of:                                 We would probably expect:
        x = self.transformer1(x)                    f = self.transformer1.layers[0](x)
        f = self.transformer1.layers[0](x)          s = self.transformer1.layers[-1](f)
        s = self.transformer1.layers[-1](x)         ...

        So we might want to create a variation of this encoder that follows the paper description more closely.

        Args:
            embed_dim (int): Dimensionality of input and output embeddings.
            ff_dim (int): Dimensionality of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super(DeepLeg, self).__init__()

        self.linear1 = nn.Linear(embed_dim, mlp_out)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out, mlp_out)

        encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out, nhead=1, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.transformer2 = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.pooling = MeanPooling()


    def forward(self, x):
        """
        Forward pass of the Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # import pdb; pdb.set_trace()

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        # print(x.shape)
        x = self.transformer1(x)

        f = self.transformer1.layers[0](x)
        s = self.transformer1.layers[-1](x)
        
        # print(x.shape, f.shape, s.shape)
        # exit(0)
        #print(x.shape)
        x = self.pooling(x)
        # print(x.shape)
        
        return x, f, s


class BaseEnc(nn.Module):
    def __init__(self, embed_dim, mlp_out, dropout=0.1):
        """
        Modified DeepLeg encoder, so that it follows the paper description.
        Note that this implementation seems to disagree with the paper description.

        Args:
            embed_dim (int): Dimensionality of input and output embeddings.
            ff_dim (int): Dimensionality of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super(BaseEnc, self).__init__()

        self.linear1 = nn.Linear(embed_dim, mlp_out)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out, mlp_out)

        encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out, nhead=1, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.pooling = MeanPooling()

    def forward(self, x):
        """
        Forward pass of the Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        # import pdb; pdb.set_trace()

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        f = self.transformer1.layers[0](x)   # h1 <- TransformerLayer(x)
        s = self.transformer1.layers[-1](f)  # h2 <- TransformerLayer(h1)
        x = self.pooling(s)                  # h3 <- POOL(h2)
        
        return x, f, s


class VarDepthEnc(nn.Module):
    def __init__(self, embed_dim, mlp_out, prepool=2, postpool=0, dropout=0.1):
        """
        A BaseEnc encoder, with the ability to vary the depth of the Transformer layers.
        The original BaseEnc would have prepool=2 and postpool=0.

        Args:
            embed_dim (int): Dimensionality of input and output embeddings.
            ff_dim (int): Dimensionality of the feed-forward layer.
            prepool: int): Number of Transformer layers before pooling.
            postpool (int): Number of Transformer layers after pooling.
            dropout (float): Dropout rate.
        """
        super(VarDepthEnc, self).__init__()
        assert prepool >= 1, "prepool must be >= 1"
        assert postpool >= 0, "postpool must be non-negative"

        self.linear1 = nn.Linear(embed_dim, mlp_out)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out, mlp_out)

        encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out, nhead=1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=prepool, enable_nested_tensor=False)

        #self.pooling = MeanPooling()
        self.pooling = MultiHeadAttentionPooling(mlp_out, num_heads=8)  # Use attention pooling instead of mean pooling

        # self.linear_layers = LinearEncoder(mlp_out, postpool, dropout=dropout)
        # self.postpool_layers = BottleneckMLP(mlp_out, postpool, factor=2, dropout=dropout)
        # self.postpool_layers = MultiScaleEncoder(mlp_out, postpool, dropout=dropout)
        self.postpool_layers = ConvolutionalEncoder(mlp_out, postpool, dropout=dropout)


    def forward(self, x):
        """
        Forward pass of the Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
        
        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        prepool = [x] # Store pre-pooling representations
        for layer in self.transformer.layers:
            prepool.append(layer(prepool[-1])) 
        prepool.pop(0)  # The first element is the LL output, and we need it only for the loop

        x = self.pooling(prepool[-1])  # Apply pooling on the last pre-pooling representation

        if not self.postpool_layers:
            return prepool+[x]  # Return all pre-pooling representations and the pooled representation
        
        postpool = [x]  # Store post-pooling representations
        for layer in self.postpool_layers:
            postpool.append(layer(postpool[-1]))

        return prepool + postpool  # Return all pre-pooling and post-pooling representations

class UnimodalEncoder(nn.Module):
    """Three-layer encoder for each modality following DeepSER structure"""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super(UnimodalEncoder, self).__init__()
        
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
