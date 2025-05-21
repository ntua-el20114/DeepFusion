import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
import random


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x):
        # Mean pooling along the sequence dimension
        # x is expected to be shape [batch_size, seq_len, hidden_dim]
        return torch.mean(x, dim=1)


def mixup_features(features, labels, alpha=0.2):
    """
    Applies mixup to features and corresponding labels
    
    Args:
        features: Feature tensors of shape [batch_size, feat_dim]
        labels: Labels of shape [batch_size, num_classes] (one-hot) or [batch_size]
        alpha: Mixup interpolation strength parameter
    
    Returns:
        Mixed features and labels
    """
    batch_size = features.size(0)
    
    # Sample mixup coefficient
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(features.device)
    else:
        lam = torch.ones(1).to(features.device)
    
    # Create random permutation indices
    index = torch.randperm(batch_size).to(features.device)
    
    # Mix the features
    mixed_features = lam * features + (1 - lam) * features[index, :]
    
    # If labels are one-hot encoded
    if len(labels.shape) > 1:
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    else:
        # For non-one-hot labels, we return both labels and the mixup coefficient
        mixed_labels = (labels, labels[index], lam)
    
    return mixed_features, mixed_labels


class DeepLeg(nn.Module):
    def __init__(self, embed_dim, mlp_out, dropout=0.1):
        """
        A simple, dysfunctional leg.

        Args:
            embed_dim (int): Dimensionality of input and output embeddings.
            mlp_out (int): Dimensionality of the feed-forward layer.
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
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        
        Returns:
            torch.Tensor: Output tensor after processing.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        x = self.transformer1(x)

        f = self.transformer1.layers[0](x)
        s = self.transformer1.layers[-1](x)
        
        x = self.pooling(x)
        
        return x, f, s


class DeepSkeletonMulti(nn.Module):
    def __init__(self, embed_dim1, mlp_out1, embed_dim2, mlp_out2, mlp_out3, dropout=0.1, mixup=0.0):
        """
        A simple skeleton with two legs.
        """
        super(DeepSkeletonMulti, self).__init__()

        self.mixup = mixup

        self.leg1 = DeepLeg(embed_dim1, mlp_out1)
        self.leg2 = DeepLeg(embed_dim2, mlp_out2)
        self.leg3 = DeepLeg(mlp_out1, mlp_out1)

        self.linear1 = nn.Linear(mlp_out1 + mlp_out2 + mlp_out1, mlp_out3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out3, 8)
        self.linear3 = nn.Linear(mlp_out3, 3)

    def forward(self, x1, x2, labels=None, dev=False):
        x1, f1, s1 = self.leg1(x1)
        x2, f2, s2 = self.leg2(x2)

        f12 = torch.cat((f1, f2), dim=1)
        xf, ff, sf = self.leg3(f12)

        f12 = torch.cat((s1, ff, s2), dim=1)
        x3, f3, s3 = self.leg3(f12)

        x = torch.cat((x1, x3, x2), dim=-1)
        x = self.linear1(x)

        p = torch.rand(size=(1,), device=x.device)

        if p < self.mixup and labels is not None and not dev:
            x, labels = mixup_features(x, labels)

        x = self.relu(x)
        y = self.linear3(x)

        x = self.linear2(x)
        y = 1 + 6*torch.sigmoid(y)
        
        if self.mixup > 0.0 and labels is not None and not dev:
            return x, y, labels
        else:
            return x, y


class TrimodalDeepLeg(nn.Module):
    """
    A model designed to handle 3 input modalities separately with dedicated encoders
    and then fuse them together for classification/regression.
    """
    def __init__(self, dim_l, dim_a, dim_v, mlp_out1, mlp_out2, mlp_out3, dropout=0.1, mixup=0.0):
        super(TrimodalDeepLeg, self).__init__()
        
        self.mixup = mixup
        
        # Three separate legs for each modality
        self.leg_l = DeepLeg(dim_l, mlp_out1, dropout)
        self.leg_a = DeepLeg(dim_a, mlp_out1, dropout)
        self.leg_v = DeepLeg(dim_v, mlp_out1, dropout)
        
        # Fusion legs
        self.fusion_leg1 = DeepLeg(mlp_out1 * 2, mlp_out2, dropout)  # For L-A fusion
        self.fusion_leg2 = DeepLeg(mlp_out1 * 2, mlp_out2, dropout)  # For L-V fusion
        self.fusion_leg3 = DeepLeg(mlp_out1 * 2, mlp_out2, dropout)  # For A-V fusion
        
        # Final fusion and output layers
        self.final_fusion = nn.Linear(mlp_out1 * 3 + mlp_out2 * 3, mlp_out3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Classification and regression heads
        self.classifier = nn.Linear(mlp_out3, 8)
        self.regressor = nn.Linear(mlp_out3, 3)
    
    def forward(self, x_l, x_a, x_v, labels=None, dev=False):
        # Process each modality separately
        x_l_pooled, f_l, s_l = self.leg_l(x_l)
        x_a_pooled, f_a, s_a = self.leg_a(x_a)
        x_v_pooled, f_v, s_v = self.leg_v(x_v)
        
        # Pairwise fusion of modalities
        # L-A fusion
        f_la = torch.cat((f_l, f_a), dim=2)
        x_la, f_la_mid, s_la = self.fusion_leg1(f_la)
        
        # L-V fusion
        f_lv = torch.cat((f_l, f_v), dim=2)
        x_lv, f_lv_mid, s_lv = self.fusion_leg2(f_lv)
        
        # A-V fusion
        f_av = torch.cat((f_a, f_v), dim=2)
        x_av, f_av_mid, s_av = self.fusion_leg3(f_av)
        
        # Combine all features for final prediction
        combined = torch.cat((x_l_pooled, x_a_pooled, x_v_pooled, x_la, x_lv, x_av), dim=1)
        
        # Apply final fusion
        fused = self.final_fusion(combined)
        
        # Apply mixup if training
        p = torch.rand(size=(1,), device=fused.device)
        if p < self.mixup and labels is not None and not dev:
            fused, labels = mixup_features(fused, labels)
        
        # Apply nonlinearity and dropout
        fused = self.dropout(self.relu(fused))
        
        # Get classification and regression outputs
        class_out = self.classifier(fused)
        reg_out = self.regressor(fused)
        reg_out = 1 + 6 * torch.sigmoid(reg_out)  # Scale regression output to [1, 7]
        
        if self.mixup > 0.0 and labels is not None and not dev:
            return class_out, reg_out, labels
        else:
            return class_out, reg_out


class MULTModel(nn.Module):
    """
    Wrapper around TrimodalDeepLeg to maintain the same interface as the original MULTModel.
    This allows for a drop-in replacement without modifying training and main files.
    """
    def __init__(self, hyp_params):
        """
        Initialize with hyperparameters consistent with the original MULT model interface
        but using the TrimodalDeepLeg architecture internally.
        """
        super(MULTModel, self).__init__()
        
        # Store the original dimensions for compatibility
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        
        # Set hidden dimensions for the model parts
        mlp_out1 = 512 # First-level output dimension for each modality
        mlp_out2 = 512  # Second-level fusion dimension
        mlp_out3 = 512  # Final fusion dimension
        
        # Get dropout from hyperparameters or use default
        dropout = hyp_params.out_dropout if hasattr(hyp_params, 'out_dropout') else 0.1
        
        # Set mixup parameter (can be adjusted based on hyperparameters if needed)
        mixup = 0.0
        if hasattr(hyp_params, 'mixup'):
            mixup = hyp_params.mixup
        
        # Initialize the trimodal deep leg model
        self.model = TrimodalDeepLeg(
            dim_l=self.orig_d_l,
            dim_a=self.orig_d_a,
            dim_v=self.orig_d_v,
            mlp_out1=mlp_out1,
            mlp_out2=mlp_out2,
            mlp_out3=mlp_out3,
            dropout=dropout,
            mixup=mixup
        )
        
        # Store the output dimension for compatibility
        self.output_dim = hyp_params.output_dim
        
        # Add a final projection layer if the output dimensions don't match
        if self.output_dim != 8:  # 8 is the default classifier output dimension
            self.output_layer = nn.Linear(8, self.output_dim)
        
        # Store various parameters from the original model for compatibility
        self.vonly = hyp_params.vonly if hasattr(hyp_params, 'vonly') else False
        self.aonly = hyp_params.aonly if hasattr(hyp_params, 'aonly') else False
        self.lonly = hyp_params.lonly if hasattr(hyp_params, 'lonly') else False
        
        # Initialize modal weights attribute for compatibility
        self.last_modal_weights = torch.tensor([0.33, 0.33, 0.33])

    def forward(self, x_l, x_a, x_v, epoch=None, steps_per_epoch=None, labels=None, dev=False):
        """
        Forward pass that maintains the same interface as the original MULTModel
        but uses the TrimodalDeepLeg architecture internally.
        
        Args:
            x_l (torch.Tensor): Language input tensor [batch_size, seq_len, d_l]
            x_a (torch.Tensor): Audio input tensor [batch_size, seq_len, d_a]
            x_v (torch.Tensor): Visual input tensor [batch_size, seq_len, d_v]
            epoch (int, optional): Current epoch (for compatibility)
            steps_per_epoch (int, optional): Steps per epoch (for compatibility)
            labels (torch.Tensor, optional): Labels for mixup
            dev (bool, optional): Whether in dev/eval mode
            
        Returns:
            tuple: (output logits, fused representation)
        """
        # Process through the trimodal model
        class_out, reg_out = self.model(x_l, x_a, x_v, labels, dev)
        
        # Apply final projection if needed
        if hasattr(self, 'output_layer'):
            output = self.output_layer(class_out)
        else:
            output = class_out
        
        # For compatibility, use the regression output as the fused representation
        fused_representation = reg_out
        
        # Update modal weights for compatibility (equal weights for simplicity)
        self.last_modal_weights = torch.tensor([[1/3, 1/3, 1/3]], device=output.device)
        
        return output, fused_representation