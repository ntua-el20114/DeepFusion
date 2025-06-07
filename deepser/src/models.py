import torch
from torch import nn
from typing import List, Optional, Tuple

# Import modular components
from modules.m3 import HardMultimodalMasking
from modules.gmu import ThreeWayGMU, FourWayGMU, TwoWayGMU  
from modules.encoders import DeepLeg, BaseEnc, UnimodalEncoder, FusionEncoder
from modules.attention import MultiHeadAttention
from modules.mixup import mixup_features


class DeepSERBase(nn.Module):
    def __init__(self, embed_dim1, mlp_out1, embed_dim2, mlp_out2, embed_dim3, mlp_out3, mlp_out4, encoder = "code", dropout=0.1, mixup=0.0):
        """

        DON'T CHANGE THIS MODEL
        -----------------------
        This is here for refrence to the original model.
        If you want to perform any changes, make a designated copy.

        The original DeepSER model, as implemented in the medusa repo (class DeepTriSkeletonMulti).

        Note that this model uses the original DeepLeg encoder from the repo, that seems to disagree with the 
        paper description of the model. To use an encoder that follows the paper, you can use the parameter:

        encoder = "code"/"paper"
        code: use the DeepLeg encoder, like the original repo
        paper: use a custom encoder that follows the paper description (not yet implemented)


        You can call this model through the MultModel wrapper, using the parameters:

        self.model = DeepSERBase(
            embed_dim1=self.orig_d_l,    # Language/text modality input dimension
            mlp_out1=hidden_dim,         # Output dimension for first leg
            embed_dim2=self.orig_d_a,    # Audio modality input dimension  
            mlp_out2=hidden_dim,         # Output dimension for second leg
            embed_dim3=self.orig_d_v,    # Visual modality input dimension
            mlp_out3=hidden_dim,         # Output dimension for third leg
            mlp_out4=hidden_dim,         # Final output dimension for legs 4&5
            encoder="code",              # Use original encoder implementation
            dropout=dropout,             # Dropout rate
            mixup=mixup                  # Mixup probability
        )
 
        """

        super(DeepSERBase, self).__init__()

        self.mixup = mixup

        if encoder == "code":
            self.leg1 = DeepLeg(embed_dim1, mlp_out1)
            self.leg2 = DeepLeg(embed_dim2, mlp_out2)
            self.leg3 = DeepLeg(embed_dim3, mlp_out3)
            self.leg4 = DeepLeg(mlp_out1, mlp_out4)
            self.leg5 = DeepLeg(mlp_out1, mlp_out4)
        elif encoder == "paper":
            self.leg1 = BaseEnc(embed_dim1, mlp_out1)
            self.leg2 = BaseEnc(embed_dim2, mlp_out2)
            self.leg3 = BaseEnc(embed_dim3, mlp_out3)
            self.leg4 = BaseEnc(mlp_out1, mlp_out4)
            self.leg5 = BaseEnc(mlp_out1, mlp_out4)
        else:
            raise ValueError("Encoder must be 'code' or 'paper'.")

        self.linear1 = nn.Linear(mlp_out1 + mlp_out2 + mlp_out3 + mlp_out4, mlp_out4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out4, 8)
        self.linear3 = nn.Linear(mlp_out4, 3)
        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, x1, x2, x3, labels=None, dev=False):
        x1, f1, s1 = self.leg1(x1)
        x2, f2, s2 = self.leg2(x2)
        x3, f3, s3 = self.leg3(x3)

        cf = torch.cat((f1,f2,f3), dim = 1)

        xc1, fc1, sc1 = self.leg4(cf)

        cs = torch.cat((s1,s2,s3,sc1), dim = 1)

        xc2, fc2, sc2 = self.leg5(cs)

        x = torch.cat((x1, x2, x3, xc2), dim=-1)

        x = self.linear1(x)

        p = torch.rand(size=(1,), device=x.device)

        if p < self.mixup and labels is not None and not dev:
            x, labels = mixup_features(x, labels)

        x = self.relu(x)
        y = self.linear3(x)

        x = self.linear2(x)
        y = 1 + 6*torch.sigmoid(y)
        # x = self.softmax(x)
        
        if self.mixup > 0.0 and labels is not None and not dev:
            return x, y, labels
        else:
            return x, y


class DeepSERModel(nn.Module):
   """
   Implementation of DeepSER algorithm with GMU fusion at all 3 layers + attention
   Following Algorithm 2 structure with enhanced final fusion
   """
   def __init__(self, dim_h, dim_g, dim_z, hidden_dim=512, dropout=0.1, mixup=0.0, m3_p=0.5):
       super(DeepSERModel, self).__init__()
       self.mixup = mixup
      
       # M^3 masking layer
       self.m3 = HardMultimodalMasking(p=m3_p, n_modalities=3)
      
       # Unimodal encoders (ENCh, ENCg, ENCz)
       self.enc_h = UnimodalEncoder(dim_h, hidden_dim, dropout)
       self.enc_g = UnimodalEncoder(dim_g, hidden_dim, dropout) 
       self.enc_z = UnimodalEncoder(dim_z, hidden_dim, dropout)
      
       # First fusion layer: f^(1) = ENCf^(1)(h1||g1||z1) using GMU
       self.fusion_gmu_1 = ThreeWayGMU(hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout)
       self.enc_f1 = FusionEncoder(hidden_dim, hidden_dim*2, dropout)
      
       # Second fusion layer: f^(2) = ENCf^(2)(h2||g2||z2||f2) using 4-way GMU
       self.fusion_gmu_2 = FourWayGMU(hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout)
       self.enc_f2 = FusionEncoder(hidden_dim, hidden_dim*2, dropout)
      
       # Final fusion with GMU + Attention instead of simple linear transformation
       self.final_gmu = FourWayGMU(hidden_dim, hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout)
       self.final_attention = MultiHeadAttention(hidden_dim, num_heads=8, dropout=dropout)
       self.final_fusion = nn.Linear(hidden_dim, hidden_dim)  # Final projection after attention
      
       # Output layers
       self.dropout = nn.Dropout(dropout)
       self.relu = nn.ReLU()
      
       self.classifier = nn.Linear(hidden_dim, 8)   # Wc
       self.regressor = nn.Linear(hidden_dim, 3)    # Wr
      
       # Store modal weights and attention weights
       self.register_buffer("last_modal_weights", torch.tensor([0.33, 0.33, 0.33]))
       self.register_buffer("last_attention_weights", torch.zeros(1, 8, 4, 4))


   def forward(self, x_h, x_g, x_z, labels=None, dev=False):
       """
       Forward pass following DeepSER Algorithm 2 exactly
      
       Args:
           x_h: h0 - First modality input
           x_g: g0 - Second modality input 
           x_z: z0 - Third modality input
           labels: Labels for mixup
           dev: Development/evaluation mode flag
       """
       # Apply M^3 masking to inputs
       x_h, x_g, x_z = self.m3(x_h, x_g, x_z)
      
       # Unimodal encodings: [h1, h2, h3] ← ENCh(h0), etc.
       h1, h2, h3 = self.enc_h(x_h)
       g1, g2, g3 = self.enc_g(x_g)
       z1, z2, z3 = self.enc_z(x_z)
      
       # First fusion: [_, f2, _] ← ENCf^(1)(h1||g1||z1)
       # Using GMU instead of concatenation for better fusion
       batch_size, seq_len, _ = h1.size()
      
       # Reshape for sequence-wise GMU processing
       h1_flat = h1.reshape(-1, h1.size(-1))
       g1_flat = g1.reshape(-1, g1.size(-1))
       z1_flat = z1.reshape(-1, z1.size(-1))
      
       f1_flat, gates_1 = self.fusion_gmu_1(h1_flat, g1_flat, z1_flat)
       f1 = f1_flat.reshape(batch_size, seq_len, -1)
      
       # Process through fusion encoder to get f2
       f1_encoded = self.enc_f1(f1)
       f2 = f1_encoded  # This is our f2 from the algorithm
      
       # Second fusion: [_, _, f3] ← ENCf^(2)(h2||g2||z2||f2)
       # Pool f2 to match h2, g2, z2 dimensions for fusion
       f2_pooled = f2.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
      
       # Reshape for sequence-wise GMU processing
       h2_flat = h2.reshape(-1, h2.size(-1))
       g2_flat = g2.reshape(-1, g2.size(-1))
       z2_flat = z2.reshape(-1, z2.size(-1))
       f2_flat = f2_pooled.reshape(-1, f2_pooled.size(-1))
      
       f2_fused_flat, gates_2 = self.fusion_gmu_2(h2_flat, g2_flat, z2_flat, f2_flat)
       f2_fused = f2_fused_flat.reshape(batch_size, seq_len, -1)
      
       # Process through second fusion encoder to get f3
       f2_encoded = self.enc_f2(f2_fused)
       f3 = f2_encoded.mean(dim=1)  # Pool to get sequence-level representation
      
       # Final fusion: Enhanced x ← GMU + Attention(h3||g3||z3||f3)
       h3_pooled = h3.mean(dim=1)  # Pool sequence dimension
       g3_pooled = g3.mean(dim=1)
       z3_pooled = z3.mean(dim=1)
      
       # Apply final GMU fusion
       x_gmu, final_gates = self.final_gmu(h3_pooled, g3_pooled, z3_pooled, f3)
      
       # Prepare for attention: stack modality representations
       # Shape: [batch_size, 4, hidden_dim] for (h3, g3, z3, f3)
       modal_stack = torch.stack([h3_pooled, g3_pooled, z3_pooled, f3], dim=1)
      
       # Apply multi-head attention across modalities
       attended_modalities, attention_weights = self.final_attention(modal_stack)
      
       # Combine GMU output with attention-weighted average
       attention_fused = attended_modalities.mean(dim=1)  # Average across modalities
      
       # Final combination: blend GMU and attention outputs
       x = 0.0 * x_gmu + 1.0 * attention_fused
       x = self.final_fusion(x)  # Final projection
      
       # Store attention weights for analysis
       self.last_attention_weights = attention_weights.detach()
      
       # MixUp augmentation: x, yc, yr ← MixUp(x, yc, yr)
       if not dev and labels is not None and torch.rand(1) < self.mixup:
           x, labels = mixup_features(x, labels)
      
       # Final outputs: yc ← Wc(σ(x)), yr ← Wr(σ(x))
       x_activated = self.relu(x)
       x_activated = self.dropout(x_activated)
      
       yc = self.classifier(x_activated)  # Classification output
       yr = 1 + 6 * torch.sigmoid(self.regressor(x_activated))  # Regression output
      
       # Update modal weights (using final layer gates for comprehensive monitoring)
       self.last_modal_weights = final_gates.detach().mean(0)[:3]  # First 3 components (h, g, z)
      
       if self.mixup > 0.0 and labels is not None and not dev:
           return yc, yr, labels
       return yc, yr, self.last_modal_weights




class MULTModel(nn.Module):
   """
   Wrapper around DeepSERModel to maintain interface compatibility
   """
   def __init__(self, hyp_params):
       super(MULTModel, self).__init__()
      
       # Store original dimensions
       self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
      
       # Model parameters
       hidden_dim = 1024
       dropout = hyp_params.out_dropout if hasattr(hyp_params, 'out_dropout') else 0.1
       mixup = hyp_params.mixup if hasattr(hyp_params, 'mixup') else 0.2
       m3_p = hyp_params.m3_p if hasattr(hyp_params, 'm3_p') else 0.4
      
       # Initialize DeepSER model
       self.model = DeepSERBase(
            embed_dim1=self.orig_d_l,    # Language/text modality input dimension
            mlp_out1=hidden_dim,         # Output dimension for first leg
            embed_dim2=self.orig_d_a,    # Audio modality input dimension  
            mlp_out2=hidden_dim,         # Output dimension for second leg
            embed_dim3=self.orig_d_v,    # Visual modality input dimension
            mlp_out3=hidden_dim,         # Output dimension for third leg
            mlp_out4=hidden_dim,         # Final output dimension for legs 4&5
            encoder="code",              # Use original encoder implementation
            dropout=dropout,             # Dropout rate
            mixup=mixup                  # Mixup probability
        )

       # Output dimension compatibility
       self.output_dim = hyp_params.output_dim
       if self.output_dim != 8:
           self.output_layer = nn.Linear(8, self.output_dim)
      
       # Compatibility attributes
       self.vonly = hyp_params.vonly if hasattr(hyp_params, 'vonly') else False
       self.aonly = hyp_params.aonly if hasattr(hyp_params, 'aonly') else False
       self.lonly = hyp_params.lonly if hasattr(hyp_params, 'lonly') else False
      
       self.register_buffer("last_modal_weights", torch.tensor([0.33, 0.33, 0.33]))
       self.register_buffer("last_attention_weights", torch.zeros(1, 8, 4, 4))


   def forward(self, x_l, x_a, x_v, epoch=None, steps_per_epoch=None, labels=None, dev=False):
       """
       Forward pass maintaining original interface
       Maps inputs to h, g, z as per DeepSER naming convention
       """
       # Process through DeepSER model (x_l->h, x_a->g, x_v->z)
       class_out, reg_out = self.model(x_l, x_a, x_v, labels, dev)
      
       # Update modal weights and attention weights
    #    self.last_modal_weights = modal_weights
    #    if hasattr(self.model, 'last_attention_weights'):
    #        self.last_attention_weights = self.model.last_attention_weights
      
       # Apply output projection if needed
       if hasattr(self, 'output_layer'):
           output = self.output_layer(class_out)
       else:
           output = class_out
      
       return output, reg_out
