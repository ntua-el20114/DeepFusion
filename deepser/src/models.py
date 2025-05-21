import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional
import random


class HardMultimodalMasking(nn.Module):
   def __init__(
       self,
       p: float = 0.5,
       n_modalities: int = 3,
       p_mod: Optional[List[float]] = None,
       masking: bool = True,
       m3_sequential: bool = True,
   ):
       """M^3 layer implementation

       For each sample in a batch mask one of the modalities with probability p.
       When dealing with sequential data it randomly masks an instance at every timestep.

       Args:
           p (float): mask/drop probability, 1-p is the prob to leave the sequence unaffected
           n_modalities (int): number of modalities
           p_mod (Optional[List[float]]): Mask probabilities for each modality
           masking (bool): masking flag variable, when False uses resca;ing trick
           m3_sequential (bool): mask different instances of the sequence for each modality
       """
       super(HardMultimodalMasking, self).__init__()
       self.p = p
       self.n_modalities = n_modalities
       self.masking = masking
       self.m3_sequential = m3_sequential

       self.p_mod = [1.0 / n_modalities for _ in range(n_modalities)]

       if p_mod is not None:
           self.p_mod = p_mod

   def forward(self, *mods):
       """Fast M^3 forward implementation

       Iterate over batch and timesteps and randomly choose modality to mask

       Args:
           mods (varargs torch.Tensor): [B, L, D_m] Modality representations

       Returns:
           (List[torch.Tensor]): The modality representations. Some of them are dropped
       """
       mods = list(mods)

       # List of [B, L, D]

       if self.training:
           if random.random() < self.p:
               # mask different modality for each sample in batch

               if self.m3_sequential: # mask different modality at every timestep
                   bsz, seqlen = mods[0].size(0), mods[0].size(1)
                   p_modal = torch.distributions.categorical.Categorical(
                       torch.tensor(self.p_mod)
                   )
                   m_cat = p_modal.sample((bsz, seqlen)).to(mods[0].device)
                   for m in range(self.n_modalities):
                       mask = torch.where(m_cat == m, 0, 1).unsqueeze(2)
                       mods[m] = mods[m] * mask

               else:
                   for batch in range(mods[0].size(0)):
                       m = random.choices(
                           list(range(self.n_modalities)), weights=self.p_mod, k=1
                       )[0]

                       # m = random.randint(0, self.n_modalities - 1)
                       mask = torch.ones_like(mods[m])
                       mask[batch] = 0.0
                       mods[m] = mods[m] * mask

       # rescaling trick
       if not self.masking:
           if self.p > 0:
               for m in range(len(mods)):
                   keep_prob = 1 - (self.p / self.n_modalities)
                   mods[m] = mods[m] * (1 / keep_prob)

       return mods

   def __repr__(self):
       shout = (
           self.__class__.__name__
           + "("
           + "p_mask="
           + str(self.p)
           + ", masking="
           + str(self.masking)
           + ", sequential="
           + str(self.m3_sequential)
           + ", p_mod="
           + str(self.p_mod)
           + ")"
       )
       return shout


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=1)


# --- Mixup Utility ---
def mixup_features(features, labels, alpha=0.2):
    batch_size = features.size(0)
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(features.device) if alpha > 0 else torch.ones(1).to(features.device)
    index = torch.randperm(batch_size).to(features.device)
    mixed_features = lam * features + (1 - lam) * features[index, :]
    if len(labels.shape) > 1:
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    else:
        mixed_labels = (labels, labels[index], lam)
    return mixed_features, mixed_labels


class DeepLeg(nn.Module):
    def __init__(self, embed_dim, mlp_out, dropout=0.1):
        super(DeepLeg, self).__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_out)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out, mlp_out)
        encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out, nhead=1, batch_first=True)
        self.transformer1 = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pooling = MeanPooling()

    def forward(self, x):
        x = self.relu(self.linear2(self.relu(self.linear1(x))))
        x = self.transformer1(x)
        f = self.transformer1.layers[0](x)
        s = self.transformer1.layers[-1](x)
        x = self.pooling(x)
        return x, f, s


# --- Gated Multimodal Unit Implementation ---
class GatedMultimodalUnit(nn.Module):
    def __init__(self, dim_1, dim_2, output_dim, dropout=0.3):
        """
        Gated Multimodal Unit for two modalities fusion.
        
        Args:
            dim_1 (int): First modality dimension
            dim_2 (int): Second modality dimension
            output_dim (int): Output dimension after fusion
            dropout (float): Dropout probability
        """
        super(GatedMultimodalUnit, self).__init__()
        self.transform_1 = nn.Linear(dim_1, output_dim)
        self.transform_2 = nn.Linear(dim_2, output_dim)
        self.gate = nn.Linear(dim_1 + dim_2, 2)
        self.gate.bias.data = torch.tensor([1.0, 1.0])
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x_1, x_2):
        h_1 = torch.tanh(self.transform_1(x_1))
        h_2 = torch.tanh(self.transform_2(x_2))
        
        # Apply dropout to modality representations before fusion
        h_1 = self.dropout(h_1)
        h_2 = self.dropout(h_2)
        
        # Compute gates
        combined = torch.cat([x_1, x_2], dim=1)
        gates = torch.softmax(self.gate(combined), dim=1)
        g_1, g_2 = gates.chunk(2, dim=1)
        
        # Apply gating and fusion
        h_fusion = self.dropout(g_1 * h_1 + g_2 * h_2)
        
        return h_fusion, gates


# --- Three-Way Gated Multimodal Unit ---
class ThreeWayGMU(nn.Module):
    def __init__(self, dim_l, dim_a, dim_v, output_dim, dropout=0.3):
        """
        Three-way Gated Multimodal Unit for fusion of three modalities.
        
        Args:
            dim_l (int): Linguistic modality dimension
            dim_a (int): Audio modality dimension
            dim_v (int): Visual modality dimension
            output_dim (int): Output dimension after fusion
            dropout (float): Dropout probability
        """
        super(ThreeWayGMU, self).__init__()
        self.transform_l = nn.Linear(dim_l, output_dim)
        self.transform_a = nn.Linear(dim_a, output_dim)
        self.transform_v = nn.Linear(dim_v, output_dim)
        self.gate = nn.Linear(dim_l + dim_a + dim_v, 3)
        self.gate.bias.data = torch.tensor([1.0, 1.0, 1.0])
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x_l, x_a, x_v):
        h_l = torch.tanh(self.transform_l(x_l))
        h_a = torch.tanh(self.transform_a(x_a))
        h_v = torch.tanh(self.transform_v(x_v))
        
        # Apply dropout to modality representations before fusion
        h_l = self.dropout(h_l)
        h_a = self.dropout(h_a)
        h_v = self.dropout(h_v)
        
        # Compute gates
        combined = torch.cat([x_l, x_a, x_v], dim=1)
        gates = torch.softmax(self.gate(combined), dim=1)
        g_l, g_a, g_v = gates.chunk(3, dim=1)
        
        # Apply gating and fusion
        h_fusion = self.dropout(g_l * h_l + g_a * h_a + g_v * h_v)
        
        return h_fusion, gates


# --- Updated TrimodalDeepLeg with GMU ---
class TrimodalDeepLeg(nn.Module):
    """
    A model designed to handle 3 input modalities separately with dedicated encoders,
    applying M^3 masking on the inputs, then fuse using Gated Multimodal Units (GMUs).
    """
    def __init__(self, dim_l, dim_a, dim_v, mlp_out1, mlp_out2, mlp_out3, dropout=0.1, mixup=0.0, m3_p=0.5):
        super(TrimodalDeepLeg, self).__init__()
        self.mixup = mixup
        
        # Initialize M^3 masking layer for 3 modalities
        self.m3 = HardMultimodalMasking(p=m3_p, n_modalities=3)
        
        # Three separate legs for each modality
        self.leg_l = DeepLeg(dim_l, mlp_out1, dropout)
        self.leg_a = DeepLeg(dim_a, mlp_out1, dropout)
        self.leg_v = DeepLeg(dim_v, mlp_out1, dropout)
        
        # Pairwise fusion using GMUs instead of concatenation for the sequence-level features
        self.gmu_la = GatedMultimodalUnit(mlp_out1, mlp_out1, mlp_out2, dropout)
        self.gmu_lv = GatedMultimodalUnit(mlp_out1, mlp_out1, mlp_out2, dropout)
        self.gmu_av = GatedMultimodalUnit(mlp_out1, mlp_out1, mlp_out2, dropout)
        
        # Transformer encoders for the fused representations (after GMU)
        encoder_layer = nn.TransformerEncoderLayer(d_model=mlp_out2, nhead=1, batch_first=True)
        self.transformer_la = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_lv = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_av = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final fusion using three-way GMU
        self.final_gmu = ThreeWayGMU(mlp_out1, mlp_out1, mlp_out1, mlp_out2, dropout)
        self.transformer_final = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Final linear layer to combine all processed outputs
        self.final_fusion = nn.Linear(mlp_out2 * 4, mlp_out3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(mlp_out3, 8)
        self.regressor = nn.Linear(mlp_out3, 3)
        
        # Store the modal weight history
        self.register_buffer("last_modal_weights", torch.tensor([0.33, 0.33, 0.33]))

    def forward(self, x_l, x_a, x_v, labels=None, dev=False):
        # Apply M^3 masking to the raw modality inputs
        x_l, x_a, x_v = self.m3(x_l, x_a, x_v)
        
        # Encode each modality
        x_l_p, f_l, s_l = self.leg_l(x_l)
        x_a_p, f_a, s_a = self.leg_a(x_a)
        x_v_p, f_v, s_v = self.leg_v(x_v)
        
        # --- Pairwise GMU fusion followed by transformer processing ---
        # Create sequence-level fusion features using GMUs
        # For each pair of features from transformer outputs
        batch_size, seq_len, _ = f_l.size()
        
        # Reshape for GMU processing - process each sequence position with GMU
        f_l_flat = f_l.reshape(batch_size * seq_len, -1)
        f_a_flat = f_a.reshape(batch_size * seq_len, -1)
        f_v_flat = f_v.reshape(batch_size * seq_len, -1)
        
        # Apply GMU fusion for each pair at sequence level
        f_la_flat, gates_la = self.gmu_la(f_l_flat, f_a_flat)
        f_lv_flat, gates_lv = self.gmu_lv(f_l_flat, f_v_flat)
        f_av_flat, gates_av = self.gmu_av(f_a_flat, f_v_flat)
        
        # Reshape back to sequence form
        f_la = f_la_flat.reshape(batch_size, seq_len, -1)
        f_lv = f_lv_flat.reshape(batch_size, seq_len, -1)
        f_av = f_av_flat.reshape(batch_size, seq_len, -1)
        
        # Process through transformers
        f_la = self.transformer_la(f_la)
        f_lv = self.transformer_lv(f_lv)
        f_av = self.transformer_av(f_av)
        
        # Apply mean pooling to get sequence representations
        x_la = torch.mean(f_la, dim=1)
        x_lv = torch.mean(f_lv, dim=1)
        x_av = torch.mean(f_av, dim=1)
        
        # Direct three-way fusion using GMU at the sequence level (using pooled features)
        x_lav, gates_lav = self.final_gmu(x_l_p, x_a_p, x_v_p)
        
        # Update modal weights for monitoring
        self.last_modal_weights = gates_lav.detach().mean(0)
        
        # Combine all outputs
        combined = torch.cat((x_la, x_lv, x_av, x_lav), dim=1)
        fused = self.final_fusion(combined)
        
        # Mixup if training
        if not dev and labels is not None and torch.rand(1) < self.mixup:
            fused, labels = mixup_features(fused, labels)
        
        fused = self.dropout(self.relu(fused))
        class_out = self.classifier(fused)
        reg_out = 1 + 6 * torch.sigmoid(self.regressor(fused))
        
        if self.mixup > 0.0 and labels is not None and not dev:
            return class_out, reg_out, labels
        return class_out, reg_out, self.last_modal_weights


class MULTModel(nn.Module):
    """
    Wrapper around TrimodalDeepLeg with GMU to maintain the same interface as the original MULTModel.
    This allows for a drop-in replacement without modifying training and main files.
    """
    def __init__(self, hyp_params):
        """
        Initialize with hyperparameters consistent with the original MULT model interface
        but using the TrimodalDeepLeg architecture with GMU fusion internally.
        """
        super(MULTModel, self).__init__()
        
        # Store the original dimensions for compatibility
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        
        # Set hidden dimensions for the model parts
        mlp_out1 = 512  # First-level output dimension for each modality
        mlp_out2 = 512  # Second-level fusion dimension
        mlp_out3 = 512  # Final fusion dimension
        
        # Get dropout from hyperparameters or use default
        dropout = hyp_params.out_dropout if hasattr(hyp_params, 'out_dropout') else 0.1
        
        # Set mixup parameter (can be adjusted based on hyperparameters if needed)
        mixup = 0.0
        if hasattr(hyp_params, 'mixup'):
            mixup = hyp_params.mixup
        
        # M3 masking probability
        m3_p = 0.5
        if hasattr(hyp_params, 'm3_p'):
            m3_p = hyp_params.m3_p
        
        # Initialize the trimodal deep leg model with GMU fusion
        self.model = TrimodalDeepLeg(
            dim_l=self.orig_d_l,
            dim_a=self.orig_d_a,
            dim_v=self.orig_d_v,
            mlp_out1=mlp_out1,
            mlp_out2=mlp_out2,
            mlp_out3=mlp_out3,
            dropout=dropout,
            mixup=mixup,
            m3_p=m3_p
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
        self.register_buffer("last_modal_weights", torch.tensor([0.33, 0.33, 0.33]))

    def forward(self, x_l, x_a, x_v, epoch=None, steps_per_epoch=None, labels=None, dev=False):
        """
        Forward pass that maintains the same interface as the original MULTModel
        but uses the TrimodalDeepLeg architecture with GMU fusion internally.
        
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
        # Process through the trimodal model with GMU fusion
        class_out, reg_out, modal_weights = self.model(x_l, x_a, x_v, labels, dev)
        
        # Update modal weights for compatibility
        self.last_modal_weights = modal_weights
        
        # Apply final projection if needed
        if hasattr(self, 'output_layer'):
            output = self.output_layer(class_out)
        else:
            output = class_out
        
        # For compatibility, use the regression output as the fused representation
        fused_representation = reg_out
        
        return output, fused_representation
