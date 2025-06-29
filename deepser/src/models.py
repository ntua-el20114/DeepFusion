import torch
from torch import nn
from typing import List, Optional, Tuple

# Import modular components
from modules.m3 import HardMultimodalMasking
from modules.gmu import ThreeWayGMU, FourWayGMU, TwoWayGMU  
from modules.encoders import *
from modules.attention import MultiHeadAttention
from modules.mixup import mixup_features


class DeepSERBase(nn.Module):
    def __init__(self, embed_dim1, mlp_out1, embed_dim2, mlp_out2, embed_dim3, mlp_out3, mlp_out4, encoder = "paper", dropout=0.1, mixup=0.0):
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
        paper: use a custom encoder that follows the paper description


        You can call this model through the MultModel wrapper, using the parameters:

        self.model = DeepSERBase(
            embed_dim1=self.orig_d_l,    # Language/text modality input dimension
            mlp_out1=hidden_dim,         # Output dimension for first leg
            embed_dim2=self.orig_d_a,    # Audio modality input dimension  
            mlp_out2=hidden_dim,         # Output dimension for second leg
            embed_dim3=self.orig_d_v,    # Visual modality input dimension
            mlp_out3=hidden_dim,         # Output dimension for third leg
            mlp_out4=hidden_dim,         # Final output dimension for legs 4&5
            encoder="paper",              
            dropout=dropout,             # Dropout rate
            mixup=mixup                  # Mixup probability
        )
 
        """

        super(DeepSERBase, self).__init__()

        print(f"Using DeepSERBase: \n\tembed_dim1={embed_dim1}\n\tmlp_out1={mlp_out1}\n\tembed_dim2={embed_dim2}\n\tmlp_out2={mlp_out2}\n\tembed_dim3={embed_dim3}\n\tmlp_out3={mlp_out3}\n\tmlp_out4={mlp_out4}\n\tencoder={encoder}\n\tdropout={dropout}\n\tmixup={mixup}\n\t")
        
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


class VarDepthDeepSER(nn.Module):
    def __init__(self, embed_dim1, mlp_out1, embed_dim2, mlp_out2, embed_dim3, mlp_out3, mlp_out4, prepool=2, postpool=0, dropout=0.1, mixup=0.0):
        """
        A modified DeepSER model that allows for variable depth in the encoders.

        prepool (int): Number of Transformer layers before pooling.
        postpool (int): Number of Transformer layers after pooling.

        (The original DeepSER would have prepool=2 and postpool=0)

        Note: for now, the
        """

        super(VarDepthDeepSER, self).__init__()

        print(f"Using VarDepthDeepSER: \n\tembed_dim1={embed_dim1}\n\tmlp_out1={mlp_out1}\n\tembed_dim2={embed_dim2}\n\tmlp_out2={mlp_out2}\n\tembed_dim3={embed_dim3}\n\tmlp_out3={mlp_out3}\n\tmlp_out4={mlp_out4}\n\tprepool={prepool}\n\tpostpool={postpool}\n\tdropout={dropout}\n\tmixup={mixup}\n\t")


        self.prepool = prepool
        self.postpool = postpool

        self.mixup = mixup

        self.linear1 = nn.Linear(mlp_out1 + mlp_out2 + mlp_out3 + mlp_out4, mlp_out4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(mlp_out4, 8)
        self.linear3 = nn.Linear(mlp_out4, 3)

        self.leg1 = VarDepthEnc(embed_dim1, mlp_out1, prepool, postpool, dropout)
        self.leg2 = VarDepthEnc(embed_dim2, mlp_out2, prepool, postpool, dropout)
        self.leg3 = VarDepthEnc(embed_dim3, mlp_out3, prepool, postpool, dropout)

        self.prepool_fusion = [VarDepthEnc(mlp_out1, mlp_out4, prepool, postpool) for _ in range(prepool)]
        self.postpool_fusion = nn.ModuleList([
            nn.Sequential(nn.Linear(mlp_out1*4, mlp_out1), nn.ReLU()) 
            for _ in range(postpool)
        ])

    def forward(self, x1, x2, x3, labels=None, dev=False):
        h = self.leg1(x1)
        g = self.leg2(x2)
        z = self.leg3(x3)

        for i in range(self.prepool):
            if i==0:
                c = torch.cat((h[i], g[i], z[i]), dim=1)
            else:
                c = torch.cat((h[i], g[i], z[i], f), dim=1) # concat with previous fusion output

            # pass concatenated vectors through the fusion encoder
            f = self.prepool_fusion[i](c)

            if i == self.prepool-1:
                f = f[-1] # keep the last (pooled) representation
            else:
                f = f[self.prepool-1] # keep the last unpooled representation

        for i in range(self.postpool):
            # print(i, self.prepool+i, h[self.prepool+i].shape, f.shape, [h_j.shape for h_j in h])
            c = torch.cat((h[self.prepool+i], g[self.prepool+i], z[self.prepool+i], f), dim=1)

            # pass concatenated vectors through the fusion encoder
            f = self.postpool_fusion[i](c)

        # Concatenate the outputs of the three legs and the final fusion output
        x = torch.cat((h[-1], g[-1], z[-1], f), dim=-1)

        # Continue with the original DeepSER forward pass
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



class PairwiseVarDepthDeepSER(nn.Module):
  """
  Pairwise DeepSER implementation using VarDepthEncoder structure with 3 multimodal flows:
  - Flow 1: h–g (language–audio)
  - Flow 2: h–z (language–vision)
  - Flow 3: g–z (audio–vision)
   Uses VarDepthEncoder pattern with prepool and postpool layers for each pairwise flow.
  """
 def __init__(self, embed_dim1, mlp_out1, embed_dim2, mlp_out2, embed_dim3, mlp_out3,
               mlp_out4, prepool=2, postpool=0, dropout=0.1, mixup=0.0, m3_p=0.3):
      """
      Args:
          embed_dim1, embed_dim2, embed_dim3: Input dimensions for each modality
          mlp_out1, mlp_out2, mlp_out3, mlp_out4: Output dimensions
          prepool: Number of layers before pooling in VarDepthEncoder
          postpool: Number of layers after pooling in VarDepthEncoder
          dropout: Dropout rate
          mixup: Mixup probability
          m3_p: M3 masking probability
      """
      super(PairwiseVarDepthDeepSER, self).__init__()
    
      print(f"Using PairwiseVarDepthDeepSER: \n\tembed_dim1={embed_dim1}\n\tmlp_out1={mlp_out1}")
      print(f"\tembed_dim2={embed_dim2}\n\tmlp_out2={mlp_out2}\n\tembed_dim3={embed_dim3}")
      print(f"\tmlp_out3={mlp_out3}\n\tmlp_out4={mlp_out4}\n\tprepool={prepool}")
      print(f"\tpostpool={postpool}\n\tdropout={dropout}\n\tmixup={mixup}\n\tm3_p={m3_p}")
    
      self.prepool = prepool
      self.postpool = postpool
      self.mixup = mixup
    
      # M³ masking layer
      self.m3 = HardMultimodalMasking(p=m3_p, n_modalities=3)
    
      # Unimodal encoders using VarDepthEncoder
      self.leg1 = VarDepthEnc(embed_dim1, mlp_out1, prepool, postpool, dropout)  # h encoder
      self.leg2 = VarDepthEnc(embed_dim2, mlp_out2, prepool, postpool, dropout)  # g encoder
      self.leg3 = VarDepthEnc(embed_dim3, mlp_out3, prepool, postpool, dropout)  # z encoder
    
      # === PAIRWISE FLOW 1: h–g (Language–Audio) ===
      self.hg_prepool_fusion = nn.ModuleList()
      for i in range(prepool):
          # first layer: concat h+g (1024+1024)
          # subsequent layers: concat h+g+prev_repr (1024+1024+1024)
          in_dim = (mlp_out1 + mlp_out2) if i == 0 else (mlp_out1 + mlp_out2 + mlp_out4)
          self.hg_prepool_fusion.append(
              VarDepthEnc(in_dim, mlp_out4, 3, 1, dropout)
          )
      self.hg_postpool_fusion = nn.ModuleList([
          nn.Sequential(
              nn.Linear(mlp_out1 + mlp_out2 + mlp_out4, mlp_out4),
              nn.ReLU(),
              nn.Dropout(dropout)
          )
          for _ in range(postpool)
      ])
    
      # === PAIRWISE FLOW 2: h–z (Language–Vision) ===
      self.hz_prepool_fusion = nn.ModuleList()
      for i in range(prepool):
          in_dim = (mlp_out1 + mlp_out3) if i == 0 else (mlp_out1 + mlp_out3 + mlp_out4)
          self.hz_prepool_fusion.append(
              VarDepthEnc(in_dim, mlp_out4, 3, 1, dropout)
          )
      self.hz_postpool_fusion = nn.ModuleList([
          nn.Sequential(
              nn.Linear(mlp_out1 + mlp_out3 + mlp_out4, mlp_out4),
              nn.ReLU(),
              nn.Dropout(dropout)
          )
          for _ in range(postpool)
      ])
    
      # === PAIRWISE FLOW 3: g–z (Audio–Vision) ===
      self.gz_prepool_fusion = nn.ModuleList()
      for i in range(prepool):
          in_dim = (mlp_out2 + mlp_out3) if i == 0 else (mlp_out2 + mlp_out3 + mlp_out4)
          self.gz_prepool_fusion.append(
              VarDepthEnc(in_dim, mlp_out4, 3, 1, dropout)
          )




      self.gz_postpool_fusion = nn.ModuleList([
          nn.Sequential(
              nn.Linear(mlp_out2 + mlp_out3 + mlp_out4, mlp_out4),
              nn.ReLU(),
              nn.Dropout(dropout)
          )
          for _ in range(postpool)
      ])
    
      # === FINAL DECISION FUSION ===
      # Multi-head attention over the three pairwise flows
      self.decision_attention = nn.MultiheadAttention(
          embed_dim=mlp_out4,
          num_heads=8,
          dropout=dropout,
          batch_first=True
      )
    
      # Final projection layers
      self.final_projection = nn.Sequential(
          nn.Linear(mlp_out4, mlp_out4),
          nn.LayerNorm(mlp_out4),
          nn.ReLU(),
          nn.Dropout(dropout)
      )
    
      # Output layers
      self.linear1 = nn.Linear(mlp_out1 + mlp_out2 + mlp_out3 + mlp_out4, mlp_out4)
      self.relu = nn.ReLU()
      self.linear2 = nn.Linear(mlp_out4, 8)   # Classification
      self.linear3 = nn.Linear(mlp_out4, 3)   # Regression
    
      # Buffers for analysis
      self.register_buffer("last_attention_weights", torch.zeros(1, 3, 3))
      self.register_buffer("last_flow_weights", torch.ones(3) / 3)
    
 def forward(self, x1, x2, x3, labels=None, dev=False):
      """
      Forward pass with pairwise VarDepth fusion flows
    
      Args:
          x1: Language modality input (h)
          x2: Audio modality input (g)
          x3: Vision modality input (z)
          labels: Labels for mixup
          dev: Development/evaluation mode flag
      """
      # Apply M³ masking
      x1, x2, x3 = self.m3(x1, x2, x3)
    
      # Unimodal encodings: get all intermediate representations
      h = self.leg1(x1)  # List of representations [h0, h1, ..., h_{prepool+postpool}]
      g = self.leg2(x2)  # List of representations [g0, g1, ..., g_{prepool+postpool}]
      z = self.leg3(x3)  # List of representations [z0, z1, ..., z_{prepool+postpool}]
    
      # === PAIRWISE FLOW 1: h–g Processing ===
      hg_fusion_repr = None
    
      # Prepool fusion for h-g flow
      for i in range(self.prepool):
          if i == 0:
              hg_concat = torch.cat((h[i], g[i]), dim=-1)
          else:
              hg_concat = torch.cat((h[i], g[i], hg_fusion_repr), dim=-1)
        
          # Pass through VarDepthEnc fusion layer
          hg_fusion_outputs = self.hg_prepool_fusion[i](hg_concat)
        
          if i == self.prepool - 1:
              hg_fusion_repr = hg_fusion_outputs[-1]  # Keep the last (pooled) representation
          else:
              hg_fusion_repr = hg_fusion_outputs[0]   # Keep the first unpooled representation
    
      # Postpool fusion for h-g flow
      for i in range(self.postpool):
          hg_concat = torch.cat((h[self.prepool + i], g[self.prepool + i], hg_fusion_repr), dim=1)
          hg_fusion_repr = self.hg_postpool_fusion[i](hg_concat)
    
      # === PAIRWISE FLOW 2: h–z Processing ===
      hz_fusion_repr = None
    
      # Prepool fusion for h-z flow
      for i in range(self.prepool):
          if i == 0:
              hz_concat = torch.cat((h[i], z[i]), dim=-1)
          else:
              hz_concat = torch.cat((h[i], z[i], hz_fusion_repr), dim=-1)
        
          hz_fusion_outputs = self.hz_prepool_fusion[i](hz_concat)
        
          if i == self.prepool - 1:
              hz_fusion_repr = hz_fusion_outputs[-1]
          else:
              hz_fusion_repr = hz_fusion_outputs[0]
    
      # Postpool fusion for h-z flow
      for i in range(self.postpool):
          hz_concat = torch.cat((h[self.prepool + i], z[self.prepool + i], hz_fusion_repr), dim=1)
          hz_fusion_repr = self.hz_postpool_fusion[i](hz_concat)
    
      # === PAIRWISE FLOW 3: g–z Processing ===
      gz_fusion_repr = None
    
      # Prepool fusion for g-z flow
      for i in range(self.prepool):
          if i == 0:
              gz_concat = torch.cat((g[i], z[i]), dim=-1)
          else:
              gz_concat = torch.cat((g[i], z[i], gz_fusion_repr), dim=-1)
        
          gz_fusion_outputs = self.gz_prepool_fusion[i](gz_concat)
        
          if i == self.prepool - 1:
              gz_fusion_repr = gz_fusion_outputs[-1]
          else:
              gz_fusion_repr = gz_fusion_outputs[0]
    
      # Postpool fusion for g-z flow
      for i in range(self.postpool):
          gz_concat = torch.cat((g[self.prepool + i], z[self.prepool + i], gz_fusion_repr), dim=1)
          gz_fusion_repr = self.gz_postpool_fusion[i](gz_concat)
    
      # === FINAL DECISION FUSION ===
      # Stack the three pairwise fusion results
      pairwise_stack = torch.stack([hg_fusion_repr, hz_fusion_repr, gz_fusion_repr], dim=1)
      # Shape: (batch_size, 3, mlp_out4)
    
      # Apply multi-head attention over the three flows
      attn_out, attn_weights = self.decision_attention(
          query=pairwise_stack,
          key=pairwise_stack,
          value=pairwise_stack
      )
    
      # Pool across the 3 flow tokens
      final_fusion = attn_out.mean(dim=1)  # (batch_size, mlp_out4)
    
      # Apply final projection
      final_fusion = self.final_projection(final_fusion)
    
      # Concatenate with final unimodal representations (following original DeepSER pattern)
      x = torch.cat((h[-1], g[-1], z[-1], final_fusion), dim=-1)
    
      # Final linear transformation
      x = self.linear1(x)
    
      # Store attention weights
      self.last_attention_weights = attn_weights.detach()
    
      # Mixup augmentation
      p = torch.rand(size=(1,), device=x.device)
      if p < self.mixup and labels is not None and not dev:
          x, labels = mixup_features(x, labels)
    
      # Final activation and outputs
      x = self.relu(x)
    
      # Classification and regression outputs
      yc = self.linear2(x)  # Classification logits
      yr = 1 + 6 * torch.sigmoid(self.linear3(x))  # Regression outputs
    
      if self.mixup > 0.0 and labels is not None and not dev:
          return yc, yr, labels
      else:
          return yc, yr






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
       dropout = 0.4
       mixup = hyp_params.mixup if hasattr(hyp_params, 'mixup') else 0.2
       m3_p = hyp_params.m3_p if hasattr(hyp_params, 'm3_p') else 0.4
      
       # Uncomment the model you want to use (Deepser base or pairwise)
       # If you want to use Vardepth model check the parameters and initialize it like the others
       """
       self.model = DeepSERBase(
            embed_dim1=self.orig_d_l,    # Language/text modality input dimension
            mlp_out1=hidden_dim,         # Output dimension for first leg
            embed_dim2=self.orig_d_a,    # Audio modality input dimension  
            mlp_out2=hidden_dim,         # Output dimension for second leg
            embed_dim3=self.orig_d_v,    # Visual modality input dimension
            mlp_out3=hidden_dim,         # Output dimension for third leg
            mlp_out4=hidden_dim,         # Final output dimension for legs 4&5
            encoder="paper",              # Use original encoder implementation
            dropout=dropout,             # Dropout rate
            mixup=mixup                  # Mixup probability
        )
        """


        self.model = PairwiseVarDepthDeepSER(
          embed_dim1=self.orig_d_l,    # Language/text modality input dimension
          mlp_out1=hidden_dim,         # Output dimension for first modality
          embed_dim2=self.orig_d_a,    # Audio modality input dimension
          mlp_out2=hidden_dim,         # Output dimension for second modality
          embed_dim3=self.orig_d_v,    # Visual modality input dimension
          mlp_out3=hidden_dim,         # Output dimension for third modality
          mlp_out4=hidden_dim,         # Final output dimension
          prepool=4,             # Layers before pooling
          postpool=4,           # Layers after pooling
          dropout=dropout,             # Dropout rate
          mixup=mixup,                 # Mixup probability
          m3_p=m3_p                    # M3 masking probability
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
