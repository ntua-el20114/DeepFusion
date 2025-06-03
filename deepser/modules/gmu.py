import torch
from torch import nn

class TwoWayGMU(nn.Module):
    """Gated Multimodal Unit for two modalities"""
    def __init__(self, dim_x1, dim_x2, output_dim, dropout=0.3):
        super(TwoWayGMU, self).__init__()
        # Linear transforms for each modality
        self.transform_x1 = nn.Linear(dim_x1, output_dim)
        self.transform_x2 = nn.Linear(dim_x2, output_dim)

        # Gate network: outputs 2 logits, one per modality
        self.gate = nn.Linear(dim_x1 + dim_x2, 2)
        # Initialize bias so neither modality is favored initially
        self.gate.bias.data = torch.tensor([1.0, 1.0])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x1, x2):
        """
        Args:
            x1: Tensor of shape [..., dim_x1]
            x2: Tensor of shape [..., dim_x2]
        Returns:
            h_fusion: fused representation of shape [..., output_dim]
            gates:      gate weights of shape [..., 2] (sums to 1 along last dim)
        """
        # 1) Transform each modality into the fused space
        h1 = torch.tanh(self.transform_x1(x1))  # [..., output_dim]
        h2 = torch.tanh(self.transform_x2(x2))  # [..., output_dim]

        # 2) Apply dropout to each transformed vector
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)

        # 3) Compute gate weights
        combined = torch.cat([x1, x2], dim=-1)     # [..., dim_x1 + dim_x2]
        gate_logits = self.gate(combined)          # [..., 2]
        gates = torch.softmax(gate_logits, dim=-1) # [..., 2]
        g1, g2 = gates.chunk(2, dim=-1)            # Each [..., 1]

        # 4) Fuse the two modality vectors via gated sum
        h_fusion = self.dropout(g1 * h1 + g2 * h2) # [..., output_dim]

        return h_fusion, gates



class ThreeWayGMU(nn.Module):
   """Gated Multimodal Unit for three modalities"""
   def __init__(self, dim_h, dim_g, dim_z, output_dim, dropout=0.3):
       super(ThreeWayGMU, self).__init__()
       self.transform_h = nn.Linear(dim_h, output_dim)
       self.transform_g = nn.Linear(dim_g, output_dim)
       self.transform_z = nn.Linear(dim_z, output_dim)
      
       # Gate network
       self.gate = nn.Linear(dim_h + dim_g + dim_z, 3)
       self.gate.bias.data = torch.tensor([1.0, 1.0, 1.0])
       self.dropout = nn.Dropout(p=dropout)
  
   def forward(self, x_h, x_g, x_z):
       # Transform each modality
       h_h = torch.tanh(self.transform_h(x_h))
       h_g = torch.tanh(self.transform_g(x_g))
       h_z = torch.tanh(self.transform_z(x_z))
      
       # Apply dropout
       h_h = self.dropout(h_h)
       h_g = self.dropout(h_g)
       h_z = self.dropout(h_z)
      
       # Compute gates
       combined = torch.cat([x_h, x_g, x_z], dim=-1)
       gates = torch.softmax(self.gate(combined), dim=-1)
       g_h, g_g, g_z = gates.chunk(3, dim=-1)
      
       # Apply gating and fusion
       h_fusion = self.dropout(g_h * h_h + g_g * h_g + g_z * h_z)
      
       return h_fusion, gates




class FourWayGMU(nn.Module):
   """Gated Multimodal Unit for four inputs (three modalities + previous fusion)"""
   def __init__(self, dim_h, dim_g, dim_z, dim_f, output_dim, dropout=0.3):
       super(FourWayGMU, self).__init__()
       self.transform_h = nn.Linear(dim_h, output_dim)
       self.transform_g = nn.Linear(dim_g, output_dim)
       self.transform_z = nn.Linear(dim_z, output_dim)
       self.transform_f = nn.Linear(dim_f, output_dim)
      
       # Gate network
       self.gate = nn.Linear(dim_h + dim_g + dim_z + dim_f, 4)
       self.gate.bias.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
       self.dropout = nn.Dropout(p=dropout)
  
   def forward(self, x_h, x_g, x_z, x_f):
       # Transform each input
       h_h = torch.tanh(self.transform_h(x_h))
       h_g = torch.tanh(self.transform_g(x_g))
       h_z = torch.tanh(self.transform_z(x_z))
       h_f = torch.tanh(self.transform_f(x_f))
      
       # Apply dropout
       h_h = self.dropout(h_h)
       h_g = self.dropout(h_g)
       h_z = self.dropout(h_z)
       h_f = self.dropout(h_f)
      
       # Compute gates
       combined = torch.cat([x_h, x_g, x_z, x_f], dim=-1)
       gates = torch.softmax(self.gate(combined), dim=-1)
       g_h, g_g, g_z, g_f = gates.chunk(4, dim=-1)
      
       # Apply gating and fusion
       h_fusion = self.dropout(g_h * h_h + g_g * h_g + g_z * h_z + g_f * h_f)
      
       return h_fusion, gates
