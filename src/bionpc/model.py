import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedAPLNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.tensor(2.0))
        self.gain = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        B, T, D = x.shape
        decay = torch.sigmoid(self.decay_logit)
        gain = F.relu(self.gain)
        activity = x.abs().mean(dim=-1)
        powers = torch.arange(T, device=x.device, dtype=x.dtype)
        h = (decay ** powers).flip(0).view(1, 1, T)
        activity_pad = F.pad(activity.unsqueeze(1), (T-1, 0))
        state = F.conv1d(activity_pad, h).squeeze(1) * gain
        return x * (1.0 - state.unsqueeze(-1))

class EmbodiedNPCModel(nn.Module):
    """
    Biological Foundation Model for NPCs.
    Uses Continuous Time Recurrent Neural Networks (CTRNN) inspired by the Drosophila Connectome.
    """
    def __init__(self, d_in, d_out, d_model=16, n_layers=1):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.apl = nn.ModuleList([FusedAPLNorm(d_model) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model * 4), 
            nn.SiLU(), 
            nn.Linear(d_model * 4, d_model)
        ) for _ in range(n_layers)])
        self.proj_out = nn.Linear(d_model, d_out)
        
    def forward(self, x):
        x = self.proj_in(x)
        for apl, mlp in zip(self.apl, self.mlp):
            x = x + mlp(apl(x))
        return self.proj_out(x)
