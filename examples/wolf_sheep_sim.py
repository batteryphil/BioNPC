import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. THE BIOLOGICAL FOUNDATION ARCHITECTURE ---
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
    def __init__(self, d_in, d_out, d_model=16, n_layers=1):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.apl = nn.ModuleList([FusedAPLNorm(d_model) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model)
        ) for _ in range(n_layers)])
        self.proj_out = nn.Linear(d_model, d_out)
        
    def forward(self, x):
        x = self.proj_in(x)
        for apl, mlp in zip(self.apl, self.mlp):
            x = x + mlp(apl(x))
        return self.proj_out(x)

# --- 2. SYNTHESIZE BEHAVIORAL DATASETS ---
# Wolf Inputs: [See_Sheep, See_Shepherd, Hunger, Aggression]
# Wolf Outputs: [Chase_Sheep, Attack_Shepherd, Flee]
X_wolf = torch.rand(1, 2000, 4)
y_wolf = torch.zeros(1, 2000, 3)
for t in range(2000):
    sheep, shep, hunger, aggro = X_wolf[0, t]
    if shep > 0.4 and aggro < 0.8:
        y_wolf[0, t, 2] = 1.0 # Flee
    elif shep > 0.4 and aggro >= 0.8:
        y_wolf[0, t, 1] = 1.0 # Attack Shepherd
    elif sheep > 0.2 and hunger > 0.3:
        y_wolf[0, t, 0] = 1.0 # Chase Sheep

# Sheep Inputs: [See_Wolf, Hunger]
# Sheep Outputs: [Eat_Grass, Flee]
X_sheep = torch.rand(1, 2000, 2)
y_sheep = torch.zeros(1, 2000, 2)
for t in range(2000):
    wolf, hunger = X_sheep[0, t]
    if wolf > 0.2:
        y_sheep[0, t, 1] = 1.0 # Flee
    else:
        y_sheep[0, t, 0] = 1.0 # Eat

# --- 3. RAPID BEHAVIORAL CLONING ---
print("Training Wolf Brain...")
wolf_brain = EmbodiedNPCModel(d_in=4, d_out=3)
opt_w = torch.optim.Adam(wolf_brain.parameters(), lr=0.05)
for _ in range(50):
    loss = nn.BCEWithLogitsLoss()(wolf_brain(X_wolf), y_wolf)
    opt_w.zero_grad(); loss.backward(); opt_w.step()

print("Training Sheep Brain...")
sheep_brain = EmbodiedNPCModel(d_in=2, d_out=2)
opt_s = torch.optim.Adam(sheep_brain.parameters(), lr=0.05)
for _ in range(50):
    loss = nn.BCEWithLogitsLoss()(sheep_brain(X_sheep), y_sheep)
    opt_s.zero_grad(); loss.backward(); opt_s.step()

# --- 4. THE LIVE SIMULATION ---
print("\n=== STARTING LIVE ECOSYSTEM SIMULATION ===")
with torch.no_grad():
    wolf_mem = torch.zeros(1, 1, 4)
    sheep_mem = torch.zeros(1, 1, 2)
    
    # State flags
    shepherd_alive = True
    sheep_alive = True
    
    for t in range(1, 10):
        print(f"\n--- TIMESTEP {t} ---")
        
        # Scenario Dynamics
        if t == 1:
            print("Event: The Wolf spots the flock. The Shepherd is guarding them.")
            aggro = 0.5
        elif t == 4:
            print("Event: The Wolf grows starving and ENRAGED! (Aggression spikes)")
            aggro = 0.95
        
        # Build Wolf Senses
        see_sheep = 1.0 if sheep_alive else 0.0
        see_shep = 1.0 if shepherd_alive else 0.0
        hunger = min(1.0, 0.5 + t * 0.1)
        wolf_input = torch.tensor([[[see_sheep, see_shep, hunger, aggro]]])
        
        # Run Wolf Brain
        wolf_mem = torch.cat([wolf_mem, wolf_input], dim=1)
        wolf_logits = wolf_brain(wolf_mem)
        w_probs = torch.sigmoid(wolf_logits[0, -1])
        
        chase, attack, flee = w_probs[0].item(), w_probs[1].item(), w_probs[2].item()
        
        # Wolf Action
        if flee > 0.5 and flee > attack and flee > chase:
            print(f"[WOLF] Sees the Shepherd. Fear overrides hunger! (Flee=100%)")
            see_wolf_for_sheep = 0.0 # Wolf ran away
        elif attack > 0.5 and attack > chase:
            print(f"[WOLF] Blind rage! The Wolf charges the Shepherd! (Attack=100%)")
            print("       >> The Shepherd is eliminated! <<")
            shepherd_alive = False
            see_wolf_for_sheep = 1.0
        elif chase > 0.5:
            print(f"[WOLF] No Shepherd in sight. The Wolf chases the Sheep! (Chase=100%)")
            print("       >> A Sheep is eaten! <<")
            sheep_alive = False
            see_wolf_for_sheep = 1.0
        else:
            print("[WOLF] Prowling in the shadows...")
            see_wolf_for_sheep = 0.5
            
        # Build Sheep Senses
        if sheep_alive:
            sheep_input = torch.tensor([[[see_wolf_for_sheep, 0.8]]])
            sheep_mem = torch.cat([sheep_mem, sheep_input], dim=1)
            sheep_logits = sheep_brain(sheep_mem)
            s_probs = torch.sigmoid(sheep_logits[0, -1])
            eat, s_flee = s_probs[0].item(), s_probs[1].item()
            
            if s_flee > 0.5:
                print(f"[SHEEP] Panics! The flock scatters! (Flee={s_flee*100:.0f}%)")
            else:
                print(f"[SHEEP] Peacefully grazing on grass. (Eat={eat*100:.0f}%)")

