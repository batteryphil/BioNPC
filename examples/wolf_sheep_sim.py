import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
from bionpc import EmbodiedNPCModel, FusedAPLNorm


def main():
    # --- 1. SYNTHESIZE BEHAVIORAL DATASETS ---
    # Wolf Inputs: [See_Sheep, See_Shepherd, Hunger, Aggression]
    # Wolf Outputs: [Chase_Sheep, Attack_Shepherd, Flee]
    X_wolf = torch.rand(1, 2000, 4)
    y_wolf = torch.zeros(1, 2000, 3)
    for t in range(2000):
        sheep, shep, hunger, aggro = X_wolf[0, t]
        if shep > 0.4 and aggro < 0.8:
            y_wolf[0, t, 2] = 1.0  # Flee
        elif shep > 0.4 and aggro >= 0.8:
            y_wolf[0, t, 1] = 1.0  # Attack Shepherd
        elif sheep > 0.2 and hunger > 0.3:
            y_wolf[0, t, 0] = 1.0  # Chase Sheep

    # Sheep Inputs: [See_Wolf, Hunger]
    # Sheep Outputs: [Eat_Grass, Flee]
    X_sheep = torch.rand(1, 2000, 2)
    y_sheep = torch.zeros(1, 2000, 2)
    for t in range(2000):
        wolf, hunger = X_sheep[0, t]
        if wolf > 0.2:
            y_sheep[0, t, 1] = 1.0  # Flee
        else:
            y_sheep[0, t, 0] = 1.0  # Eat

    # --- 2. RAPID BEHAVIORAL CLONING ---
    print("Training Wolf Brain...")
    wolf_brain = EmbodiedNPCModel(d_in=4, d_out=3)
    opt_w = torch.optim.Adam(wolf_brain.parameters(), lr=0.05)
    for _ in range(50):
        loss = nn.BCEWithLogitsLoss()(wolf_brain(X_wolf), y_wolf)
        opt_w.zero_grad()
        loss.backward()
        opt_w.step()

    print("Training Sheep Brain...")
    sheep_brain = EmbodiedNPCModel(d_in=2, d_out=2)
    opt_s = torch.optim.Adam(sheep_brain.parameters(), lr=0.05)
    for _ in range(50):
        loss = nn.BCEWithLogitsLoss()(sheep_brain(X_sheep), y_sheep)
        opt_s.zero_grad()
        loss.backward()
        opt_s.step()

    # --- 3. THE LIVE SIMULATION ---
    print("\n=== STARTING LIVE ECOSYSTEM SIMULATION ===")
    with torch.no_grad():
        wolf_mem = torch.zeros(1, 1, 4)
        sheep_mem = torch.zeros(1, 1, 2)

        shepherd_alive = True
        sheep_alive = True
        aggro = 0.5

        for t in range(1, 10):
            print(f"\n--- TIMESTEP {t} ---")

            if t == 1:
                print("Event: The Wolf spots the flock. The Shepherd is guarding them.")
                aggro = 0.5
            elif t == 4:
                print("Event: The Wolf grows starving and ENRAGED! (Aggression spikes)")
                aggro = 0.95

            see_sheep = 1.0 if sheep_alive else 0.0
            see_shep = 1.0 if shepherd_alive else 0.0
            hunger = min(1.0, 0.5 + t * 0.1)
            wolf_input = torch.tensor([[[see_sheep, see_shep, hunger, aggro]]])

            wolf_mem = torch.cat([wolf_mem, wolf_input], dim=1)
            wolf_logits = wolf_brain(wolf_mem)
            w_probs = torch.sigmoid(wolf_logits[0, -1])

            chase, attack, flee = w_probs[0].item(), w_probs[1].item(), w_probs[2].item()

            if flee > 0.5 and flee > attack and flee > chase:
                print("[WOLF] Sees the Shepherd. Fear overrides hunger! (Flee=100%)")
                see_wolf_for_sheep = 0.0
            elif attack > 0.5 and attack > chase:
                print("[WOLF] Blind rage! The Wolf charges the Shepherd! (Attack=100%)")
                print("       >> The Shepherd is eliminated! <<")
                shepherd_alive = False
                see_wolf_for_sheep = 1.0
            elif chase > 0.5:
                print("[WOLF] No Shepherd in sight. The Wolf chases the Sheep! (Chase=100%)")
                print("       >> A Sheep is eaten! <<")
                sheep_alive = False
                see_wolf_for_sheep = 1.0
            else:
                print("[WOLF] Prowling in the shadows...")
                see_wolf_for_sheep = 0.5

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


if __name__ == "__main__":
    main()
