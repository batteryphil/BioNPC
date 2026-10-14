import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
from bionpc import EmbodiedNPCModel, FusedAPLNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. TRAIN SPECIES BRAINS ---
def train_brain(d_in, d_out, generate_data_fn):
    X, y = generate_data_fn()
    brain = EmbodiedNPCModel(d_in, d_out).to(device)
    opt = torch.optim.Adam(brain.parameters(), lr=0.05)
    for _ in range(30):
        loss = nn.BCEWithLogitsLoss()(brain(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    brain.eval()
    return brain

# Sheep: In=[See_Predator, See_Grass, Hunger], Out=[Flee, Eat]
def sheep_data():
    X = torch.rand(1, 1000, 3).to(device)
    y = torch.zeros(1, 1000, 2).to(device)
    for t in range(1000):
        pred, grass, hunger = X[0, t]
        if pred > 0.2: y[0, t, 0] = 1.0
        elif grass > 0.2 and hunger > 0.3: y[0, t, 1] = 1.0
    return X, y

# Wolf: In=[See_Prey, See_Bear, Hunger], Out=[Chase, Flee]
def wolf_data():
    X = torch.rand(1, 1000, 3).to(device)
    y = torch.zeros(1, 1000, 2).to(device)
    for t in range(1000):
        prey, bear, hunger = X[0, t]
        if bear > 0.3: y[0, t, 1] = 1.0
        elif prey > 0.1 and hunger > 0.4: y[0, t, 0] = 1.0
    return X, y

# Bear: In=[See_Prey, Hunger], Out=[Chase]
def bear_data():
    X = torch.rand(1, 1000, 2).to(device)
    y = torch.zeros(1, 1000, 1).to(device)
    for t in range(1000):
        prey, hunger = X[0, t]
        if prey > 0.1 and hunger > 0.2: y[0, t, 0] = 1.0
    return X, y

print("Training biological brains...")
brains = {
    'sheep': train_brain(3, 2, sheep_data),
    'wolf': train_brain(3, 2, wolf_data),
    'bear': train_brain(2, 1, bear_data)
}

# --- 3. SIMULATION SETUP ---
NUM_SHEEP = 40
NUM_WOLVES = 8
NUM_BEARS = 2
WORLD_SIZE = 100.0

agents = []
for i in range(NUM_SHEEP):
    agents.append({'id': i, 'type': 'sheep', 'x': random.uniform(0, WORLD_SIZE), 'y': random.uniform(0, WORLD_SIZE), 'hunger': random.uniform(0, 0.5), 'alive': True})
for i in range(NUM_WOLVES):
    agents.append({'id': NUM_SHEEP+i, 'type': 'wolf', 'x': random.uniform(0, WORLD_SIZE), 'y': random.uniform(0, WORLD_SIZE), 'hunger': random.uniform(0, 0.5), 'alive': True})
for i in range(NUM_BEARS):
    agents.append({'id': NUM_SHEEP+NUM_WOLVES+i, 'type': 'bear', 'x': random.uniform(0, WORLD_SIZE), 'y': random.uniform(0, WORLD_SIZE), 'hunger': random.uniform(0, 0.5), 'alive': True})

grass_patches = [{'x': random.uniform(10, 90), 'y': random.uniform(10, 90), 'food': 1.0} for _ in range(20)]

history = []
events = []

def get_closest(x, y, target_types):
    min_d = 999.0
    target_dx, target_dy = 0, 0
    target_agent = None
    for a in agents:
        if a['alive'] and a['type'] in target_types and not (a['x']==x and a['y']==y):
            d = np.sqrt((a['x']-x)**2 + (a['y']-y)**2)
            if d < min_d:
                min_d = d
                target_dx, target_dy = a['x']-x, a['y']-y
                target_agent = a
    if min_d == 0: return 0, 0, 0, None
    return max(0, 1.0 - min_d/30.0), target_dx/(min_d+1e-5), target_dy/(min_d+1e-5), target_agent

# Initialize memory arrays
memory = {
    'sheep': torch.zeros(NUM_SHEEP, 1, 3).to(device),
    'wolf': torch.zeros(NUM_WOLVES, 1, 3).to(device),
    'bear': torch.zeros(NUM_BEARS, 1, 2).to(device)
}

print("Running Ecosystem Simulation for 600 frames...")
for frame in range(600):
    # Prepare inputs
    sheep_idx, wolf_idx, bear_idx = 0, 0, 0
    inputs_sheep = torch.zeros(NUM_SHEEP, 1, 3).to(device)
    inputs_wolf = torch.zeros(NUM_WOLVES, 1, 3).to(device)
    inputs_bear = torch.zeros(NUM_BEARS, 1, 2).to(device)
    
    vec_sheep = []
    vec_wolf = []
    vec_bear = []
    
    for a in agents:
        if not a['alive']:
            if a['type'] == 'sheep': sheep_idx += 1; vec_sheep.append((0,0,0,0,None,None))
            elif a['type'] == 'wolf': wolf_idx += 1; vec_wolf.append((0,0,0,0,None,None))
            elif a['type'] == 'bear': bear_idx += 1; vec_bear.append((0,0,None))
            continue
            
        a['hunger'] += 0.002 # Hunger builds up!
        if a['hunger'] >= 1.0:
            a['alive'] = False
            events.append(f"Frame {frame}: A {a['type']} starved to death.")
            
        if a['type'] == 'sheep':
            pred_signal, pdx, pdy, _ = get_closest(a['x'], a['y'], ['wolf', 'bear'])
            
            # Find closest grass
            min_gd = 999.0; gdx, gdy = 0, 0; closest_g = None
            for g in grass_patches:
                if g['food'] > 0:
                    d = np.sqrt((g['x']-a['x'])**2 + (g['y']-a['y'])**2)
                    if d < min_gd: min_gd = d; gdx, gdy = g['x']-a['x'], g['y']-a['y']; closest_g = g
            grass_signal = max(0, 1.0 - min_gd/20.0) if min_gd != 999.0 else 0.0
            
            inputs_sheep[sheep_idx, 0] = torch.tensor([pred_signal, grass_signal, a['hunger']])
            vec_sheep.append((pdx, pdy, gdx, gdy, closest_g, min_gd))
            sheep_idx += 1
            
        elif a['type'] == 'wolf':
            prey_signal, pdx, pdy, prey_agent = get_closest(a['x'], a['y'], ['sheep'])
            bear_signal, bdx, bdy, _ = get_closest(a['x'], a['y'], ['bear'])
            inputs_wolf[wolf_idx, 0] = torch.tensor([prey_signal, bear_signal, a['hunger']])
            vec_wolf.append((pdx, pdy, bdx, bdy, prey_agent, prey_signal))
            wolf_idx += 1
            
        elif a['type'] == 'bear':
            prey_signal, pdx, pdy, prey_agent = get_closest(a['x'], a['y'], ['wolf', 'sheep'])
            inputs_bear[bear_idx, 0] = torch.tensor([prey_signal, a['hunger']])
            vec_bear.append((pdx, pdy, prey_agent))
            bear_idx += 1

    # Run PyTorch Brains
    with torch.no_grad():
        memory['sheep'] = torch.cat([memory['sheep'], inputs_sheep], dim=1)[:, -10:] # keep last 10 steps to save RAM
        out_sheep = torch.sigmoid(brains['sheep'](memory['sheep'])[:, -1])
        
        memory['wolf'] = torch.cat([memory['wolf'], inputs_wolf], dim=1)[:, -10:]
        out_wolf = torch.sigmoid(brains['wolf'](memory['wolf'])[:, -1])
        
        memory['bear'] = torch.cat([memory['bear'], inputs_bear], dim=1)[:, -10:]
        out_bear = torch.sigmoid(brains['bear'](memory['bear'])[:, -1])

    # Apply Actions
    s_idx, w_idx, b_idx = 0, 0, 0
    for a in agents:
        if not a['alive']:
            if a['type'] == 'sheep': s_idx += 1
            elif a['type'] == 'wolf': w_idx += 1
            elif a['type'] == 'bear': b_idx += 1
            continue
            
        dx, dy = 0.0, 0.0
        speed = 1.0
        
        if a['type'] == 'sheep':
            flee, eat = out_sheep[s_idx].tolist()
            pdx, pdy, gdx, gdy, closest_g, min_gd = vec_sheep[s_idx]
            if flee > 0.5:
                dx, dy = -pdx, -pdy # run away from predator!
                speed = 1.5
                a['hunger'] += 0.002 # fleeing burns energy
            elif eat > 0.5 and closest_g:
                mag = np.sqrt(gdx**2 + gdy**2) + 1e-5
                dx, dy = gdx/mag, gdy/mag
                speed = 0.8
                if min_gd < 2.0: # Eating grass!
                    a['hunger'] = max(0, a['hunger'] - 0.1)
                    closest_g['food'] -= 0.1
            else: # Wander
                dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
                speed = 0.5
            s_idx += 1
            
        elif a['type'] == 'wolf':
            chase, flee = out_wolf[w_idx].tolist()
            pdx, pdy, bdx, bdy, prey_agent, prey_signal = vec_wolf[w_idx]
            if flee > 0.5:
                dx, dy = -bdx, -bdy # run from bear
                speed = 1.8
                a['hunger'] += 0.003 # sprinting
            elif chase > 0.5 and prey_agent:
                mag = np.sqrt(pdx**2 + pdy**2) + 1e-5
                dx, dy = pdx/mag, pdy/mag
                speed = 1.6
                if prey_signal > 0.9: # Caught sheep!
                    prey_agent['alive'] = False
                    a['hunger'] = 0.0
                    events.append(f"Frame {frame}: A Wolf hunted a Sheep!")
            else:
                dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
                speed = 0.7
            w_idx += 1
            
        elif a['type'] == 'bear':
            chase = out_bear[b_idx].tolist()[0]
            pdx, pdy, prey_agent = vec_bear[b_idx]
            if chase > 0.5 and prey_agent:
                mag = np.sqrt(pdx**2 + pdy**2) + 1e-5
                dx, dy = pdx/mag, pdy/mag
                speed = 1.2
                dist = np.sqrt((prey_agent['x']-a['x'])**2 + (prey_agent['y']-a['y'])**2)
                if dist < 2.0: # Caught prey!
                    prey_agent['alive'] = False
                    a['hunger'] = 0.0
                    events.append(f"Frame {frame}: A Bear hunted a {prey_agent['type']}!")
            else:
                dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
                speed = 0.6
            b_idx += 1

        # Move
        mag = np.sqrt(dx**2 + dy**2) + 1e-5
        a['x'] += (dx/mag) * speed
        a['y'] += (dy/mag) * speed
        # Bounds
        a['x'] = np.clip(a['x'], 0, WORLD_SIZE)
        a['y'] = np.clip(a['y'], 0, WORLD_SIZE)

    # Grow grass
    for g in grass_patches:
        g['food'] = min(1.0, g['food'] + 0.005)

    # Save state for animation
    state = []
    for a in agents:
        if a['alive']: state.append((a['x'], a['y'], a['type'], a['hunger']))
    history.append(state)

print(f"Simulation finished! Total events: {len(events)}")
for e in events: print(e)

# --- 4. RENDER ANIMATION ---
print("Rendering MP4 video...")
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_title("Biological Ecosystem Simulation (PyTorch FlyBrains)")
ax.set_facecolor('#222222')

colors = {'sheep': 'white', 'wolf': '#ff5555', 'bear': '#ffaa00'}
sizes = {'sheep': 20, 'wolf': 50, 'bear': 100}

scatter = ax.scatter([], [])

def init():
    scatter.set_offsets(np.empty((0, 2)))
    return scatter,

def update(frame):
    state = history[frame]
    if not state:
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,
        
    x = [s[0] for s in state]
    y = [s[1] for s in state]
    c = [colors[s[2]] for s in state]
    s = [sizes[s[2]] for s in state]
    
    scatter.set_offsets(np.c_[x, y])
    scatter.set_color(c)
    scatter.set_sizes(s)
    
    # Text overlay
    ax.set_title(f"Biological Ecosystem Simulation | Frame {frame}")
    return scatter,

ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, interval=50)
ani.save('ecosystem.mp4', writer='ffmpeg', fps=20)
print("Saved to ecosystem.mp4")
