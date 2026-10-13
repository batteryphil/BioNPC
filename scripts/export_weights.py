#!/usr/bin/env python3
"""
BioNPC Weight Exporter
Exports PyTorch EmbodiedNPCModel weights to:
1. Binary format (.bionpc) for dynamic runtime loading via bionpc_load_weights()
2. C Header format (.h) for zero-latency static embedding in game executables
"""

import sys
import struct
import argparse
from pathlib import Path

# Add src to path so bionpc model can be imported if available
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MAGIC = b"BIONPC01"

def export_binary(model, filepath):
    """Exports model weights to native .bionpc binary format."""
    cfg = [model.proj_in.in_features, model.proj_out.out_features, model.proj_in.out_features, len(model.apl)]
    
    with open(filepath, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4i", *cfg))
        
        # proj_in
        f.write(model.proj_in.weight.detach().cpu().numpy().astype("float32").tobytes())
        f.write(model.proj_in.bias.detach().cpu().numpy().astype("float32").tobytes())
        
        # layers
        for apl, mlp in zip(model.apl, model.mlp):
            f.write(apl.decay_logit.detach().cpu().numpy().astype("float32").tobytes())
            f.write(apl.gain.detach().cpu().numpy().astype("float32").tobytes())
            f.write(mlp[0].weight.detach().cpu().numpy().astype("float32").tobytes())
            f.write(mlp[0].bias.detach().cpu().numpy().astype("float32").tobytes())
            f.write(mlp[2].weight.detach().cpu().numpy().astype("float32").tobytes())
            f.write(mlp[2].bias.detach().cpu().numpy().astype("float32").tobytes())
            
        # proj_out
        f.write(model.proj_out.weight.detach().cpu().numpy().astype("float32").tobytes())
        f.write(model.proj_out.bias.detach().cpu().numpy().astype("float32").tobytes())
        
    print(f"Successfully exported weights to binary: {filepath}")

def export_cheader(model, filepath, var_name="g_bionpc_weights"):
    """Exports model weights to a standalone C header file."""
    # Write a clean C header with static const float arrays
    with open(filepath, "w") as f:
        f.write(f"/* Auto-generated BioNPC model weights */\n")
        f.write(f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n")
        f.write('#include "bionpc.h"\n\n')
        
        d_in = model.proj_in.in_features
        d_out = model.proj_out.out_features
        d_model = model.proj_in.out_features
        n_layers = len(model.apl)
        
        f.write(f"static const bionpc_config_t {var_name}_config = {{\n")
        f.write(f"    .d_in = {d_in},\n    .d_out = {d_out},\n    .d_model = {d_model},\n    .n_layers = {n_layers}\n")
        f.write(f"}};\n\n")
        
        # Flattened weights
        def fmt_floats(arr):
            return ", ".join(f"{v:.7e}f" for v in arr.flatten())
            
        f.write(f"static const float {var_name}_w_in[] = {{ {fmt_floats(model.proj_in.weight.detach().cpu().numpy())} }};\n")
        f.write(f"static const float {var_name}_b_in[] = {{ {fmt_floats(model.proj_in.bias.detach().cpu().numpy())} }};\n")
        
        for i, (apl, mlp) in enumerate(zip(model.apl, model.mlp)):
            f.write(f"static const float {var_name}_l{i}_decay_logit = {apl.decay_logit.item():.7e}f;\n")
            f.write(f"static const float {var_name}_l{i}_gain = {apl.gain.item():.7e}f;\n")
            f.write(f"static const float {var_name}_l{i}_w_mlp1[] = {{ {fmt_floats(mlp[0].weight.detach().cpu().numpy())} }};\n")
            f.write(f"static const float {var_name}_l{i}_b_mlp1[] = {{ {fmt_floats(mlp[0].bias.detach().cpu().numpy())} }};\n")
            f.write(f"static const float {var_name}_l{i}_w_mlp2[] = {{ {fmt_floats(mlp[2].weight.detach().cpu().numpy())} }};\n")
            f.write(f"static const float {var_name}_l{i}_b_mlp2[] = {{ {fmt_floats(mlp[2].bias.detach().cpu().numpy())} }};\n")
            
        f.write(f"static const float {var_name}_w_out[] = {{ {fmt_floats(model.proj_out.weight.detach().cpu().numpy())} }};\n")
        f.write(f"static const float {var_name}_b_out[] = {{ {fmt_floats(model.proj_out.bias.detach().cpu().numpy())} }};\n\n")
        f.write(f"#endif /* {var_name.upper()}_H */\n")
        
    print(f"Successfully exported C header to: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BioNPC PyTorch weights to native C formats")
    parser.add_argument("--format", choices=["bin", "header"], default="bin", help="Output format")
    parser.add_argument("--out", type=str, required=True, help="Output destination path")
    args = parser.parse_args()
    
    try:
        import torch
        from bionpc import EmbodiedNPCModel
    except ImportError:
        print("Note: PyTorch not installed in this environment. Script ready for environments with PyTorch.")
        sys.exit(0)
        
    brain = EmbodiedNPCModel(d_in=4, d_out=3, d_model=32, n_layers=2)
    if args.format == "bin":
        export_binary(brain, args.out)
    else:
        export_cheader(brain, args.out)
