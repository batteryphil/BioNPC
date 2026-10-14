#!/usr/bin/env python3
"""
BioNPC Weight & Archetype Exporter
Exports PyTorch EmbodiedNPCModel weights or pre-trained Biological Archetypes to:
1. Binary format (.bionpc) for dynamic runtime loading via bionpc_load_weights()
2. C Header format (.h) for zero-latency static embedding in game executables
3. Full archetype bundle (.bionpc + .ini designer tuning profiles)
"""

import sys
import struct
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MAGIC = b"BIONPC01"


def export_binary(model, filepath):
    """Exports model weights to native .bionpc binary format."""
    model.save_weights(filepath)
    print(f"Successfully exported weights to binary: {filepath}")


def export_cheader(model, filepath, var_name="g_bionpc_weights"):
    """Exports model weights to a standalone C header file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated BioNPC model weights */\n")
        f.write(f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n")
        f.write('#include "bionpc.h"\n\n')

        d_in = model.proj_in.in_features
        d_out = model.proj_out.out_features
        d_model = model.proj_in.out_features
        n_layers = len(model.apl)

        f.write(f"static const bionpc_config_t {var_name}_config = {{\n")
        f.write(f"    .d_in = {d_in},\n    .d_out = {d_out},\n    .d_model = {d_model},\n    .n_layers = {n_layers}\n")
        f.write("};\n\n")

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
    parser = argparse.ArgumentParser(description="Export BioNPC PyTorch weights and archetypes to native C formats")
    parser.add_argument("--format", choices=["bin", "header"], default="bin", help="Output format")
    parser.add_argument("--out", type=str, default="", help="Output destination path")
    parser.add_argument("--archetype", type=str, default="", help="Optional archetype name to train & export")
    parser.add_argument("--build-all-archetypes", type=str, default="", help="Directory to export all 5 pre-trained archetypes (.bionpc + .ini)")
    args = parser.parse_args()

    from bionpc import EmbodiedNPCModel, build_all_archetypes, train_archetype

    if args.build_all_archetypes:
        exported = build_all_archetypes(args.build_all_archetypes)
        for name, (bin_p, ini_p) in exported.items():
            print(f"  [Archetype: {name:18s}] -> {bin_p} & {ini_p}")
        sys.exit(0)

    if not args.out:
        parser.error("--out is required unless --build-all-archetypes is specified")

    if args.archetype:
        brain, _ = train_archetype(args.archetype)
    else:
        brain = EmbodiedNPCModel(d_in=4, d_out=3, d_model=32, n_layers=2)

    if args.format == "bin":
        export_binary(brain, args.out)
    else:
        export_cheader(brain, args.out)
