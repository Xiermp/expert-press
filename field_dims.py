#!/usr/bin/env python3
"""Print the compression dims note for a field artifact ("how much onto how much").

Reads config.json (field geometry), field_meta.json (measured accounting) and
the safetensors files of the artifact; prints the dimension story:
d_model x d_ff x N experts  ->  rank r, per-block and total params/bytes.

Usage:
  python3 field_dims.py --artifact results/field_xxx_r128
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True)
    a = ap.parse_args()

    adir = a.artifact.rstrip("/")
    with open(os.path.join(adir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    fi = cfg.get("field")
    if not fi:
        sys.exit("not a field artifact (no 'field' in config.json)")
    d, dff, N, r = fi["d_model"], fi["d_ff"], fi["n_exp"], fi["rank"]
    L = fi["n_layers"]

    # params per MoE block
    exp_p = N * 3 * dff * d                       # gate+up (2dff*d) + down (d*dff)
    cen_p = 3 * dff * d                           # centroids w1d+w2d
    uv_p = r * (3 * dff + 2 * d)                  # Ugu,Vgu,Udn,Vdn
    c_p = 2 * N * r                               # coordinate tables
    field_p = cen_p + uv_p + c_p

    meta_path = os.path.join(adir, "field_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    disk = sum(os.path.getsize(os.path.join(adir, f)) for f in os.listdir(adir)
               if f.endswith(".safetensors"))

    print(f"artifact : {adir}")
    print(f"dims     : d_model={d}, d_ff={dff}, experts N={N} (top-{fi.get('top_k', '?')}) "
          f"-> field rank r={r}, layers={L}")
    print(f"1 block  : experts {exp_p / 1e6:.2f}M params -> field {field_p / 1e6:.3f}M "
          f"params ({exp_p / field_p:.1f}x)")
    print(f"           experts {exp_p * 2 / 1e6:.1f} MB -> field {field_p * 2 / 1e6:.1f} MB (fp16)")
    print(f"field mix: centroids {100 * cen_p / field_p:.1f}% | "
          f"U,V factors {100 * uv_p / field_p:.1f}% | coords C {100 * c_p / field_p:.1f}%")
    if meta:
        fe, fl = meta.get("full_experts_mb"), meta.get("field_mb")
        if fe and fl:
            print(f"all {L} layers: experts {fe:.0f} MB -> field {fl:.0f} MB "
                  f"(x{fe / fl:.1f}, fp16 accounting)")
    print(f"on disk  : {disk / 1e9:.2f} GB safetensors (backbone + field)")

    line = (f"dims: d_model={d} x d_ff={dff}, {N} experts -> rank r={r} "
            f"(per block {exp_p / 1e6:.1f}M -> {field_p / 1e6:.2f}M params, "
            f"x{exp_p / field_p:.1f})")
    if meta.get("full_experts_mb") and meta.get("field_mb"):
        line += (f"; experts {meta['full_experts_mb']:.0f} MB -> field "
                 f"{meta['field_mb']:.0f} MB (x{meta['full_experts_mb'] / meta['field_mb']:.1f})")
    print("\nreport line:")
    print(f"  {line}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        pass
