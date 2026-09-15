#!/usr/bin/env python3
"""Build the DeepSeek V4.1 Flash DSpark drafter sidecar GGUF from the HF
checkpoint's mtp.* namespace.

The main quantizer (deepseek41_quantize.py) deliberately omits mtp.*; this
builds the companion sidecar the engine loads with --mtp-model to enable
block-draft speculative decoding (DS4_SUPPORT_DSPARK). Each of the 3 DSpark
stages is a full V4.1 layer (windowed attention + HC mixing + a 128-expert
MoE), plus a main_proj/main_norm backbone combiner at stage 0 and
norm/markov/confidence heads at the final stage.

The engine additionally requires hc_head_{base,fn,scale} at the final stage,
which the checkpoint does NOT store. They are DERIVED here, exactly: the
reference kernel (inference/kernel.py hc_split_sinkhorn) computes the head's
pre-mix as pre[j] = sigmoid(mixes[j]*scale[0] + base[j]), a per-element
function of ONLY the first N_HC entries of the hc_ffn projection -- fully
independent of post/comb (which is the only part the Sinkhorn iteration
touches). So hc_head is the pre-slice of the final stage's hc_ffn:
  hc_head_fn    = hc_ffn_fn[0:N_HC, :]     (first N_HC output rows)
  hc_head_base  = hc_ffn_base[0:N_HC]
  hc_head_scale = hc_ffn_scale[0:1]
Shapes match the engine validator (ds4.c:5852) exactly. If measured draft
acceptance is ~0, the pre/post/comb ordering in the mix vector is the sole
suspect (mix layout is [pre(hc) | post(hc) | comb(hc*hc)]).

Quant policy mirrors the main model: experts Q4_K (q4) / IQ2+Q2_K (q2),
attention + shared + heads + main_proj Q8_0, norms + router + hc mixers +
markov/confidence F32/F16 as the engine's layout validators require.

Usage:
  uv run gguf-tools/deepseek41_dspark.py --hf <ckpt> --out ds41-dspark.gguf \
      --source-revision <40hex> [--quant q4] [--dry-run]
"""
import argparse
import dataclasses
import json
import os
import re
import sys

from deepseek41_metadata import GGUF_ALIGNMENT
from glm53_quantize import (
    SourceDB, TensorPlan, QTYPE_F32, QTYPE_F16, QTYPE_Q8_0, QTYPE_Q2_K,
    QTYPE_Q4_K, QTYPE_IQ2_XXS, align, kv_string, print_plan, qtype_nbytes,
    tensor_header,
)
from deepseek41_quantize import NativeQuantizer, scale_name
import struct, shutil

# --- DSpark metadata KV keys the engine's model_dspark_summary reads ---
def kv_u32(key, value):
    import struct
    kb = key.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 4) + struct.pack("<I", value & 0xFFFFFFFF)

def kv_u32_array(key, values):
    import struct
    kb = key.encode()
    body = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 9)  # ARRAY
    body += struct.pack("<I", 4)  # elem type U32
    body += struct.pack("<Q", len(values))
    body += b"".join(struct.pack("<I", v & 0xFFFFFFFF) for v in values)
    return body


def build_dspark_plan(db, config, quant="q4"):
    c = config["text_config"] if "text_config" in config else config
    dim = c["hidden_size"]
    heads, hd = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc, vocab = c["hc_mult"], c["vocab_size"]
    experts = c["dspark_n_routed_experts"]
    inter = c.get("dspark_moe_intermediate_size", c["moe_intermediate_size"])
    markov_rank = c["dspark_markov_rank"]
    mix_hc = (hc + 2) * hc

    # stage count = distinct mtp.N present in the checkpoint
    stages = sorted({int(m.group(1)) for m in
                     (re.match(r"mtp\.(\d+)\.", n) for n in db.tensors) if m})
    n_stages = len(stages)
    final = stages[-1]

    plan, consumed = [], set()

    def claim(name, expected, dtype=None):
        info = db.info(name)
        if list(info["shape"]) != list(expected) or (dtype and info["dtype"] != dtype):
            raise ValueError(f"{name}: got {info['dtype']} {info['shape']}, expected {dtype} {expected}")
        consumed.add(name)
        if info["dtype"] in ("I8", "F8_E4M3"):
            consumed.add(scale_name(name))
        return info

    def regular(name, source, shape, qt, role):
        claim(source, shape)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qt, role, source=source))

    for st in stages:
        src, dst = f"mtp.{st}", f"mtp.{st}"
        # HC mixers (identical layout to main layers)
        for site in ("attn", "ffn"):
            regular(f"{dst}.hc_{site}_fn.weight", f"{src}.hc_{site}_fn",
                    (mix_hc, hc * dim), QTYPE_F16, "mhc")
            regular(f"{dst}.hc_{site}_base.weight", f"{src}.hc_{site}_base",
                    (mix_hc,), QTYPE_F32, "mhc")
            regular(f"{dst}.hc_{site}_scale.weight", f"{src}.hc_{site}_scale",
                    (3,), QTYPE_F32, "mhc")
        regular(f"{dst}.attn_norm.weight", f"{src}.attn_norm.weight", (dim,), QTYPE_F32, "norm")
        regular(f"{dst}.ffn_norm.weight", f"{src}.ffn_norm.weight", (dim,), QTYPE_F32, "norm")
        # attention
        for target, source, shape in (
            ("attn_sinks.weight", "attn.attn_sink", (heads,)),
            ("attn_q_a.weight", "attn.wq_a.weight", (qrank, dim)),
            ("attn_q_b.weight", "attn.wq_b.weight", (heads * hd, qrank)),
            ("attn_q_a_norm.weight", "attn.q_norm.weight", (qrank,)),
            ("attn_kv.weight", "attn.wkv.weight", (hd, dim)),
            ("attn_kv_a_norm.weight", "attn.kv_norm.weight", (hd,)),
            ("attn_output_a.weight", "attn.wo_a.weight", (groups * orank, heads * hd // groups)),
            ("attn_output_b.weight", "attn.wo_b.weight", (dim, groups * orank)),
        ):
            qt = QTYPE_F32 if target.endswith(("sinks.weight", "norm.weight")) else QTYPE_Q8_0
            regular(f"{dst}.{target}", f"{src}.{source}", shape, qt, "attention")
        # router
        regular(f"{dst}.ffn_gate_inp.weight", f"{src}.ffn.gate.weight", (experts, dim), QTYPE_F32, "router")
        regular(f"{dst}.exp_probs_b.bias", f"{src}.ffn.gate.bias", (experts,), QTYPE_F32, "router")
        regular(f"{dst}.exp_probs_b_vl.bias", f"{src}.ffn.gate.bias_vl", (experts,), QTYPE_F32, "router")
        # experts + shared
        for part, source, shape in (("gate", "w1", (inter, dim)),
                                    ("up", "w3", (inter, dim)),
                                    ("down", "w2", (dim, inter))):
            regular(f"{dst}.ffn_{part}_shexp.weight",
                    f"{src}.ffn.shared_experts.{source}.weight", shape, QTYPE_Q8_0, "shared")
            pattern = f"{src}.ffn.experts.{{expert}}.{source}.weight"
            for e in range(experts):
                claim(pattern.format(expert=e), (shape[0], shape[1] // 2), "I8")
            plan.append(TensorPlan(f"{dst}.ffn_{part}_exps.weight", (*reversed(shape), experts),
                                   QTYPE_Q4_K if quant == "q4" else
                                   (QTYPE_Q2_K if part == "down" else QTYPE_IQ2_XXS),
                                   "experts", source=pattern, expert_layer=st,
                                   expert_part=part, expert_count=experts))
        if st == stages[0]:
            regular(f"{dst}.main_proj.weight", f"{src}.main_proj.weight",
                    (dim, dim * len(c["dspark_target_layer_ids"])), QTYPE_Q8_0, "dspark")
            regular(f"{dst}.main_norm.weight", f"{src}.main_norm.weight", (dim,), QTYPE_F32, "norm")
        if st == final:
            regular(f"{dst}.norm.weight", f"{src}.norm.weight", (dim,), QTYPE_F32, "norm")
            regular(f"{dst}.markov_head.markov_w1.weight",
                    f"{src}.markov_head.embed.weight", (vocab, markov_rank), QTYPE_F16, "dspark")
            regular(f"{dst}.markov_head.markov_w2.weight",
                    f"{src}.markov_head.head.weight", (vocab, markov_rank), QTYPE_F16, "dspark")
            regular(f"{dst}.confidence_head.proj.weight",
                    f"{src}.confidence_head.proj.weight", (1, dim + markov_rank), QTYPE_F32, "dspark")
            # hc_head: DERIVED pre-slice of this stage's hc_ffn (see module docstring)
            plan.append(TensorPlan(f"{dst}.hc_head_fn.weight", (hc * dim, hc), QTYPE_F16,
                                   "mhc", source=f"{src}.hc_ffn_fn", transform="hc_head_fn"))
            plan.append(TensorPlan(f"{dst}.hc_head_base.weight", (hc,), QTYPE_F32,
                                   "mhc", source=f"{src}.hc_ffn_base", transform="hc_head_base"))
            plan.append(TensorPlan(f"{dst}.hc_head_scale.weight", (1,), QTYPE_F32,
                                   "mhc", source=f"{src}.hc_ffn_scale", transform="hc_head_scale"))

    omitted = {n for n in db.tensors if not n.startswith("mtp.")}
    leftover = set(db.tensors) - consumed - omitted
    # scale sidecars of consumed tensors are already in `consumed`
    leftover = {n for n in leftover if not n.endswith(".scale")}
    if leftover:
        raise ValueError(f"unclaimed mtp source tensors: {sorted(leftover)[:12]}")

    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan, n_stages, c


def dspark_records(c, revision):
    tl = c["dspark_target_layer_ids"]
    recs = [
        kv_string("general.architecture", "deepseek41-dspark"),
        kv_string("general.source.revision", revision),
        kv_u32("deepseek4.dspark_block_size", c["dspark_block_size"]),
        kv_u32("deepseek4.dspark_markov_rank", c["dspark_markov_rank"]),
        kv_u32("deepseek4.dspark_noise_token_id", c["dspark_noise_token_id"]),
        kv_u32_array("deepseek4.dspark_target_layer_ids", tl),
    ]
    return recs



def write_dspark(args, plan, records, db):
    """Compact single-pass writer: no imatrix, no resume (8 GiB). hc_head
    items (transform set) are the pre-slice of the stage's hc_ffn."""
    q = NativeQuantizer(args.quants_library)
    np = q.np
    data_start, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
    if os.path.exists(args.out):
        raise ValueError(f"refusing to overwrite {args.out}")
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out)) or ".").free
    if free < data_start + data_bytes + (8 << 30):
        raise ValueError("insufficient disk space (+8 GiB reserve)")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    partial = args.out + ".partial"
    N_HC = db_hc(db)
    with open(partial, "xb") as fp:
        fp.write(header)
        for index, item in enumerate(plan):
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"offset drift at {item.name}")
            if item.transform:
                src = q.to_f32(db, item.source)              # hc_ffn_{fn|base|scale}
                if item.transform == "hc_head_fn":
                    arr = src[:N_HC, :]
                elif item.transform == "hc_head_base":
                    arr = src[:N_HC]
                else:                                        # hc_head_scale
                    arr = src[:1]
                data = q.encode(np.ascontiguousarray(arr, dtype=np.float32), item.qtype)
            elif item.is_expert:
                parts = []
                for e in range(item.expert_count):
                    v = q.to_f32(db, item.source.format(expert=e))
                    parts.append(q.encode(v, item.qtype))
                data = b"".join(parts)
            else:
                data = q.encode(q.to_f32(db, item.source), item.qtype)
            if len(data) != item.nbytes:
                raise ValueError(f"{item.name}: {len(data)} bytes, expected {item.nbytes}")
            fp.write(data)
            fp.write(bytes(align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            print(f"[{index+1}/{len(plan)}] {item.name}: {item.nbytes/(1<<20):.1f} MiB", flush=True)
    os.rename(partial, args.out)
    print(f"wrote {args.out} ({data_start + data_bytes} bytes)", file=sys.stderr)


def db_hc(db):
    # N_HC from a known hc_ffn_base length (mix_hc = (hc+2)*hc) -> solve hc.
    n = db.info("mtp.0.hc_ffn_base")["shape"][0]
    hc = int(round(((n) ** 0.5)))
    while (hc + 2) * hc != n:
        hc += 1 if (hc + 2) * hc < n else -1
        if hc < 1 or hc > 64:
            raise ValueError(f"cannot solve hc from mix_hc={n}")
    return hc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--quant", choices=("q2", "q4"), default="q4")
    ap.add_argument("--dry-run", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    ap.add_argument("--quants-library", default=os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        ap.error("source revision must be a full commit hash")

    config = json.load(open(os.path.join(args.hf, "config.json")))
    db = SourceDB(args.hf, index_validator=lambda _: None, scale_validator=lambda *_: None)
    try:
        plan, n_stages, c = build_dspark_plan(db, config, args.quant)
        records = dspark_records(c, args.source_revision)
        print(f"DSpark sidecar: {n_stages} stages, {len(plan)} tensors, quant {args.quant}",
              file=sys.stderr)
        if args.dry_run:
            for item in plan:
                d = {k: getattr(item, k) for k in ("name", "shape", "qtype", "role")}
                d["transform"] = item.transform
                print(json.dumps(d, sort_keys=True))
            total = sum(align(qtype_nbytes(i.qtype, i.shape), GGUF_ALIGNMENT) for i in plan)
            print(f"planned data bytes: {total} ({total/2**30:.2f} GiB)", file=sys.stderr)
        else:
            write_dspark(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as e:
        sys.exit(f"deepseek41-dspark: {e}")
