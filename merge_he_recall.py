# merge_he_recall.py
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel
import gc
import math

def is_block_list(obj):
    if not (hasattr(obj, "__len__") and hasattr(obj, "__getitem__")):
        return False
    if len(obj) < 2:
        return False
    first = obj[0]
    if not hasattr(first, "state_dict"):
        return False
    sd = first.state_dict()
    return len(sd.keys()) > 0

def find_transformer_blocks(peft_model):
    candidates = []
    roots = [peft_model]
    if hasattr(peft_model, "model"): roots.append(peft_model.model)
    if hasattr(peft_model, "base_model"):
        roots.append(peft_model.base_model)
        if hasattr(peft_model.base_model, "model"):
            roots.append(peft_model.base_model.model)
    field_names = ["layers", "blocks", "h", "transformer", "decoder", "model"]

    seen = set()
    def collect(obj, depth=0):
        if obj is None or id(obj) in seen or depth > 3: return
        seen.add(id(obj))
        if is_block_list(obj): candidates.append(obj)
        for name in dir(obj):
            if name.startswith("_"): continue
            if name in field_names:
                child = getattr(obj, name, None)
                if child is not None:
                    if is_block_list(child): candidates.append(child)
                    collect(child, depth + 1)

    for r in roots: collect(r, 0)
    if not candidates:
        raise RuntimeError("No transformer block list found.")
    best = max(candidates, key=lambda c: len(c))
    print(f"[find_transformer_blocks] picked {len(best)} layers")
    return best

def load_model_with_adapter(base_model_name, adapter_path):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    peft_model.eval()
    return peft_model

# ==========================
# 核心：HE-RECALL 計算
# ==========================

def rbf_similarity(a, b, sigma=1.0):
    diff2 = ((a.mean(dim=0) - b.mean(dim=0)) ** 2).sum()
    return torch.exp(-diff2 / (2 * sigma**2)).item()

def compute_entropy_he_weights(repr_blobs, entropy_blobs, alpha=0.3, sigma=1.0):
    num_models = len(repr_blobs)
    L = repr_blobs[0]["reprs"].shape[1]
    weights_per_layer = []

    # 熵統一長度 & 正規化
    normed_entropies = []
    for blob in entropy_blobs:
        e = blob["entropy"].float()
        e = (e - e.mean()) / (e.std() + 1e-6)
        normed_entropies.append(e)

    for i in range(L):
        sims = []
        anchor = repr_blobs[0]["reprs"][:, i, :]  # anchor = 新任務
        for m_idx in range(num_models):
            cur = repr_blobs[m_idx]["reprs"][:, i, :]
            sim = rbf_similarity(anchor, cur, sigma=sigma)
            sims.append(sim)

        sims_t = torch.tensor(sims)
        # 熵加權：只影響 top 1/3 層
        top_th = int(L * (2 / 3))
        if i >= top_th:
            ent_weight = torch.tensor([e.mean().item() for e in normed_entropies])
            sims_t = sims_t * (1 + alpha * ent_weight)
        w = torch.softmax(sims_t, dim=0)
        weights_per_layer.append(w)
    return weights_per_layer

# ==========================
# 模型融合
# ==========================

def fuse_models(base_model_name, adapter_paths, weights_per_layer, out_dir):
    fused_model = load_model_with_adapter(base_model_name, adapter_paths[0])
    fused_model = fused_model.to("cuda")
    fused_blocks = find_transformer_blocks(fused_model)
    num_layers = len(fused_blocks)
    L_full = len(weights_per_layer)
    offset = L_full - num_layers
    if offset < 0:
        raise RuntimeError(f"repr layers ({L_full}) < transformer blocks ({num_layers})")
    weights_aligned = weights_per_layer[offset:]

    for layer_idx in range(num_layers):
        w_layer = weights_aligned[layer_idx]
        fused_block = fused_blocks[layer_idx]
        fused_sd = {k: v.detach().clone().to("cpu") for k, v in fused_block.state_dict().items()}
        for k in fused_sd.keys():
            fused_sd[k] = fused_sd[k] * w_layer[0].item()
        for m_i in range(1, len(adapter_paths)):
            tmp_model = load_model_with_adapter(base_model_name, adapter_paths[m_i])
            tmp_blocks = find_transformer_blocks(tmp_model)
            tmp_sd = {k: v.detach().clone().to("cpu") for k, v in tmp_blocks[layer_idx].state_dict().items()}
            for k in fused_sd.keys():
                fused_sd[k] += tmp_sd[k] * w_layer[m_i].item()
            del tmp_model, tmp_blocks, tmp_sd
            gc.collect()
            torch.cuda.empty_cache()
        fused_block.load_state_dict(fused_sd)
    fused_model.save_pretrained(out_dir)
    print(f"✅ HE-RECALL fused -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--reprs", nargs="+", required=True)
    ap.add_argument("--entropies", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--out_dir", default="./fused_he_recall")
    args = ap.parse_args()

    repr_blobs = [torch.load(p, map_location="cpu") for p in args.reprs]
    entropy_blobs = [torch.load(p, map_location="cpu") for p in args.entropies]

    weights = compute_entropy_he_weights(repr_blobs, entropy_blobs, alpha=args.alpha)
    fuse_models(args.base_model, args.adapters, weights, args.out_dir)

if __name__ == "__main__":
    main()
