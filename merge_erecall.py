import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel
import gc

########################
# helper: same as merge_recall.py
########################

def is_block_list(obj):
    # 檢查是不是 "一串 transformer blocks"
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
    if hasattr(peft_model, "model"):
        roots.append(peft_model.model)
    if hasattr(peft_model, "base_model"):
        roots.append(peft_model.base_model)
        if hasattr(peft_model.base_model, "model"):
            roots.append(peft_model.base_model.model)

    field_names = [
        "layers",
        "blocks",
        "h",
        "transformer",
        "decoder",
        "model",
    ]

    seen = set()

    def collect(obj, depth=0):
        if obj is None:
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))

        # 直接檢查自己
        if is_block_list(obj):
            candidates.append(obj)

        # 繼續往下挖
        if depth > 3:
            return
        for name in dir(obj):
            if name.startswith("_"):
                continue
            if name in field_names:
                child = getattr(obj, name, None)
                if child is None:
                    continue
                if is_block_list(child):
                    candidates.append(child)
                collect(child, depth + 1)

    for r in roots:
        collect(r, 0)

    if not candidates:
        raise RuntimeError("No transformer block list found in model structure.")
    best = max(candidates, key=lambda c: len(c))
    print(f"[find_transformer_blocks] picked candidate with {len(best)} layers")
    return best  # ModuleList-like of decoder blocks

def load_model_with_adapter(base_model_name, adapter_path):
    """
    載一個 base + adapter 到 GPU (4bit)，回傳 peft model
    """
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"":0},
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    peft_model.eval()
    return peft_model

########################
# eRECALL-specific parts
########################

def compute_entropy_weights_per_layer(repr_blobs, entropy_blob):
    """
    repr_blobs: list of { "task": str, "reprs": Tensor[N, L, H] }
    entropy_blob: Tensor[N_ent]  (from the entropy_file you pass in)
                  這是你的 high-entropy buffer，代表哪一些樣本比較"關鍵"

    我們要做的事：
    - 計算每一層的 "anchor vs each task" 相似度
    - 但相似度是用 entropy 當權重 (高熵樣本影響力比較大)
    - 回傳 weights_per_layer: list[L_full] where each element is tensor[num_models]
    """

    num_models = len(repr_blobs)
    anchor_repr = repr_blobs[0]["reprs"]  # [N0, L, H]
    L_full = anchor_repr.shape[1]

    # 我們把 entropy_blob broadcast 成 anchor用的 sample weight
    # 假設 entropy_blob 對應 anchor 的那些樣本。長度要 >= anchor N0
    N0 = anchor_repr.shape[0]
    entropy_weights = entropy_blob[:N0].float()  # [N0]
    if entropy_weights.sum() == 0:
        entropy_weights = torch.ones_like(entropy_weights)

    weights_per_layer = []

    for layer_idx in range(L_full):
        # anchor 該層: [N0, H]
        a_layer = anchor_repr[:, layer_idx, :]  # [N0, H]

        sims_this_layer = []
        for m_idx in range(num_models):
            b_layer = repr_blobs[m_idx]["reprs"][:, layer_idx, :]  # [Nm, H]

            # 要比較 a_layer vs b_layer。
            # 我們做法：先把 b_layer 平均成一個中心，然後計算 (a_layer[i] vs b_center) 的 cos sim，
            # 再用 entropy_weights 對 a_layer 的 sim 做加權平均。
            b_center = b_layer.mean(dim=0, keepdim=True)  # [1,H]
            # cos sim for each anchor sample:
            # a_layer: [N0,H], b_center: [1,H] -> broadcast to [N0,H]
            sim_vec = F.cosine_similarity(a_layer, b_center, dim=-1)  # [N0]

            # 熵加權平均
            w = entropy_weights.to(sim_vec.device)
            weighted_sim = (sim_vec * w).sum() / w.sum()
            sims_this_layer.append(weighted_sim.item())

        sims_this_layer = torch.tensor(sims_this_layer)  # [num_models]

        # 讓權重非負 + normalize
        sims_clamped = torch.clamp(sims_this_layer, min=0.0)
        if sims_clamped.sum() == 0:
            layer_w = torch.ones_like(sims_clamped) / len(sims_clamped)
        else:
            layer_w = sims_clamped / sims_clamped.sum()

        weights_per_layer.append(layer_w)  # tensor[num_models]

    return weights_per_layer  # list of length L_full

def fuse_models_entropy(base_model_name, adapter_paths, weights_per_layer, out_dir):
    """
    幾乎和 merge_recall 的 fuse_models 一樣，
    但這裡用的是 entropy-aware 的 weights_per_layer。
    """

    # 1. 先載第0個 adapter，這會是 fused_model 起點
    fused_model = load_model_with_adapter(base_model_name, adapter_paths[0])
    fused_model = fused_model.to("cuda")

    fused_blocks = find_transformer_blocks(fused_model)
    num_layers = len(fused_blocks)

    L_full = len(weights_per_layer)
    offset = L_full - num_layers
    if offset < 0:
        raise RuntimeError(
            f"repr layers ({L_full}) < transformer blocks ({num_layers}), can't align."
        )
    weights_aligned = weights_per_layer[offset:]  # list[num_layers], each tensor[num_models]

    # 2. 對每一層 transformer block 做加權平均
    for layer_idx in range(num_layers):
        w_layer = weights_aligned[layer_idx]  # tensor[num_models]

        fused_block = fused_blocks[layer_idx]
        fused_sd = {k: v.detach().clone().to("cpu") for k,v in fused_block.state_dict().items()}

        # 先乘自己第0個模型的權重
        for key in fused_sd.keys():
            fused_sd[key] = fused_sd[key] * w_layer[0].item()

        # 再依序加進其他 adapter
        for m_i in range(1, len(adapter_paths)):
            tmp_model = load_model_with_adapter(base_model_name, adapter_paths[m_i])
            tmp_blocks = find_transformer_blocks(tmp_model)
            tmp_block_sd = {k: v.detach().clone().to("cpu") for k,v in tmp_blocks[layer_idx].state_dict().items()}

            for key in fused_sd.keys():
                fused_sd[key] += tmp_block_sd[key] * w_layer[m_i].item()

            # 清暫存模型，釋放顯存
            del tmp_model, tmp_blocks, tmp_block_sd
            gc.collect()
            torch.cuda.empty_cache()

        # 寫回 fused_model 這一層
        fused_block.load_state_dict(fused_sd)

    # 3. 存起來
    fused_model.save_pretrained(out_dir)
    print(f"✅ eRECALL fused -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--reprs", nargs="+", required=True)
    ap.add_argument("--entropy_file", required=True)
    ap.add_argument("--out_dir", default="./fused_erecall")
    args = ap.parse_args()

    # 1. 讀入 repr (CPU)
    repr_blobs = []
    for rep_path in args.reprs:
        blob = torch.load(rep_path, map_location="cpu")
        # blob["reprs"]: [N, L, H]
        repr_blobs.append({
            "task": blob["task"],
            "reprs": blob["reprs"],
        })

    # 2. 讀入 entropy buffer
    entropy_blob = torch.load(args.entropy_file, map_location="cpu")["entropy"].float()
    # 這個 entropy 是你的 high-entropy replay 重要性分數
    # 我們拿它來放大「難樣本」在相似度裡的影響力

    # 3. 算每層權重 (entropy-aware)
    weights_per_layer = compute_entropy_weights_per_layer(repr_blobs, entropy_blob)

    # 4. 加權融合 (省顯存版)
    fuse_models_entropy(args.base_model, args.adapters, weights_per_layer, args.out_dir)

if __name__ == "__main__":
    main()
