# merge_recall.py
import argparse
import os
import json
import shutil
import gc
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ---------------------------
# utils: 找到 transformer blocks
# ---------------------------
def is_block_list(obj):
    if not (hasattr(obj, "__len__") and hasattr(obj, "__getitem__")):
        return False
    if len(obj) < 2:
        return False
    first = obj[0]
    if not hasattr(first, "state_dict"):
        return False
    return len(first.state_dict().keys()) > 0

def find_transformer_blocks(peft_model):
    candidates = []
    roots = [peft_model]
    if hasattr(peft_model, "model"):
        roots.append(peft_model.model)
    if hasattr(peft_model, "base_model"):
        roots.append(peft_model.base_model)
        if hasattr(peft_model.base_model, "model"):
            roots.append(peft_model.base_model.model)

    field_names = ["layers", "blocks", "h", "transformer", "decoder", "model"]
    seen = set()

    def collect(obj, depth=0):
        if obj is None or id(obj) in seen or depth > 3:
            return
        seen.add(id(obj))

        if is_block_list(obj):
            candidates.append(obj)

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
    print(f"[find_transformer_blocks] picked {len(best)} layers")
    return best

def load_model_with_adapter(base_model_name, adapter_path):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,              # 新版會 warning，但仍可用；若要乾淨可改 BitsAndBytesConfig
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    peft_model.eval()
    return peft_model

# ---------------------------
# 權重計算（RECALL baseline）
# ---------------------------
def cosine_layer_sim(anchor_layer, cur_layer):
    """
    anchor_layer: [Na, H]
    cur_layer   : [Nc, H]
    以樣本平均後的向量做 cosine，相當於「該層的語義中心」相似度。
    """
    a = anchor_layer.mean(dim=0, keepdim=True)  # [1,H]
    b = cur_layer.mean(dim=0, keepdim=True)     # [1,H]
    return F.cosine_similarity(a, b, dim=-1).item()  # scalar

def compute_layer_weights(repr_blobs, anchor_task=None, temp_head=1.0, temp_tail=1.0):
    """
    repr_blobs: list of dicts { "task": str, "reprs": Tensor[N, L, H] }
    anchor_task: 選哪個任務當 anchor（若 None -> 取第 0 個）
    temp_head / temp_tail: 逐層溫度（從靠近 embedding 的頭層到尾層）線性插值

    回傳: list[L_full]，每個元素是 Tensor[num_models]（該層各模型的權重，softmax 後）
    """
    # 找 anchor index
    anchor_idx = 0
    if anchor_task is not None:
        for i, b in enumerate(repr_blobs):
            if str(b.get("task", "")).strip() == str(anchor_task).strip():
                anchor_idx = i
                break

    num_models = len(repr_blobs)
    L_full = repr_blobs[anchor_idx]["reprs"].shape[1]
    print(f"[weights] anchor = {repr_blobs[anchor_idx]['task']} (index={anchor_idx}), layers={L_full}")

    def layer_temp(i):
        if L_full <= 1:
            return float(temp_tail)
        # 線性插值：i=0 用 temp_head, i=L_full-1 用 temp_tail
        t = temp_head + (temp_tail - temp_head) * (i / (L_full - 1))
        return float(max(1e-6, t))

    weights_per_layer = []
    for i in range(L_full):
        sims = []
        a_layer = repr_blobs[anchor_idx]["reprs"][:, i, :]  # [Na,H]
        for m_idx in range(num_models):
            cur_layer = repr_blobs[m_idx]["reprs"][:, i, :]  # [Nm,H]
            sim = cosine_layer_sim(a_layer, cur_layer)
            sims.append(sim)

        sims_t = torch.tensor(sims, dtype=torch.float32)  # [num_models]
        # 將負相似度 clamp 至 0，可避免「反向權重」
        sims_t = torch.clamp(sims_t, min=0.0)

        # 使用 softmax / temperature
        temp = layer_temp(i)
        if temp != 1.0:
            sims_t = sims_t / temp
        if sims_t.abs().sum() == 0:
            w = torch.ones_like(sims_t) / len(sims_t)
        else:
            w = torch.softmax(sims_t, dim=0)

        weights_per_layer.append(w)
    return weights_per_layer

# ---------------------------
# 模型融合 + 正確儲存 LoRA
# ---------------------------
def fuse_models(base_model_name, adapter_paths, weights_per_layer, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    fused_model = load_model_with_adapter(base_model_name, adapter_paths[0]).to("cuda")
    fused_blocks = find_transformer_blocks(fused_model)
    num_layers = len(fused_blocks)

    L_full = len(weights_per_layer)
    offset = L_full - num_layers
    if offset < 0:
        raise RuntimeError(
            f"repr layers ({L_full}) < transformer blocks ({num_layers}), can't align."
        )
    weights_aligned = weights_per_layer[offset:]  # list[num_layers]

    # 逐層融合
    for layer_idx in range(num_layers):
        w_layer = weights_aligned[layer_idx]  # tensor[num_models]
        fused_block = fused_blocks[layer_idx]

        fused_sd = {k: v.detach().clone().to("cpu") for k, v in fused_block.state_dict().items()}
        # 先乘第 0 個模型的權重
        for k in fused_sd.keys():
            fused_sd[k] = fused_sd[k] * w_layer[0].item()

        # 疊加其他專家
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

    # -------- 正確儲存 LoRA（避免 adapter_config.json 壞掉）--------
    ok = False
    try:
        fused_model.save_pretrained(out_dir)
        ok = os.path.isfile(os.path.join(out_dir, "adapter_model.bin")) or \
             os.path.isfile(os.path.join(out_dir, "adapter_model.safetensors"))
    except Exception as e:
        print(f"[WARN] save_pretrained failed: {e}")

    if not ok:
        print("[WARN] Default save failed, manually saving LoRA weights...")
        # 1) 只存 LoRA 權重
        try:
            lora_state = fused_model.get_peft_model_state_dict()
        except Exception:
            full_sd = fused_model.state_dict()
            lora_state = {k: v for k, v in full_sd.items() if "lora_" in k}
        torch.save(lora_state, os.path.join(out_dir, "adapter_model.bin"))

        # 2) adapter_config.json：從第一個 adapter 拷貝（保證 target_modules/r/alpha 等一致）
        src_cfg = os.path.join(adapter_paths[0], "adapter_config.json")
        dst_cfg = os.path.join(out_dir, "adapter_config.json")
        if os.path.isfile(src_cfg):
            shutil.copy(src_cfg, dst_cfg)
        else:
            # 極少見：若來源沒有 config，就用 peft_config 產一份最小可用版本
            try:
                peft_cfg_obj = list(fused_model.peft_config.values())[0]
                with open(dst_cfg, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "peft_type": str(getattr(peft_cfg_obj, "peft_type", "LORA")),
                            "task_type": str(getattr(peft_cfg_obj, "task_type", "CAUSAL_LM")),
                            "base_model_name_or_path": getattr(peft_cfg_obj, "base_model_name_or_path", ""),
                            "r": int(getattr(peft_cfg_obj, "r", 16)),
                            "lora_alpha": int(getattr(peft_cfg_obj, "lora_alpha", 16)),
                            "lora_dropout": float(getattr(peft_cfg_obj, "lora_dropout", 0.0)),
                            "bias": getattr(peft_cfg_obj, "bias", "none"),
                            "target_modules": getattr(peft_cfg_obj, "target_modules", None),
                            "inference_mode": getattr(peft_cfg_obj, "inference_mode", True),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                raise RuntimeError(
                    "Cannot build adapter_config.json; please ensure the source adapters contain adapter_config.json"
                ) from e

    # 檢查輸出完整性
    have_cfg = os.path.isfile(os.path.join(out_dir, "adapter_config.json"))
    have_bin = os.path.isfile(os.path.join(out_dir, "adapter_model.bin")) or \
               os.path.isfile(os.path.join(out_dir, "adapter_model.safetensors"))
    if not (have_cfg and have_bin):
        raise RuntimeError(f"Fuse saved, but adapter files missing in {out_dir}")

    print(f"✅ Saved fused adapter to {out_dir}")

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True, help="LoRA adapters (each task expert)")
    ap.add_argument("--reprs",    nargs="+", required=True, help="*.pt produced by extract_representations.py")
    ap.add_argument("--anchor_task", type=str, default=None, help="task name to use as anchor (default: reprs[0])")
    ap.add_argument("--temp_head", type=float, default=1.0, help="temperature at shallow layers")
    ap.add_argument("--temp_tail", type=float, default=1.0, help="temperature at deep layers")
    ap.add_argument("--out_dir", default="./fused_recall")
    args = ap.parse_args()

    # 讀 repr blobs（CPU）
    repr_blobs = []
    for p in args.reprs:
        blob = torch.load(p, map_location="cpu")
        if "reprs" not in blob:
            raise ValueError(f"{p} missing key 'reprs'")
        if "task" not in blob:
            # 若沒 task，就從檔名猜
            guess = os.path.basename(p).replace("_repr.pt", "")
            blob["task"] = guess
        repr_blobs.append({"task": blob["task"], "reprs": blob["reprs"]})

    # 算每層權重
    weights = compute_layer_weights(
        repr_blobs,
        anchor_task=args.anchor_task,
        temp_head=args.temp_head,
        temp_tail=args.temp_tail,
    )

    # 融合 + 正確輸出 LoRA
    fuse_models(args.base_model, args.adapters, weights, args.out_dir)

if __name__ == "__main__":
    main()
