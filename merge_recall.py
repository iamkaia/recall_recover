import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel
import copy
import gc
import types
import inspect

def cosine_layer_sim(a_layer, b_layer):
    # a_layer: [Na, H], b_layer: [Nb, H]
    a_mean = a_layer.mean(dim=0, keepdim=True)  # [1,H]
    b_mean = b_layer.mean(dim=0, keepdim=True)  # [1,H]
    sim = F.cosine_similarity(a_mean, b_mean, dim=-1)  # [1]
    return sim.item()

def compute_layer_weights(repr_blobs):
    """
    repr_blobs: list of dicts:
      { "task": str, "reprs": Tensor[N,L,H] }
    回傳:
      weights_per_layer: list of Tensors[num_models]  (每一層的融合權重)
    """
    num_models = len(repr_blobs)
    anchor = repr_blobs[0]["reprs"]  # [N0, L, H]
    L = anchor.shape[1]

    weights_per_layer = []
    for layer_idx in range(L):
        sims = []
        for m_idx in range(num_models):
            cur = repr_blobs[m_idx]["reprs"][:, layer_idx, :]      # [Nm,H]
            anchor_layer = anchor[:, layer_idx, :]                 # [N0,H]
            sim_val = cosine_layer_sim(anchor_layer, cur)
            sims.append(sim_val)

        sims_t = torch.tensor(sims)  # [num_models]
        if sims_t.abs().sum() == 0:
            w = torch.ones_like(sims_t) / len(sims_t)
        else:
            # normalize to sum=1, keep negatives from blowing things? -> clamp at 0
            sims_clamped = torch.clamp(sims_t, min=0.0)
            if sims_clamped.sum() == 0:
                w = torch.ones_like(sims_t) / len(sims_t)
            else:
                w = sims_clamped / sims_clamped.sum()
        weights_per_layer.append(w)  # list length L, each [num_models]

    return weights_per_layer  # list of length L

def load_model_with_adapter(base_model_name, adapter_path):
    """
    載一個 base + adapter (4bit) 到 GPU。
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

def is_block_list(obj):
    """
    我們想辨識 "這是不是一串 transformer blocks"
    條件大概是:
      - obj 是 list / ModuleList / tuple 類似的可索引序列
      - len(obj) > 1
      - 元素有 weight / attn / mlp 之類常見屬性
    這很 heuristic，但夠用來找 Qwen2 / Llama2 block stack。
    """
    if not (hasattr(obj, "__len__") and hasattr(obj, "__getitem__")):
        return False
    if len(obj) < 2:
        return False
    first = obj[0]
    # 我們檢查這是不是 nn.Module 並且裡面有一些transformer block常見結構
    if not hasattr(first, "state_dict"):
        return False
    # 很寬鬆地接受，因為不同架構叫法不同
    # 我們只要它是一個有參數的 module 就好了
    sd = first.state_dict()
    return len(sd.keys()) > 0

def find_transformer_blocks(peft_model):
    """
    自動找出 "一層一層 transformer block" 的 list。
    我們會試很多可能的路徑：
      - model.model.layers
      - model.model.decoder.layers
      - model.model.transformer.blocks
      - model.base_model.model.layers
      - model.base_model.model.transformer.blocks
    等等
    """
    candidates = []

    # 全部潛在根節點：peft_model 本身 & peft_model.model & peft_model.base_model
    roots = []
    roots.append(peft_model)
    if hasattr(peft_model, "model"):
        roots.append(peft_model.model)
    if hasattr(peft_model, "base_model"):
        roots.append(peft_model.base_model)
    # base_model 可能還包一層 .model
        if hasattr(peft_model.base_model, "model"):
            roots.append(peft_model.base_model.model)

    # 從這些 roots 去找常見 field 名稱
    field_names = [
        "layers",
        "blocks",
        "h",
        "transformer",
        "decoder",
        "model",
    ]

    seen_objs = set()

    def collect(obj, depth=0):
        if obj is None:
            return
        if id(obj) in seen_objs:
            return
        seen_objs.add(id(obj))

        # 1. 直接測 obj 本身是不是 block list
        if is_block_list(obj):
            candidates.append(obj)

        # 2. 嘗試取屬性
        if depth > 3:
            return
        for name in dir(obj):
            if name.startswith("_"):
                continue
            if name in field_names:
                child = getattr(obj, name, None)
                if child is not None:
                    # child 可能本身就是list of blocks
                    if is_block_list(child):
                        candidates.append(child)
                    # 否則往下爬
                    collect(child, depth+1)

    for r in roots:
        collect(r, depth=0)

    # 選一個最像 transformer stack 的 candidate
    # heuristics：長度最大者
    if not candidates:
        raise RuntimeError("No transformer block list found in model structure.")
    best = max(candidates, key=lambda c: len(c))
    return best  # 這應該是 ModuleList-like of blocks

def fuse_models(base_model_name, adapter_paths, weights_per_layer, out_dir):
    """
    逐層加權融合，多卡記憶體友善版
    - adapter_paths: list[str]，每個任務的 LoRA checkpoint
    - weights_per_layer: list[L_full] of Tensor[num_models]
    """
    # 先載第一個做 fused 起點
    fused_model = load_model_with_adapter(base_model_name, adapter_paths[0])
    fused_model = fused_model.to("cuda")

    fused_blocks = find_transformer_blocks(fused_model)
    num_layers = len(fused_blocks)

    # hidden_states 的 L_full 可能比 num_layers 多 (因為還有 embedding / final norm)
    L_full = len(weights_per_layer)
    offset = L_full - num_layers
    if offset < 0:
        raise RuntimeError(
            f"repr layers ({L_full}) < transformer blocks ({num_layers}), can't align."
        )
    weights_per_layer_aligned = weights_per_layer[offset:]  # list[num_layers], each tensor[num_models]

    # 對每一層 transformer block 做加權平均
    for layer_idx in range(num_layers):
        w_layer = weights_per_layer_aligned[layer_idx]  # tensor[num_models]

        # 先抓 fused block (第0個模型) 的 state_dict 當 accumulator
        fused_block = fused_blocks[layer_idx]
        fused_sd = {k: v.detach().clone().to("cpu") for k,v in fused_block.state_dict().items()}

        # 先乘上自己的權重 w_layer[0]
        fused_sd = {k: fused_sd[k] * w_layer[0].item() for k in fused_sd.keys()}

        # 依序把剩下模型加進來
        for m_i in range(1, len(adapter_paths)):
            tmp_model = load_model_with_adapter(base_model_name, adapter_paths[m_i])
            tmp_blocks = find_transformer_blocks(tmp_model)
            tmp_block_sd = {k: v.detach().clone().to("cpu") for k,v in tmp_blocks[layer_idx].state_dict().items()}

            for key in fused_sd.keys():
                fused_sd[key] += tmp_block_sd[key] * w_layer[m_i].item()

            # 清掉暫時載的模型，釋放 VRAM
            del tmp_model, tmp_blocks, tmp_block_sd
            gc.collect()
            torch.cuda.empty_cache()

        # 把融合後結果 load 回 fused_model 對應層
        fused_block.load_state_dict(fused_sd)

    # 儲存 fused model
    fused_model.save_pretrained(out_dir)
    print(f"✅ Saved fused model to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--reprs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="./fused_recall")
    args = ap.parse_args()

    # 載 repr blobs (CPU)
    repr_blobs = []
    for rep_path in args.reprs:
        blob = torch.load(rep_path, map_location="cpu")
        repr_blobs.append({
            "task": blob["task"],
            "reprs": blob["reprs"],  # [N,L,H]
        })

    # 算每層融合權重
    weights_per_layer = compute_layer_weights(repr_blobs)
    # -> list[L_full], each is tensor[num_models]

    # 進行融合（VRAM 友善）
    fuse_models(args.base_model, args.adapters, weights_per_layer, args.out_dir)

if __name__ == "__main__":
    main()
