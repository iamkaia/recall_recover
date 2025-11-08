'''
# extract_representations.py
import torch
import argparse
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@torch.no_grad()
def sample_entropy(model, tok, texts, device):
    """對每個樣本計算 token-level 的預測熵（僅供挑高熵樣本或fallback用）"""
    ent = []
    model.eval()
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128).to(device)
        out = model(**enc)
        logits = out.logits[:, :-1, :]                 # [1, seq-1, vocab]
        probs  = F.softmax(logits, dim=-1)
        token_entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
        ent.append(token_entropy)
    return ent  # list[float]

@torch.no_grad()
def get_reprs(model, tok, texts, device, max_len=256):
    """回傳 [N, L, H] 的逐層表徵（attention-mask 平均池化）"""
    model.eval()
    mats = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["attention_mask"].float()[:, :, None]           # [1,T,1]
        denom = mask.sum(dim=1).clamp_min(1.0)                     # [1,1]
        layer_vecs = []
        for h in out.hidden_states:                                # 每層 [1,T,H]
            pooled = (h * mask).sum(dim=1) / denom                 # [1,H]
            layer_vecs.append(pooled.squeeze(0).detach().to("cpu"))# [H]
        mats.append(torch.stack(layer_vecs, dim=0).unsqueeze(0))   # [1,L,H]
    return torch.cat(mats, dim=0)                                  # [N,L,H]

def main(args):
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, load_in_4bit=True, torch_dtype=torch.bfloat16, device_map={"":0}
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    with open(args.sample_file, "r") as f:
        texts = [x.strip() for x in f if x.strip()]
    texts = texts[:args.max_samples]

    # （可選）先用 entropy 挑一批樣本
    ent = sample_entropy(model, tok, texts, device)
    if args.entropy_select:
        k = max(1, int(len(texts) * args.top_ratio))
        idx = torch.topk(torch.tensor(ent), k).indices.tolist()
        texts = [texts[i] for i in idx]
        ent   = [ent[i]   for i in idx]
        print(f"🔥 High-entropy select: kept {k}/{len(ent)+len(idx)-k}")

    # 逐層表徵
    reprs = get_reprs(model, tok, texts, device, max_len=args.max_len)  # [N,L,H]

    # 逐層變異量（層級熵 proxy）：先對 N 個樣本計算每層的 hidden variance，再於 hidden 維度平均
    # layer_var: [L]，代表每一層在該批樣本上的不確定/差異程度
    layer_var = reprs.var(dim=0).mean(dim=1)   # [L]

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {"task": args.task, "reprs": reprs, "layer_var": layer_var},
        os.path.join(args.out_dir, f"{args.task}_repr.pt")
    )
    torch.save(
        {"task": args.task, "entropy": torch.tensor(ent)},  # 仍保留 per-sample entropy 作備援
        os.path.join(args.out_dir, f"{args.task}_entropy.pt")
    )
    print(f"✅ Saved: reprs+layer_var+entropy for {args.task} -> {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--sample_file", required=True)
    ap.add_argument("--out_dir", default="./reprs")
    ap.add_argument("--max_samples", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--entropy_select", action="store_true")
    ap.add_argument("--top_ratio", type=float, default=0.3)
    args = ap.parse_args()
    main(args)
'''
# extract_representations.py
import torch, argparse, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@torch.no_grad()
def get_reprs(model, tok, texts, device="cuda", max_len=256):
    model.eval()
    outs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["attention_mask"].float()[:, :, None]
        msum = mask.sum(dim=1).clamp_min(1.0)
        vecs = []
        for h in out.hidden_states:                   # [1, T, H]
            pooled = (h * mask).sum(dim=1) / msum     # [1, H]
            vecs.append(pooled.squeeze(0).to("cpu"))
        outs.append(torch.stack(vecs, dim=0))         # [L, H]
    return torch.stack(outs, dim=0)                   # [N, L, H]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--sample_file", required=True)
    ap.add_argument("--out_dir", default="./reprs")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_4bit=True, torch_dtype=torch.bfloat16, device_map={"":0})
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    with open(args.sample_file) as f:
        texts = [x.strip() for x in f if x.strip()]
    texts = texts[:args.max_samples]

    reprs = get_reprs(model, tok, texts, max_len=args.max_len)  # [N,L,H]
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"task": args.task, "reprs": reprs}, os.path.join(args.out_dir, f"{args.task}_repr.pt"))
    print(f"✅ saved {args.task} repr -> {args.out_dir}")

if __name__ == "__main__":
    main()
