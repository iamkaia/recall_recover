'''
# select_typicals.py
import argparse, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.cluster import KMeans

@torch.no_grad()
def sent_rep(model, tok, text, device, max_len=256):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len, padding=True).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    mask = enc["attention_mask"].float()[:, :, None]
    mask_sum = mask.sum(dim=1).clamp_min(1.0)
    # 推薦用「中層或多層拼接」做聚類向量；這裡示範取中層
    mid = len(out.hidden_states)//2
    h = out.hidden_states[mid]                               # [1,T,H]
    vec = (h * mask).sum(dim=1) / mask_sum                  # [1,H]
    return vec.squeeze(0).detach().cpu().numpy()            # [H]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)        # 這就是 M_N 的 adapter
    ap.add_argument("--in_txt", required=True)         # D_N 的原始候選文本（可用你現有 samples 檔做母集合）
    ap.add_argument("--out_txt", required=True)        # 典型樣本輸出（給所有模型共用）
    ap.add_argument("--k", type=int, default=20)       # 典型樣本數 m
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_4bit=True, torch_dtype=torch.bfloat16, device_map={"":0})
    Mn = PeftModel.from_pretrained(base, args.adapter); Mn.eval()

    texts = [line.strip() for line in open(args.in_txt) if line.strip()]
    X = []
    for t in texts:
        X.append(sent_rep(Mn, tok, t, device="cuda"))
    X = np.stack(X, axis=0)                               # [N, H]

    km = KMeans(n_clusters=args.k, n_init="auto", random_state=0)
    km.fit(X)
    # 選每個 cluster 中心最近的樣本
    from sklearn.metrics import pairwise_distances_argmin_min
    idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    chosen = [texts[i] for i in idx]

    with open(args.out_txt, "w") as f:
        for s in chosen: f.write(s+"\n")
    print(f"Saved {len(chosen)} typical samples to {args.out_txt}")

if __name__ == "__main__":
    main()
'''
# select_typicals.py
import argparse, os, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np
from sklearn.cluster import KMeans

@torch.no_grad()
def embed_texts(model, tok, texts, device="cuda", max_len=256):
    arr = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # 用倒數第2層（一般較語義穩定），做 attention-mask 平均
        hs = out.hidden_states[-2]             # [1, T, H]
        mask = enc["attention_mask"].float()[:, :, None]
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        arr.append(pooled.squeeze(0).to("cpu").numpy())
    return np.stack(arr, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    with open(args.in_txt) as f:
        texts = [x.strip() for x in f if x.strip()]
    if len(texts) <= args.k:
        os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
        with open(args.out_txt, "w") as g:
            g.write("\n".join(texts))
        print(f"[typicals] take-all {len(texts)} -> {args.out_txt}")
        return

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_4bit=True, torch_dtype=torch.bfloat16, device_map={"":0})
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    X = embed_texts(model, tok, texts)
    kmeans = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    # 取每群距中心最近者
    idx = []
    for c in range(args.k):
        cand = np.where(kmeans.labels_==c)[0]
        sub = X[cand]
        j = np.linalg.norm(sub - centers[c:c+1], axis=1).argmin()
        idx.append(int(cand[j]))
    chosen = [texts[i] for i in idx]
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
    with open(args.out_txt, "w") as g:
        g.write("\n".join(chosen))
    print(f"[typicals] wrote {args.out_txt} ({len(chosen)})")

if __name__ == "__main__":
    main()
