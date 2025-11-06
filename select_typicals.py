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
