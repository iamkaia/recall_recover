import torch
import argparse
import os
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

@torch.no_grad()
def sample_entropy(model, tok, texts, device):
    ent = []
    model.eval()
    for t in texts:
        enc = tok(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        # 我們只要 logits 來估不確定性
        out = model(**enc)
        # out.logits: [1, seq, vocab]
        logits = out.logits[:, :-1, :]  # shift掉最後一個token
        probs = F.softmax(logits, dim=-1)
        # 熵 H = -p log p
        token_entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
        ent.append(token_entropy)
    return ent  # list[float], len = len(texts)

@torch.no_grad()
def get_reprs(model, tok, texts, device, max_len=256):
    """
    回傳 tensor shape: [num_layers, hidden_dim]
    這裡的 num_layers 指的是所有 hidden_states 堆起來 (包含 embedding 層)
    我們會對每個 text 取一份，再把多個 text 疊成 [N, num_layers, hidden_dim]
    """
    model.eval()
    all_samples = []

    for t in texts:
        enc = tok(
            t,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(device)

        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,  # 關掉快取，避免 past_key_values 警告/干擾
        )
        # out.hidden_states is a tuple: (emb, layer1, layer2, ..., final)
        # 每一層: [1, seq, hidden_dim]
        layer_vecs = []
        for layer_h in out.hidden_states:
            # mean pool over sequence -> [hidden_dim]
            pooled = layer_h.mean(dim=1).squeeze(0).detach().to("cpu")
            layer_vecs.append(pooled)  # list of [hidden_dim]

        # 堆成 [num_layers, hidden_dim]
        layer_mat = torch.stack(layer_vecs, dim=0)  # [L, H]
        all_samples.append(layer_mat.unsqueeze(0))  # [1, L, H]

    # 最後得到 [N, L, H]
    all_samples = torch.cat(all_samples, dim=0)
    return all_samples  # torch.Tensor

def main(args):
    device = "cuda"

    # 載 tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 載 base model in 4-bit
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"":0},
    )

    # 套 LoRA (task adapter)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # 載入文字樣本（typical samples）
    with open(args.sample_file, "r") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]

    # 先限制數量
    texts = texts[:args.max_samples]

    # 算每個樣本的不確定性熵
    ent = sample_entropy(model, tok, texts, device)

    # 如果有啟動熵挑樣本，留下 top k 高熵的
    if args.entropy_select:
        ent_tensor = torch.tensor(ent)
        k = max(1, int(len(texts) * args.top_ratio))
        top_idx = torch.topk(ent_tensor, k).indices.tolist()
        texts = [texts[i] for i in top_idx]
        ent   = [ent[i]   for i in top_idx]
        print(f"🔥 High-entropy select: kept {k}/{len(ent_tensor)}")

    # 抽 hidden representation
    reprs = get_reprs(model, tok, texts, device, max_len=args.max_len)

    # 存結果
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {
            "task": args.task,
            "reprs": reprs,  # [N, L, H]
        },
        os.path.join(args.out_dir, f"{args.task}_repr.pt")
    )
    torch.save(
        {
            "task": args.task,
            "entropy": torch.tensor(ent),  # [N]
        },
        os.path.join(args.out_dir, f"{args.task}_entropy.pt")
    )
    print(f"✅ Saved reprs+entropy for {args.task} to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--sample_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./reprs")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--entropy_select", action="store_true")
    parser.add_argument("--top_ratio", type=float, default=0.3)
    args = parser.parse_args()
    main(args)
