# train_sft_task.py
import os, argparse, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def get_dataset(name):
    # ↓ 每個任務轉成 text-generation 格式 (prompt + label)
    if name.lower() == "sst-2":
        ds = load_dataset("sst2")
        def fmt(e): return {"text": f"Sentiment: {e['sentence']}\nAnswer:", "label": str(e["label"])}
        ds = ds.map(fmt)
    elif name.lower() == "squad2.0":
        ds = load_dataset("squad_v2")
        def fmt(e): return {"text": f"Question: {e['question']}\nContext: {e['context']}\nAnswer:", "label": e["answers"]["text"][0] if e["answers"]["text"] else ""}
        ds = ds.map(fmt)
    elif "iwslt" in name.lower():
        ds = load_dataset("iwslt2017", "iwslt2017-en-fr")
        def fmt(e): return {"text": f"Translate English to French: {e['translation']['en']}\nFrench:", "label": e["translation"]["fr"]}
        ds = ds.map(fmt)
    elif name.lower() == "race":
        ds = load_dataset("race", "middle")
        def fmt(e): return {"text": f"Question: {e['question']}\nOptions: {', '.join(e['options'])}\nAnswer:", "label": e["answer"]}
        ds = ds.map(fmt)
    elif name.lower() == "medmcqa":
        ds = load_dataset("medmcqa")
        def fmt(e): return {"text": f"Medical question: {e['question']}\nOptions: {e['opa']}, {e['opb']}, {e['opc']}, {e['opd']}\nAnswer:", "label": e["cop"]}
        ds = ds.map(fmt)
    else:
        raise ValueError("Unsupported dataset")
    return ds["train"].shuffle(seed=42).select(range(2000))  # subset for 4090

def train(args):
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_4bit=True, device_map="auto", torch_dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.rank, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    ds = get_dataset(args.task)
    enc = tok(ds["text"], truncation=True, padding=True, max_length=512, return_tensors="pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    for step in range(args.steps):
        idx = step % enc["input_ids"].shape[0]
        inp = enc["input_ids"][idx:idx+1].to(device)
        out = model(inp, labels=inp)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 50 == 0:
            print(f"{args.task} step={step} loss={loss.item():.4f}")

    out_dir = f"./adapters/{args.task}"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"✅ Saved {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    train(args)
