'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
import sacrebleu

# -------------------
# Loader
# -------------------
def load_fused_model(fused_dir: str, base_model_name: str, device: str = "cuda"):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else None,
    )
    fused = PeftModel.from_pretrained(base, fused_dir)
    fused.eval()
    return fused

def load_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# -------------------
# Utils
# -------------------
@torch.no_grad()
def gen(model, tok, prompt, max_new_tokens=64, temperature=0.0):
    """
    產生文字；若不是取樣模式，就清掉 generation_config 裡的 top_p/top_k/temperature 以避免警告。
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # ✔ 修正：用 to_dict() 重建一份 GenerationConfig（舊版沒有 .clone()）
    gcfg = GenerationConfig(**model.generation_config.to_dict())
    if temperature and temperature > 0:
        gcfg.do_sample = True
        gcfg.temperature = float(temperature)
    else:
        gcfg.do_sample = False
        # 清掉 sample-only 參數，避免警告
        gcfg.temperature = None
        gcfg.top_p = None
        gcfg.top_k = None

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        generation_config=gcfg,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.no_grad()
def option_logprob(model, tok, context: str, option: str):
    """
    Multiple-choice 打分：計算 P(option | context) 的 logprob（逐 token 相加）。
    """
    enc_ctx = tok(context, return_tensors="pt").to(model.device)
    enc_opt = tok(option, add_special_tokens=False, return_tensors="pt").to(model.device)

    input_ids = torch.cat([enc_ctx["input_ids"], enc_opt["input_ids"]], dim=1)
    attn = torch.cat([enc_ctx["attention_mask"], enc_opt["attention_mask"]], dim=1)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [1, T, V]

    ctx_len = enc_ctx["input_ids"].shape[1]
    opt_ids = enc_opt["input_ids"][0]

    # 對齊到 option 第一個 token 的預測位置（next token）
    opt_logits = logits[0, ctx_len-1:ctx_len-1+opt_ids.shape[0], :]  # [len(opt), V]
    logprobs = torch.log_softmax(opt_logits, dim=-1)
    gather = logprobs.gather(-1, opt_ids.unsqueeze(-1)).squeeze(-1)
    return float(gather.sum().item())

# -------------------
# Evaluators
# -------------------
@torch.no_grad()
def eval_sst2(model, tok, max_examples: int = 200):
    ds = load_dataset("glue", "sst2", split="validation")
    n = min(max_examples, len(ds))
    correct = 0
    for i in range(n):
        text = ds[i]["sentence"]
        ctx = f"Review: {text}\nSentiment (Positive/Negative):"
        lp_pos = option_logprob(model, tok, ctx + " ", " Positive")
        lp_neg = option_logprob(model, tok, ctx + " ", " Negative")
        pred = 1 if lp_pos > lp_neg else 0
        if pred == ds[i]["label"]:
            correct += 1
    return correct / n if n > 0 else 0.0

@torch.no_grad()
def eval_squad(model, tok, max_examples: int = 200):
    ds = load_dataset("squad_v2", split="validation")
    n = min(max_examples, len(ds))
    em = 0
    for i in range(n):
        ex = ds[i]
        ctx = ex["context"]
        q = ex["question"]
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        ans = gen(model, tok, prompt, max_new_tokens=32, temperature=0.0)
        pred = normalize_text(ans)
        golds = [normalize_text(a) for a in (ex["answers"]["text"] or [""])]
        em += 1 if any(pred == g for g in golds) else 0
    return em / n if n > 0 else 0.0

@torch.no_grad()
def eval_iwslt(model, tok, max_examples: int = 200):
    ds = load_dataset("iwslt2017", "iwslt2017-en-fr", split="test")
    n = min(max_examples, len(ds))
    preds, refs = [], []
    for i in range(n):
        src = ds[i]["translation"]["en"]
        ref = ds[i]["translation"]["fr"]
        prompt = f"Translate to French:\nEnglish: {src}\nFrench:"
        out = gen(model, tok, prompt, max_new_tokens=64, temperature=0.0)
        preds.append(out.strip())
        refs.append(ref.strip())
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score  # 0~100
    return bleu / 100.0

@torch.no_grad()
def eval_race(model, tok, max_examples: int = 200):
    """
    取 RACE-middle + RACE-high 的驗證集子集，合併需用 concatenate_datasets（不能用 +）
    """
    ds_m = load_dataset("race", "middle", split="validation")
    ds_h = load_dataset("race", "high", split="validation")
    take_m = min(len(ds_m), max_examples // 2)
    take_h = min(len(ds_h), max_examples - take_m)
    ds = concatenate_datasets([ds_m.select(range(take_m)), ds_h.select(range(take_h))])
    n = len(ds)
    if n == 0:
        return 0.0

    correct = 0
    for i in range(n):
        ex = ds[i]
        article = ex["article"]
        question = ex["question"]
        options = ex["options"]  # ['optA','optB','optC','optD']
        answer = ex["answer"]    # 'A'/'B'/'C'/'D'
        ctx = f"Passage:\n{article}\n\nQuestion: {question}\nAnswer:"
        lps = [option_logprob(model, tok, ctx + " ", f" {chr(65+j)}. {opt}") for j, opt in enumerate(options)]
        pred_idx = int(torch.tensor(lps).argmax().item())
        gold_idx = ord(answer) - 65
        if pred_idx == gold_idx:
            correct += 1
    return correct / n

@torch.no_grad()
def eval_medmcqa(model, tok, max_examples: int = 200):
    ds = load_dataset("medmcqa", split="validation")
    n = min(max_examples, len(ds))
    if n == 0:
        return 0.0

    def gold_to_index(gold):
        # 支援 'a'/'b'/'c'/'d'、'A'~'D'、0~3、1~4、"0"/"1"/"2"/"3"/"1"/"2"/"3"/"4"
        if isinstance(gold, int):
            if gold in (0, 1, 2, 3):
                return gold
            if gold in (1, 2, 3, 4):
                return gold - 1
        if isinstance(gold, str):
            g = gold.strip().lower()
            map_abcd = {"a": 0, "b": 1, "c": 2, "d": 3}
            if g in map_abcd:
                return map_abcd[g]
            if g.isdigit():
                gi = int(g)
                if gi in (0, 1, 2, 3):
                    return gi
                if gi in (1, 2, 3, 4):
                    return gi - 1
        # 落到這裡就拋錯讓你注意資料異常
        raise ValueError(f"Unrecognized gold label format for MedMCQA: {gold!r}")

    correct = 0
    for i in range(n):
        ex = ds[i]
        q = ex["question"]

        # 兼容不同欄位：有些 split/版本是 'options'，有些是 'opa'~'opd'
        if "options" in ex and isinstance(ex["options"], list) and len(ex["options"]) >= 4:
            opts = ex["options"][:4]
        else:
            opts = [ex.get("opa", ""), ex.get("opb", ""), ex.get("opc", ""), ex.get("opd", "")]

        # 正解欄位可能叫 'cop' 或 'answer'
        gold_raw = ex.get("cop", ex.get("answer", None))
        gold_idx = gold_to_index(gold_raw)

        ctx = f"Medical question:\n{q}\nAnswer:"
        lps = [option_logprob(model, tok, ctx + " ", f" {chr(65+j)}. {opt}") for j, opt in enumerate(opts)]
        pred_idx = int(torch.tensor(lps).argmax().item())

        if pred_idx == gold_idx:
            correct += 1

    return correct / n

# -------------------
# Main
# -------------------
def main(args):
    tok = load_tokenizer(args.base_model)
    model = load_fused_model(args.model, args.base_model, device=args.device)

    results = {
        "SST-2": eval_sst2(model, tok, args.max_examples),
        "SQuAD2.0": eval_squad(model, tok, args.max_examples),
        "IWSLT2017-en-fr": eval_iwslt(model, tok, args.max_examples),
        "RACE": eval_race(model, tok, args.max_examples),
        "MedMCQA": eval_medmcqa(model, tok, args.max_examples),
    }
    # 只輸出單行 JSON，方便被 analyze_results.py 讀取
    print(json.dumps(results))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to fused adapter dir (e.g., ./fused_recall or ./fused_erecall)")
    p.add_argument("--base_model", default="Qwen/Qwen2-7B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_examples", type=int, default=200, help="Per-task eval cap for speed")
    args = p.parse_args()
    main(args)
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from datasets import load_dataset, concatenate_datasets
import sacrebleu

# -------------------
# Loader
# -------------------
def load_fused_model(fused_dir: str, base_model_name: str, device: str = "cuda"):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0} if device == "cuda" else None,
    )
    fused = PeftModel.from_pretrained(base, fused_dir)
    fused.eval()
    return fused

def load_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# -------------------
# Utils
# -------------------
@torch.no_grad()
def gen(model, tok, prompt, max_new_tokens=64, temperature=0.0):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gcfg = GenerationConfig(**model.generation_config.to_dict())
    if temperature and float(temperature) > 0.0:
        gcfg.do_sample = True
        gcfg.temperature = float(temperature)
    else:
        gcfg.do_sample = False
        gcfg.temperature = None
        gcfg.top_p = None
        gcfg.top_k = None

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        generation_config=gcfg,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@torch.no_grad()
def option_logprob(model, tok, context: str, option: str):
    enc_ctx = tok(context, return_tensors="pt").to(model.device)
    enc_opt = tok(option, add_special_tokens=False, return_tensors="pt").to(model.device)

    input_ids = torch.cat([enc_ctx["input_ids"], enc_opt["input_ids"]], dim=1)
    attn = torch.cat([enc_ctx["attention_mask"], enc_opt["attention_mask"]], dim=1)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits  # [1, T, V]

    ctx_len = enc_ctx["input_ids"].shape[1]
    opt_ids = enc_opt["input_ids"][0]

    opt_logits = logits[0, ctx_len-1:ctx_len-1+opt_ids.shape[0], :]  # [len(opt), V]
    logprobs = torch.log_softmax(opt_logits, dim=-1)
    gather = logprobs.gather(-1, opt_ids.unsqueeze(-1)).squeeze(-1)
    return float(gather.sum().item())

# -------------------
# Evaluators
# -------------------
@torch.no_grad()
def eval_sst2(model, tok, max_examples: int = 200):
    ds = load_dataset("glue", "sst2", split="validation")
    n = min(max_examples, len(ds))
    correct = 0
    for i in range(n):
        text = ds[i]["sentence"]
        ctx = f"Review: {text}\nSentiment (Positive/Negative):"
        lp_pos = option_logprob(model, tok, ctx + " ", " Positive")
        lp_neg = option_logprob(model, tok, ctx + " ", " Negative")
        pred = 1 if lp_pos > lp_neg else 0
        if pred == ds[i]["label"]:
            correct += 1
    return correct / n if n > 0 else 0.0

@torch.no_grad()
def eval_squad(model, tok, max_examples: int = 200):
    ds = load_dataset("squad_v2", split="validation")
    n = min(max_examples, len(ds))
    em = 0
    for i in range(n):
        ex = ds[i]
        ctx = ex["context"]
        q = ex["question"]
        prompt = f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        ans = gen(model, tok, prompt, max_new_tokens=32, temperature=0.0)
        pred = normalize_text(ans)
        golds = [normalize_text(a) for a in (ex["answers"]["text"] or [""])]
        em += 1 if any(pred == g for g in golds) else 0
    return em / n if n > 0 else 0.0

@torch.no_grad()
def eval_iwslt(model, tok, max_examples: int = 200):
    # 盡量用 test；沒有就 fallback 到 validation，再不行就 train（只取少量）
    ds = None
    for split in ["test", "validation", "train"]:
        try:
            ds_try = load_dataset("iwslt2017", "iwslt2017-en-fr", split=split)
            if len(ds_try) > 0:
                ds = ds_try
                break
        except Exception:
            continue

    if ds is None or len(ds) == 0:
        # 安全返回，避免 sacrebleu 在空列表上炸掉
        return 0.0

    n = min(max_examples, len(ds))
    if n <= 0:
        return 0.0

    preds, refs = [], []
    for i in range(n):
        rec = ds[i]
        # 兼容不同欄位名
        trans = rec.get("translation", None)
        if trans is None or "en" not in trans or "fr" not in trans:
            # 非預期樣本，跳過
            continue
        src = trans["en"]
        ref = trans["fr"]
        prompt = f"Translate to French:\nEnglish: {src}\nFrench:"
        out = gen(model, tok, prompt, max_new_tokens=64, temperature=0.0)
        preds.append(out.strip())
        refs.append(ref.strip())

    # 再次保護（若前面全部被跳過）
    if len(preds) == 0 or len(refs) == 0:
        return 0.0

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score  # 0~100
    return bleu / 100.0

@torch.no_grad()
def eval_race(model, tok, max_examples: int = 200):
    ds_m = load_dataset("race", "middle", split="validation")
    ds_h = load_dataset("race", "high", split="validation")
    take_m = min(len(ds_m), max_examples // 2)
    take_h = min(len(ds_h), max_examples - take_m)
    ds = concatenate_datasets([ds_m.select(range(take_m)), ds_h.select(range(take_h))])
    n = len(ds)
    if n == 0:
        return 0.0

    correct = 0
    for i in range(n):
        ex = ds[i]
        article = ex["article"]
        question = ex["question"]
        options = ex["options"]
        answer = ex["answer"]    # 'A'/'B'/'C'/'D'
        ctx = f"Passage:\n{article}\n\nQuestion: {question}\nAnswer:"
        lps = [option_logprob(model, tok, ctx + " ", f" {chr(65+j)}. {opt}") for j, opt in enumerate(options)]
        pred_idx = int(torch.tensor(lps).argmax().item())
        gold_idx = ord(answer) - 65
        if pred_idx == gold_idx:
            correct += 1
    return correct / n

@torch.no_grad()
def eval_medmcqa(model, tok, max_examples: int = 200):
    ds = load_dataset("medmcqa", split="validation")
    n = min(max_examples, len(ds))
    if n == 0:
        return 0.0

    def gold_to_index(gold):
        if isinstance(gold, int):
            if gold in (0, 1, 2, 3): return gold
            if gold in (1, 2, 3, 4): return gold - 1
        if isinstance(gold, str):
            g = gold.strip().lower()
            map_abcd = {"a": 0, "b": 1, "c": 2, "d": 3}
            if g in map_abcd: return map_abcd[g]
            if g.isdigit():
                gi = int(g)
                if gi in (0, 1, 2, 3): return gi
                if gi in (1, 2, 3, 4): return gi - 1
        raise ValueError(f"Unrecognized gold label: {gold!r}")

    correct = 0
    for i in range(n):
        ex = ds[i]
        q = ex["question"]
        if "options" in ex and isinstance(ex["options"], list) and len(ex["options"]) >= 4:
            opts = ex["options"][:4]
        else:
            opts = [ex.get("opa", ""), ex.get("opb", ""), ex.get("opc", ""), ex.get("opd", "")]
        gold_idx = gold_to_index(ex.get("cop", ex.get("answer", None)))

        ctx = f"Medical question:\n{q}\nAnswer:"
        lps = [option_logprob(model, tok, ctx + " ", f" {chr(65+j)}. {opt}") for j, opt in enumerate(opts)]
        pred_idx = int(torch.tensor(lps).argmax().item())
        if pred_idx == gold_idx:
            correct += 1
    return correct / n

# -------------------
# Main
# -------------------
def main(args):
    tok = load_tokenizer(args.base_model)
    model = load_fused_model(args.model, args.base_model, device=args.device)

    results = {
        "SST-2": eval_sst2(model, tok, args.max_examples),
        "SQuAD2.0": eval_squad(model, tok, args.max_examples),
        "IWSLT2017-en-fr": eval_iwslt(model, tok, args.max_examples),
        "RACE": eval_race(model, tok, args.max_examples),
        "MedMCQA": eval_medmcqa(model, tok, args.max_examples),
    }
    print(json.dumps(results))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to fused adapter dir (e.g., ./fused_recall or ./fused_he_recall)")
    p.add_argument("--base_model", default="Qwen/Qwen2-7B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_examples", type=int, default=200)
    args = p.parse_args()
    main(args)
