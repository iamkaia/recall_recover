# evaluate_all_tasks.py
import argparse, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def evaluate(task, model, tok):
    correct, total = 0, 0
    if task.lower() == "sst-2":
        ds = load_dataset("sst2", split="validation[:200]")
        for e in tqdm(ds):
            text = f"Sentiment: {e['sentence']}\nAnswer:"
            enc = tok(text, return_tensors="pt").to("cuda")

            gen_out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                max_new_tokens=8,
            )

            out = tok.decode(gen_out[0], skip_special_tokens=True)

            if str(e["label"]) in out: correct += 1
            total += 1
    return correct/total if total>0 else 0.0

def main(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.bfloat16)
    results = {}
    for t in ["SST-2", "SQuAD2.0", "IWSLT2017-en-fr", "RACE", "MedMCQA"]:
        results[t] = evaluate(t, model, tok)
    print(results)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    main(args)
