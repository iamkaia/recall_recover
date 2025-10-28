# prepare_samples.py
from datasets import load_dataset
import os, random

os.makedirs("samples", exist_ok=True)

def save_lines(name, lines):
    with open(f"samples/{name}.txt", "w") as f:
        for l in lines: f.write(l.strip()+"\n")
    print(f"✅ samples/{name}.txt ({len(lines)} lines)")

# 1. SST-2
ds = load_dataset("glue", "sst2", split="validation")
save_lines("SST-2", [x["sentence"] for x in ds.select(range(300))])

# 2. SQuAD2.0
ds = load_dataset("squad_v2", split="validation")
lines = [f"Question: {x['question']} Context: {x['context']}" for x in ds.select(range(200))]
save_lines("SQuAD2.0", lines)

# 3. IWSLT2017-en-fr
ds = load_dataset("iwslt2017", "iwslt2017-en-fr", split="train")
lines = [f"Translate English to French: {x['translation']['en']}" for x in ds.select(range(200))]
save_lines("IWSLT2017-en-fr", lines)

# 4. RACE
ds = load_dataset("race", "middle", split="validation")
lines = [f"Question: {x['question']} Options: {', '.join(x['options'])}" for x in ds.select(range(200))]
save_lines("RACE", lines)

# 5. MedMCQA
ds = load_dataset("medmcqa", split="validation")
lines = [f"Medical question: {x['question']} Options: {x['opa']}, {x['opb']}, {x['opc']}, {x['opd']}" for x in ds.select(range(200))]
save_lines("MedMCQA", lines)
