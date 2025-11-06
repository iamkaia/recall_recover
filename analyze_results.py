'''
import ast
import csv
import os
import math

RECALL_FILE = "recall_results.txt"
ERECALL_FILE = "erecall_results.txt"
OUT_CSV = "summary_table.csv"

TASK_ORDER = [
    "SST-2",
    "SQuAD2.0",
    "IWSLT2017-en-fr",
    "RACE",
    "MedMCQA",
]

def safe_load_results(path):
    """
    讀一個像 "{'SST-2': 0.9, 'SQuAD2.0': 0.6, ...}" 的檔案
    沒有就回全 NaN，這樣也能照常輸出 csv。
    """
    if not os.path.exists(path):
        print(f"[analyze] WARNING: {path} not found, filling with NaN.")
        return {task: float("nan") for task in TASK_ORDER}

    with open(path, "r") as f:
        txt = f.read().strip()

    # 用 ast.literal_eval 把字串轉回 dict (安全版 eval)
    try:
        data = ast.literal_eval(txt)
    except Exception as e:
        print(f"[analyze] ERROR: cannot parse {path}: {e}")
        # fallback: 全 NaN
        data = {task: float("nan") for task in TASK_ORDER}

    # 確保所有 task 都有 key，沒有就補 NaN
    norm = {}
    for t in TASK_ORDER:
        norm[t] = float(data[t]) if t in data else float("nan")
    return norm

def average_score(d):
    """算平均，只算不是 NaN 的值。"""
    vals = [v for v in d.values() if not (isinstance(v, float) and math.isnan(v))]
    if len(vals) == 0:
        return float("nan")
    return sum(vals) / len(vals)

def main():
    # 1. 讀 RECALL baseline
    recall_scores = safe_load_results(RECALL_FILE)
    recall_avg = average_score(recall_scores)

    # 2. 讀 eRECALL (ours)
    erecall_scores = safe_load_results(ERECALL_FILE)
    erecall_avg = average_score(erecall_scores)

    # 3. 準備寫 CSV
    rows = []

    recall_row = ["RECALL"]
    for t in TASK_ORDER:
        recall_row.append(recall_scores[t])
    recall_row.append(recall_avg)
    rows.append(recall_row)

    erecall_row = ["eRECALL(ours)"]
    for t in TASK_ORDER:
        erecall_row.append(erecall_scores[t])
    erecall_row.append(erecall_avg)
    rows.append(erecall_row)

    header = ["model"] + TASK_ORDER + ["avg"]

    # 4. 寫 summary_table.csv
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    # 5. 同時印在 terminal，方便你直接看結果
    print("=== SUMMARY ===")
    print(",".join(header))
    for r in rows:
        print(",".join(str(x) for x in r))

    print(f"[analyze] Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
'''
# analyze_results.py
import ast, csv, os, math

OUT_CSV = "summary_table.csv"
TASK_ORDER = ["SST-2","SQuAD2.0","IWSLT2017-en-fr","RACE","MedMCQA"]

def safe_load(path):
    if not os.path.exists(path):
        return {t: float("nan") for t in TASK_ORDER}
    with open(path, "r") as f:
        txt = f.read().strip()
    try:
        d = ast.literal_eval(txt)
    except Exception:
        d = {t: float("nan") for t in TASK_ORDER}
    return {t: float(d.get(t, float("nan"))) for t in TASK_ORDER}

def avg(d):
    vals = [v for v in d.values() if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals)/len(vals) if vals else float("nan")

def main():
    recall = safe_load("recall_results.txt")
    he     = safe_load("he_recall_results.txt")  # 讀高熵版

    rows = []
    rows.append(["RECALL"] + [recall[t] for t in TASK_ORDER] + [avg(recall)])
    rows.append(["HE-RECALL(ours)"] + [he[t] for t in TASK_ORDER] + [avg(he)])

    header = ["model"] + TASK_ORDER + ["avg"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    print("=== SUMMARY ===")
    print(",".join(header))
    for r in rows:
        print(",".join(str(x) for x in r))
    print(f"[analyze] Wrote {OUT_CSV}")

if __name__ == "__main__":
    main()
