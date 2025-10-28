#!/bin/bash

set -e  # 有任何指令出錯就停

# ===== 日誌檔帶時間戳 =====
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOGFILE="run_${TIME_TAG}.log"
exec > >(tee ${LOGFILE}) 2>&1

echo "=== RECALL pipeline start @ ${TIME_TAG} ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ===== 你要用的 base model =====
BASE="Qwen/Qwen2-7B-Instruct"
# 如果你有 Llama-2-7b-chat-hf 權限可以改成：
# BASE="meta-llama/Llama-2-7b-chat-hf"

# ===== Step 0. 準備代表樣本（typical samples）=====
echo "[Step0] prepare samples"
python prepare_samples.py

# ===== Step 1. SFT：對每個任務做 QLoRA 微調，輸出到 ./adapters/<TASK> =====
for T in SST-2 SQuAD2.0 IWSLT2017-en-fr RACE MedMCQA; do
  echo "[Step1] SFT task $T"
  python train_sft_task.py \
    --base_model $BASE \
    --task $T \
    --rank 16 \
    --lr 2e-4 \
    --steps 300
done

# ===== Step 2. 抽表徵 + 熵 =====
for T in SST-2 SQuAD2.0 IWSLT2017-en-fr RACE MedMCQA; do
  echo "[Step2] Extract repr for $T"
  python extract_representations.py \
    --base_model $BASE \
    --adapter ./adapters/$T \
    --task $T \
    --sample_file ./samples/${T}.txt \
    --out_dir ./reprs \
    --max_samples 128 \
    --entropy_select \
    --top_ratio 0.3
done

# ===== Step 3. RECALL merge (baseline) =====
echo "[Step3] merge: RECALL baseline"
python merge_recall.py \
  --base_model $BASE \
  --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
  --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
  --out_dir ./fused_recall

# ===== (Optional) Step 4. eRECALL merge (entropy-aware) =====
# 先暫時關掉，等 merge_erecall.py 也改成省顯存+熵加權版再打開
# echo "[Step4] merge: eRECALL (entropy-aware)"
# python merge_erecall.py \
#   --base_model $BASE \
#   --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
#   --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
#   --entropy_file ./reprs/SST-2_entropy.pt \
#   --out_dir ./fused_erecall

# ===== Step 5. 評估 =====
echo "[Step5] eval RECALL"
python evaluate_all_tasks.py --model ./fused_recall > recall_results.txt

# 如果你之後打開 eRECALL，就在這邊多跑一行：
# echo "[Step5b] eval eRECALL"
# python evaluate_all_tasks.py --model ./fused_erecall > erecall_results.txt

# ===== Step 6. 統整表格 =====
echo "[Step6] summarize"
python analyze_results.py
echo "Done. Check summary_table.csv and ${LOGFILE}"
