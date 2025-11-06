#!/bin/bash
set -e

# ===== 日誌 =====
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOGFILE="run_${TIME_TAG}.log"
exec > >(tee ${LOGFILE}) 2>&1

echo "=== RECALL pipeline start @ ${TIME_TAG} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BASE="Qwen/Qwen2-7B-Instruct"
TASKS=(SST-2 SQuAD2.0 IWSLT2017-en-fr RACE MedMCQA)

# -------------------------------
# Step 0. 準備母集合（原本的 samples/*.txt）
# -------------------------------
echo "[Step0] prepare raw samples"
python prepare_samples.py

# -------------------------------
# Step 1. 逐任務 SFT（你原本的流程）
# -------------------------------
for T in "${TASKS[@]}"; do
  echo "[Step1] SFT task $T"
  python train_sft_task.py \
    --base_model "$BASE" \
    --task "$T" \
    --rank 16 \
    --lr 2e-4 \
    --steps 300
done

# -------------------------------
# Step 1.5 典型樣本：用各任務的 adapter 各挑 20 筆，再串成一個代表集
# -------------------------------
echo "[Step1.5] select typcials (K-means) per task"
mkdir -p samples/typicals
> samples/typicals_all.txt
for T in "${TASKS[@]}"; do
  python select_typicals.py \
    --base_model "$BASE" \
    --adapter "./adapters/$T" \
    --in_txt "./samples/${T}.txt" \
    --out_txt "./samples/typicals/${T}.typical.txt" \
    --k 20
  cat "./samples/typicals/${T}.typical.txt" >> samples/typicals_all.txt
done
echo "✅ Built samples/typicals_all.txt"

# -------------------------------
# Step 2. 對「同一份」代表集抽各模型的逐層表示（用 mask 平均）
# -------------------------------
echo "[Step2] extract representations for ALL models on the SAME typcials"
mkdir -p reprs
for T in "${TASKS[@]}"; do
  python extract_representations.py \
    --base_model "$BASE" \
    --adapter "./adapters/$T" \
    --task "$T" \
    --sample_file "./samples/typicals_all.txt" \
    --out_dir ./reprs \
    --max_samples 100 \
    --max_len 256
done
# (可選) 熵只用來 eRECALL，就單獨算一份合併後再用
python extract_representations.py \
  --base_model "$BASE" \
  --adapter "./adapters/SST-2" \
  --task SST-2 \
  --sample_file "./samples/typicals_all.txt" \
  --out_dir ./reprs \
  --max_samples 100 \
  --max_len 256 \
  --entropy_select \
  --top_ratio 0.3

# -------------------------------
# Step 3. RECALL merge（RBF + softmax；anchor 指向你想當「代表任務」的那個）
# 建議先用 SQuAD2.0 或 MedMCQA 當 anchor，或新增一個「聯合代表」任務再挑典型
# -------------------------------
echo "[Step3] merge: RECALL"
python merge_recall.py \
  --base_model "$BASE" \
  --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
  --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
  --anchor_task SQuAD2.0 \
  --out_dir ./fused_recall

# -------------------------------
# Step 4. eRECALL (可先跳過；若要跑，entropy 要「對齊每個 repr」傳進去或在 repr blob 裡一起存）
# 暫時關閉：等你的 merge_erecall.py 支援多檔 entropy 或從 *_repr.pt 讀 entropy
# -------------------------------
# echo "[Step4] merge: eRECALL"
# python merge_erecall.py \
#   --base_model "$BASE" \
#   --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
#   --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
#   --entropy_files ./reprs/SST-2_entropy.pt ./reprs/SQuAD2.0_entropy.pt ./reprs/IWSLT2017-en-fr_entropy.pt ./reprs/RACE_entropy.pt ./reprs/MedMCQA_entropy.pt \
#   --out_dir ./fused_erecall

# -------------------------------
# Step 5. 評估
# -------------------------------
echo "[Step5] eval RECALL"
python evaluate_all_tasks.py --model ./fused_recall --base_model "$BASE" > recall_results.txt

# 若你已開 eRECALL，再加：
# echo "[Step5b] eval eRECALL"
# python evaluate_all_tasks.py --model ./fused_erecall --base_model "$BASE" > erecall_results.txt

# -------------------------------
# Step 6. 統整
# -------------------------------
echo "[Step6] summarize"
python analyze_results.py

echo "Done. Check summary_table.csv and ${LOGFILE}"
