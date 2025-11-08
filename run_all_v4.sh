#!/bin/bash
set -e

# ===== 日誌（同時印到終端＆寫檔）=====
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOGFILE="run_${TIME_TAG}.log"
exec > >(tee ${LOGFILE}) 2>&1

echo "=== RECALL pipeline start @ ${TIME_TAG} ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BASE="Qwen/Qwen2-7B-Instruct"
TASKS=(SST-2 SQuAD2.0 IWSLT2017-en-fr RACE MedMCQA)

# -------------------------------
# Step 0. 準備母集合
# -------------------------------
echo "[Step0] prepare raw samples"
python prepare_samples.py

# -------------------------------
# Step 1. 逐任務 SFT（輸出到 ./adapters/<TASK>）
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
# Step 1.5 用各任務 adapter 對自家樣本做 K-means→各取20→串成同一份代表集
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
# Step 2. 抽各模型在「同一份代表集」上的逐層表示 & 熵（有 attention_mask 加權）
# 會產生 ./reprs/<TASK>_repr.pt 及 <TASK>_entropy.pt（每個任務各1份）
# -------------------------------
# echo "[Step2] extract representations & entropy for ALL models on SAME typcials"
# mkdir -p reprs
# for T in "${TASKS[@]}"; do
#   python extract_representations.py \
#    --base_model "$BASE" \
#    --adapter "./adapters/$T" \
#    --task "$T" \
#    --sample_file "./samples/typicals_all.txt" \
#    --out_dir ./reprs \
#    --max_samples 100 \
#    --max_len 256
#done

# ----- Step 2: 針對「同一份」typicals_all 抽各任務的逐層表示 -----
mkdir -p reprs
rm -f reprs/*_repr.pt  # 建議清掉舊檔，避免讀到舊格式
for T in SST-2 SQuAD2.0 IWSLT2017-en-fr RACE MedMCQA; do
  python extract_representations.py \
    --base_model Qwen/Qwen2-7B-Instruct \
    --adapter ./adapters/$T \
    --task $T \
    --sample_file ./samples/typicals_all.txt \
    --out_dir ./reprs \
    --max_samples 100 \
    --max_len 256
done


# -------------------------------
# Step 3. RECALL merge（RBF + softmax；anchor 可換 MedMCQA 試試）
# -------------------------------
echo "[Step3] merge: RECALL (representation-aligned)"
python merge_recall.py \
  --base_model "$BASE" \
  --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
  --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
  --anchor_task SQuAD2.0 \
  --out_dir ./fused_recall

# -------------------------------
# Step 4. HE-RECALL merge（高熵版；僅影響高層、熵做層內正規化）
# 需要 merge_he_recall.py（我給你的改良版）
# -------------------------------
#echo "[Step4] merge: HE-RECALL (high-entropy + representation)"
#python merge_he_recall.py \
#  --base_model Qwen/Qwen2-7B-Instruct \
#  --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
#  --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
#  --anchor_task SQuAD2.0 \
#  --alpha 0.4 \
#  --top_layer_ratio 0.35 \
#  --out_dir ./fused_he_recall

echo "[Step4] merge: eRECALL (entropy-aware)"
python merge_erecall.py \
  --base_model $BASE \
  --adapters ./adapters/SST-2 ./adapters/SQuAD2.0 ./adapters/IWSLT2017-en-fr ./adapters/RACE ./adapters/MedMCQA \
  --reprs ./reprs/SST-2_repr.pt ./reprs/SQuAD2.0_repr.pt ./reprs/IWSLT2017-en-fr_repr.pt ./reprs/RACE_repr.pt ./reprs/MedMCQA_repr.pt \
  --entropy_file ./reprs/SST-2_entropy.pt \
  --out_dir ./fused_erecall

# -------------------------------
# Step 5. 評估（同時印到終端＆寫檔）
# -------------------------------
echo "[Step5] eval RECALL"
python evaluate_all_tasks.py --model ./fused_recall --base_model "$BASE" | tee recall_results.txt

echo "[Step5b] eval HE-RECALL"
python evaluate_all_tasks.py --model ./fused_he_recall --base_model "$BASE" | tee he_recall_results.txt

# -------------------------------
# Step 6. 統整（把兩組結果都寫進 summary_table.csv）
# -------------------------------
echo "[Step6] summarize"
# 這支 analyze_results.py 目前只吃 recall/erecall 兩欄，
# 你可以複製一份或快速改一下讓它能讀 he_recall_results.txt；這裡先做個兼容：
# cp he_recall_results.txt erecall_results.txt
python analyze_results.py

echo "Done. Check summary_table.csv and ${LOGFILE}"
