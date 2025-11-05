import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from merge_recall import find_transformer_blocks  # 如果 merge_recall.py 在同資料夾

base_model_name = "Qwen/Qwen2-7B-Instruct"
adapter_path = "./adapters/SST-2"

base = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"":0},
)
peft_m = PeftModel.from_pretrained(base, adapter_path)
peft_m.eval()

blocks = find_transformer_blocks(peft_m)
print("len(blocks) =", len(blocks))
print("type(blocks[0]) =", type(blocks[0]))
