from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

model_id = "Qwen/Qwen2-7B-Instruct"
local_dir = "./qwen2-7b-base"

print("Downloading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_id)
tok.save_pretrained(local_dir)

print("Downloading config/model skeleton...")
cfg = AutoConfig.from_pretrained(model_id)
cfg.save_pretrained(local_dir)

# 這一步會下載原始權重，很大，但是我們主要目的是把它存成一個合法HF資料夾
print("Downloading model weights...")
mdl = AutoModelForCausalLM.from_pretrained(model_id)
mdl.save_pretrained(local_dir)

print("Done.")
