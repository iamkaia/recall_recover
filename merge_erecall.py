# merge_erecall.py
import torch, argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F

def weighted_layer_sim(reprs, entropy):
    sims = []
    n_layers = len(reprs[0]["reprs"])
    for l in range(n_layers):
        layer_sims = []
        for i in range(1, len(reprs)):
            a, b = reprs[0]["reprs"][l], reprs[i]["reprs"][l]
            sim = F.cosine_similarity(a, b, dim=-1)
            w = torch.tensor(entropy[:len(sim)])  # same sample len
            sim = (sim * w).sum() / w.sum()
            layer_sims.append(sim.item())
        sims.append(layer_sims)
    return sims

def merge_entropy(models, reprs, entropy):
    fused = models[0]
    sims = weighted_layer_sim(reprs, entropy)
    with torch.no_grad():
        for l, sim_list in enumerate(sims):
            weights = torch.tensor(sim_list)
            weights = weights / weights.sum()
            for name, p in fused.model.layers[l].named_parameters():
                stacked = torch.stack([m.model.layers[l].state_dict()[name] for m in models[1:]])
                fused.model.layers[l]._parameters[name].copy_((weights[:,None,None]*stacked).sum(0))
    return fused

def main(args):
    base = args.base_model
    models = [PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True, device_map="auto"), ad) for ad in args.adapters]
    reprs = [torch.load(r) for r in args.reprs]
    entropy = torch.load(args.entropy_file)["entropy"]
    fused = merge_entropy(models, reprs, entropy)
    fused.save_pretrained(args.out_dir)
    print(f"✅ eRECALL fused -> {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--reprs", nargs="+", required=True)
    ap.add_argument("--entropy_file", required=True)
    ap.add_argument("--out_dir", default="./fused_erecall")
    args = ap.parse_args()
    main(args)
