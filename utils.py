import os
from pathlib import Path
import argparse
import torch
from models.model import HMER_BLIP2, HMER_BLIP2_LoRA

# Data process
SPECIALS = {
    "pad": "<pad>",
    "bos": "<bos>",
    "eos": "<eos>",
    "unk": "<unk>"
}


def read_dictionary_txt(path: Path):
    return [ln.strip() for ln in Path(path).read_text(encoding='utf-8').splitlines() if ln.strip()]


def build_vocab_from_dict(dict_path: Path):
    toks, seen = [], set()
    for sp in (SPECIALS["pad"], SPECIALS["bos"], SPECIALS["eos"], SPECIALS["unk"]):
        if sp not in seen:
            seen.add(sp)
            toks.append(sp)
    for t in read_dictionary_txt(dict_path):
        if t not in seen:
            seen.add(t)
            toks.append(t)
    return {"token_to_id": {t: i for i, t in enumerate(toks)}, "special": SPECIALS}


# Build model
def build_model(args, vocab_size):
    use_lora = args.use_lora

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_8bit = args.use_8bit if device == "cuda" else False

    if use_lora:
        model = HMER_BLIP2_LoRA(
            d_model=args.d_model,
            latex_vocab_size=vocab_size,
            qformer_dim=args.qformer_dim,
            qformer_heads=args.qformer_heads,
            qformer_layers=args.qformer_layers,
            num_queries=args.num_queries,
            llm_name=args.llm_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            use_8bit=use_8bit,
        )
    else:
        model = HMER_BLIP2(
            latex_vocab_size=vocab_size,
            d_model=args.d_model,
            qformer_dim=args.qformer_dim,
            qformer_heads=args.qformer_heads,
            qformer_layers=args.qformer_layers,
            num_queries=args.num_queries,
            llm_name=args.llm_name,
            use_8bit=use_8bit
        )
    return model.to(device)


# Save model
def cnt_file(dir):
    files = [f for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    return len(files)


def save_checkpoint(model, args, token_to_id, sp, vocab_obj, path):
    torch.save({
        "model": model.state_dict(),
        "token_to_id": token_to_id,
        "special": sp,
        "vocab_obj": vocab_obj,
        "args": vars(args)
    }, path)

# Load model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    args = argparse.Namespace(**ckpt["args"])
    model = build_model(args, len(ckpt["token_to_id"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, args, ckpt["token_to_id"], ckpt["special"], ckpt["vocab_obj"]
