import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from decode import beam_search, set_constraints_vocab
from utils import load_checkpoint
from datamodule.dataset import HMEDataset, resize_keep_ratio, collate_fn

import warnings
warnings.filterwarnings('ignore')


def strip_special(ids: List[int], bos: int, eos: int, pad: int) -> List[int]:
    """Removes special tokens (BOS, EOS, PAD) from a sequence of token IDs."""
    out = []
    for i in ids:
        if i in (bos, pad):
            continue
        if i == eos:
            break
        out.append(i)
    return out


def levenshtein(a: List[int], b: List[int]) -> int:
    """Calculates the Levenshtein distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[m]


def count_ok(preds: List[List[int]], refs: List[List[int]], k: int) -> int:
    """Counts how many predictions have a Levenshtein distance of k or less."""
    return sum(1 for p, r in zip(preds, refs) if levenshtein(p, r) <= k)


def run_evaluation(
    model: torch.nn.Module,
    device: str,
    data_loader: DataLoader,
    args: argparse.Namespace,
    token_to_id: Dict,
    pad: int, bos: int, eos: int, unk: int
):
    """
    Runs the full evaluation loop for a given dataset, processing data in batches.
    """
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {Path(data_loader.dataset.source_path).name}"):
            images = batch["images"].to(device)
            ref_tokens_list = batch["tokens"]

            visual_seq = model.image_encoder(images)
            q_feats = model.qformer(visual_seq)
            llm_input = model.to_llm(q_feats)

            for i in range(llm_input.size(0)):
                llm_input_single = llm_input[i:i+1]
                ref_tokens = ref_tokens_list[i]

                seq = beam_search(model, llm_input_single, bos,
                                  eos, pad, beam=args.beam, max_len=args.max_len)

                pred_ids = strip_special(seq.tolist(), bos, eos, pad)
                preds.append(pred_ids)

                ref_ids = [token_to_id.get(t, unk) for t in ref_tokens]
                refs.append(ref_ids)

    N = len(refs)
    if N == 0:
        print("No samples were evaluated.")
        return

    exact = count_ok(preds, refs, 0)
    le1 = count_ok(preds, refs, 1)
    le2 = count_ok(preds, refs, 2)
    print("-" * 30)
    print(f"Results for {Path(data_loader.dataset.source_path).name}:")
    print(f"Samples: {N}")
    print(f"ExpRate (exact match): {exact/N:.4f}")
    print(f"ExpRate<=1 (≤1 error):  {le1/N:.4f}")
    print(f"ExpRate<=2 (≤2 errors): {le2/N:.4f}")
    print("-" * 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to the model checkpoint.")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Batch size for evaluation.")
    ap.add_argument("--num_workers", type=int, default=2,
                    help="Number of workers for DataLoader.")
    ap.add_argument("--beam", type=int, default=5,
                    help="Beam size for beam search decoding.")
    ap.add_argument("--max_len", type=int, default=200,
                    help="Maximum length of decoded sequence.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set up model and load checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, model_args, token_to_id, special, vocab_obj = load_checkpoint(
        args.checkpoint, device)
    model.eval()

    pad = token_to_id[special["pad"]]
    bos = token_to_id[special["bos"]]
    eos = token_to_id[special["eos"]]
    unk = token_to_id[special["unk"]]

    set_constraints_vocab(token_to_id)

    dataset_name = model_args.dataset.lower()
    print(
        f"[INFO] Model was trained on '{dataset_name}'. Evaluating corresponding test set(s).")

    if dataset_name == "crohme":
        splits = ["2014", "2016", "2019"]
        for year in splits:
            print("\n" + "="*50)
            eval_path = Path(f"data/CROHME/{year}")
            try:
                eval_ds = HMEDataset(
                    source_path=eval_path, vocab_obj=vocab_obj)
                eval_loader = DataLoader(
                    eval_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=lambda b: collate_fn(b, eval_ds.pad_id),
                    num_workers=args.num_workers
                )
                run_evaluation(model, device, eval_loader, args,
                               token_to_id, pad, bos, eos, unk)

            except FileNotFoundError:
                print(f"Warning: Dataset not found at {eval_path}. Skipping.")

    elif dataset_name == "hme100k":
        print("\n" + "="*50)
        eval_path = Path("data/HME100k/test")
        try:
            eval_ds = HMEDataset(source_path=eval_path, vocab_obj=vocab_obj)
            eval_loader = DataLoader(
                eval_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, eval_ds.pad_id),
                num_workers=args.num_workers
            )
            run_evaluation(model, device, eval_loader, args,
                           token_to_id, pad, bos, eos, unk)
        except FileNotFoundError:
            print(f"Warning: Dataset not found at {eval_path}. Skipping.")
    else:
        print(
            f"Error: Dataset '{model_args.dataset}' from checkpoint is not recognized.")


if __name__ == "__main__":
    main()
