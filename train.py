import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from datamodule.dataset import HMEDataset, collate_fn
from utils import build_vocab_from_dict, cnt_file, save_checkpoint, build_model

import warnings
warnings.filterwarnings('ignore')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to the config file", default='./config.yaml')

    ap.add_argument("--dataset", type=str,
                    help="Choose dataset [CROHME, HME100k]")
    ap.add_argument("--val_split", type=float, default=0.1,
                    help="Validation split ratio from training data")
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=5506)

    # model
    ap.add_argument("--use_lora", type=bool, default=False)
    ap.add_argument("--llm_name", type=str, default="gpt2")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--qformer_dim", type=int, default=256)
    ap.add_argument("--qformer_heads", type=int, default=8)
    ap.add_argument("--qformer_layers", type=int, default=4)
    ap.add_argument("--num_queries", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=list, default=None)
    ap.add_argument("--use_8bit", type=bool, default=True)

    args = ap.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        ap.set_defaults(**config)
        args = ap.parse_args()

    torch.manual_seed(args.seed)

    save_dir = Path("checkpoints/")
    ver = cnt_file(save_dir)+1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset.lower() == "crohme":
        train_path = Path("data/CROHME/train")
        dict_path = Path("data/CROHME/dictionary.txt")
    elif args.dataset.lower() == "hme100k":
        train_path = Path("data/HME100k/train")
        dict_path = Path("data/HME100k/dictionary.txt")
    else:
        raise ValueError(
            f"Unknown dataset {args.dataset}, choose [CROHME, HME100k] only.")

    # Build vocab
    vocab_obj = build_vocab_from_dict(dict_path)

    token_to_id, sp = vocab_obj["token_to_id"], vocab_obj["special"]
    pad_id = token_to_id[sp["pad"]]

    # Datasets
    full_train_ds = HMEDataset(source_path=train_path, vocab_obj=vocab_obj)

    dataset_size = len(full_train_ds)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size

    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(
        f"[INFO] Dataset split: {train_size} training samples, {val_size} validation samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, full_train_ds.pad_id),
        num_workers=args.num_workers,
        drop_last=False
    )

    val_loader = DataLoader(val_ds,
                            batch_size=args.batch,
                            shuffle=False,
                            collate_fn=lambda b: collate_fn(
                                b, full_train_ds.pad_id),
                            num_workers=args.num_workers
                            )

    # Model
    V = len(token_to_id)
    model = build_model(args, V)

    # Optim
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        pbar_train = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for batch in pbar_train:
            images = batch["images"].to(device)
            tgt = batch["target_ids"].to(device)
            logits = model(images, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             tgt[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += loss.item()
            pbar_train.set_postfix(loss=f"{loss.item():.4f}")
        tr = tot / max(1, len(train_loader))
        pbar_train.close()

        # Validation
        model.eval()
        vloss, n = 0.0, 0
        with torch.no_grad():
            pbar_val = tqdm(
                val_loader, desc=f"Epoch {epoch}/{args.epochs} [Valid]", leave=False)
            for batch in pbar_val:
                images = batch["images"].to(device)
                tgt = batch["target_ids"].to(device)
                logits = model(images, tgt[:, :-1])
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
                vloss += loss.item()
                n += 1
                pbar_val.set_postfix(loss=f"{loss.item():.4f}")
            pbar_val.close()

        v = vloss / max(1, n)
        print(f"[INFO] Epoch {epoch:03d} | train {tr:.4f} | val {v:.4f}")

        if v < best_val:
            best_val = v
            model_version = "lora" if args.use_lora else "no_lora"
            path = save_dir / \
                f"run{ver}_{args.dataset.lower()}_{model_version}.pt"
            save_checkpoint(model, args, token_to_id, sp, vocab_obj, path)
            # torch.save({"model": model.state_dict(), "vocab": token_to_id, "special": sp}, path)
            print(f"[INFO] Saved best model to: {path}")


if __name__ == "__main__":
    main()
