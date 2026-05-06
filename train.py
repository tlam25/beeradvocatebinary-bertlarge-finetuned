import os
import json
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from huggingface_hub import hf_hub_download, HfApi
from dotenv import load_dotenv
from tqdm import tqdm


# =========================
# Config
# =========================
MODEL_NAME = "bert-large-uncased"
ASPECTS    = ["appearance", "aroma", "palate", "taste"]
SEEDS      = [2025, 42, 123, 7, 999]

HF_REPO_ID   = "tlam25/BeerAdvocateBinary-BERT-Large-checkpoints"
HF_REPO_TYPE = "dataset"

RESULTS_DIR = "./results"
CKPT_DIR    = "./checkpoints"

MAX_LEN            = 256
TRAIN_BATCH_SIZE   = 4
EVAL_BATCH_SIZE    = 32
ACCUM_STEPS        = 4         # effective batch size = TRAIN_BATCH_SIZE * ACCUM_STEPS
NUM_EPOCHS         = 3
PATIENCE           = 2         # early stopping
LR_BERT            = 2e-5
LR_HEAD            = 3e-4
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.1
CLASSIFIER_DROPOUT = 0.3
NUM_LABELS         = 2

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["negative", "positive"]


# =========================
# HuggingFace push (optional)
# =========================
# Initialised in main() after .env is loaded. If repo/token aren't configured,
# these stay None and the push step is silently skipped.
_HF_API           = None    # HfApi instance (or None)
_HF_MODEL_REPO    = None    # "username/repo" (or None)
_DELETE_AFTER_PUSH = False  # set by --delete-local-ckpt


def push_checkpoint_to_hf(ckpt_path: str, seed: int, aspect: str):
    """Upload a single checkpoint to HF Hub. Failures are non-fatal.
    If the repo went missing between init and now (e.g. deleted from web UI),
    try to recreate it once and retry."""
    if _HF_API is None or _HF_MODEL_REPO is None:
        return
    path_in_repo = f"seed_{seed}/best_bert_{aspect}.pt"

    def _do_upload():
        _HF_API.upload_file(
            path_or_fileobj=ckpt_path,
            path_in_repo=path_in_repo,
            repo_id=_HF_MODEL_REPO,
            repo_type="model",
            commit_message=f"add checkpoint seed={seed} aspect={aspect}",
        )

    try:
        print(f"  uploading to HF: {_HF_MODEL_REPO}/{path_in_repo} ...")
        try:
            _do_upload()
        except Exception as e:
            # Common case: repo was deleted on the web UI mid-run, or the
            # initial create_repo silently no-op'd because of HF caching.
            msg = str(e).lower()
            if "not found" in msg or "404" in msg:
                print(f"  repo missing, recreating {_HF_MODEL_REPO} and retrying...")
                _HF_API.create_repo(repo_id=_HF_MODEL_REPO, repo_type="model",
                                    exist_ok=True, private=True)
                _do_upload()
            else:
                raise

        print(f"  uploaded -> https://huggingface.co/{_HF_MODEL_REPO}/blob/main/{path_in_repo}")
        if _DELETE_AFTER_PUSH:
            try:
                os.remove(ckpt_path)
                print(f"  removed local {ckpt_path}")
            except OSError as e:
                print(f"  could not remove local file: {e}")
    except Exception as e:
        print(f"  WARNING: HF upload failed ({type(e).__name__}): {e}")


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_csv_path(aspect: str, split: str) -> str:
    """Download (or get cached) CSV file from HuggingFace Hub."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=f"{aspect}/{aspect}_{split}_binary.csv",
    )


def prefetch_all(aspects, splits=("train", "dev", "test")):
    print("Pre-fetching all CSV files from HuggingFace Hub...")
    for aspect in aspects:
        for split in splits:
            path = get_csv_path(aspect, split)
            print(f"  cached: {aspect}/{split}  ->  {path}")
    print("All files cached.\n")


def create_weighted_sampler(labels):
    class_counts = Counter(labels)
    print("  Train label counts:", dict(sorted(class_counts.items())))
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = torch.DoubleTensor([class_weights[l] for l in labels])
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# =========================
# Dataset
# =========================
class BeerAdvocateDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int, aspect: str):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        label_col = f"{aspect}_binary_label"
        df = df.dropna(subset=["text", label_col])

        df["text"] = df["text"].astype(str)
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)

        invalid = (~df[label_col].isin([0, 1])).sum()
        assert invalid == 0, f"{invalid} rows have invalid labels in {csv_path}!"

        self.texts     = df["text"].tolist()
        self.labels    = df[label_col].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# =========================
# Model
# =========================
class BertClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, classifier_dropout: float = 0.3):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    @staticmethod
    def masked_mean_pooling(sequence_output, attention_mask):
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (sequence_output * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.masked_mean_pooling(out.last_hidden_state, attention_mask)
        return self.classifier(self.dropout(pooled))


def build_optimizer(model, lr_bert, lr_head, weight_decay):
    no_decay = {"bias", "LayerNorm.weight"}
    param_groups = [
        # BERT params WITH weight decay
        {
            "params": [p for n, p in model.named_parameters()
                       if n.startswith("bert.") and not any(nd in n for nd in no_decay)],
            "lr": lr_bert,
            "weight_decay": weight_decay,
        },
        # BERT params WITHOUT weight decay (bias, LayerNorm)
        {
            "params": [p for n, p in model.named_parameters()
                       if n.startswith("bert.") and any(nd in n for nd in no_decay)],
            "lr": lr_bert,
            "weight_decay": 0.0,
        },
        # Classification head
        {
            "params": [p for n, p in model.named_parameters()
                       if not n.startswith("bert.")],
            "lr": lr_head,
            "weight_decay": weight_decay,
        },
    ]
    return torch.optim.AdamW(param_groups)


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, accum_steps):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(logits, labels) / accum_steps  # scale loss
        loss.backward()

        # Step every accum_steps batches (or at the end of the epoch)
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps  # un-scale for logging
        all_preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Return accuracy and macro precision / recall / F1."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits      = model(input_ids=input_ids, attention_mask=attention_mask)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {
        "loss":      total_loss / len(loader),
        "accuracy":  acc,
        "precision": p,   # macro
        "recall":    r,   # macro
        "f1":        f,   # macro
        "y_true":    all_labels,
        "y_pred":    all_preds,
    }


# =========================
# Run one (seed, aspect)
# =========================
def run_one(aspect: str, seed: int, tokenizer):
    print(f"\n{'='*60}")
    print(f"Seed = {seed} | Aspect = {aspect.upper()}")
    print(f"{'='*60}")

    set_seed(seed)

    # ---- Datasets / Loaders ----
    def load(split):
        return BeerAdvocateDataset(get_csv_path(aspect, split), tokenizer, MAX_LEN, aspect)

    train_ds = load("train")
    val_ds   = load("dev")
    test_ds  = load("test")

    train_sampler = create_weighted_sampler(train_ds.labels)
    train_loader  = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE,
                               sampler=train_sampler, shuffle=False,
                               num_workers=2, pin_memory=True)
    val_loader    = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)
    test_loader   = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)

    # ---- Model ----
    model = BertClassifier(MODEL_NAME, NUM_LABELS, CLASSIFIER_DROPOUT).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, LR_BERT, LR_HEAD, WEIGHT_DECAY)

    effective_steps_per_epoch = (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS
    total_steps  = effective_steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ---- Train loop with early stopping on macro-F1 ----
    ckpt_path = os.path.join(CKPT_DIR, f"seed_{seed}", f"best_bert_{aspect}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    best_val_f1      = -1.0
    patience_counter = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE, ACCUM_STEPS
        )
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train | loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"Val   | loss: {val_metrics['loss']:.4f} | acc: {val_metrics['accuracy']:.4f} "
              f"| macro_P: {val_metrics['precision']:.4f} "
              f"| macro_R: {val_metrics['recall']:.4f} "
              f"| macro_F1: {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1      = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved best -> {ckpt_path}  (val_macro_F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"  early stopping at epoch {epoch + 1}")
                break

    # ---- Test ----
    print(f"\n--- Test [seed={seed} | aspect={aspect}] ---")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Test  | loss: {test_metrics['loss']:.4f} "
          f"| acc: {test_metrics['accuracy']:.4f} "
          f"| macro_P: {test_metrics['precision']:.4f} "
          f"| macro_R: {test_metrics['recall']:.4f} "
          f"| macro_F1: {test_metrics['f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        labels=list(range(NUM_LABELS)),
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    ))

    # ---- Save JSON immediately (so we don't lose progress if job dies) ----
    out = {
        "seed":        seed,
        "aspect":      aspect,
        "best_val_f1": best_val_f1,
        "test": {
            "accuracy":  test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall":    test_metrics["recall"],
            "f1":        test_metrics["f1"],
            "loss":      test_metrics["loss"],
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"seed_{seed}_{aspect}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")

    # ---- Optional: push checkpoint to HuggingFace Hub ----
    push_checkpoint_to_hf(ckpt_path, seed, aspect)

    # ---- Free VRAM before the next aspect/seed ----
    del model, optimizer, scheduler, criterion
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return out


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="List of seeds to run (default: all 5)")
    parser.add_argument("--aspects", nargs="+", default=ASPECTS,
                        help="List of aspects to run (default: all 4)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run even if a result JSON already exists")
    parser.add_argument("--delete-local-ckpt", action="store_true",
                        help="Delete local checkpoint after successful HF upload (saves disk)")
    args = parser.parse_args()

    # ---- Load .env (HF_TOKEN, HF_MODEL_REPO) ----
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo  = os.environ.get("HF_MODEL_REPO")

    global _HF_API, _HF_MODEL_REPO, _DELETE_AFTER_PUSH
    if hf_token and hf_repo:
        _HF_API        = HfApi(token=hf_token)
        _HF_MODEL_REPO = hf_repo
        _DELETE_AFTER_PUSH = args.delete_local_ckpt

        # Sanity: token role + namespace match
        try:
            me = _HF_API.whoami()
            print(f"HF authenticated as: {me['name']}")
            expected_owner = hf_repo.split("/")[0]
            if me["name"] != expected_owner and expected_owner not in [o["name"] for o in me.get("orgs", [])]:
                print(f"  WARNING: token user is '{me['name']}' but HF_MODEL_REPO owner is '{expected_owner}'.")
                print(f"  You probably want HF_MODEL_REPO={me['name']}/{hf_repo.split('/', 1)[1]}")
        except Exception as e:
            print(f"  WARNING: whoami() failed: {e}")

        # Create the repo (idempotent). If this fails, push will be disabled.
        try:
            url = _HF_API.create_repo(repo_id=hf_repo, repo_type="model",
                                    exist_ok=True, private=True)
            print(f"HF push enabled -> {hf_repo} (repo at {url}; delete_after_push={_DELETE_AFTER_PUSH})")
        except Exception as e:
            print(f"FATAL: cannot create/access repo {hf_repo}: {type(e).__name__}: {e}")
            print("Disabling HF push for this run. Fix .env / token and re-run.")
            _HF_API = None
            _HF_MODEL_REPO = None
    else:
        missing = [v for v in ("HF_TOKEN", "HF_MODEL_REPO") if not os.environ.get(v)]
        print(f"HF push DISABLED (missing in .env: {missing})")

    print(f"Device: {DEVICE}")
    print(f"Seeds:   {args.seeds}")
    print(f"Aspects: {args.aspects}")

    prefetch_all(args.aspects)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for seed in args.seeds:
        for aspect in args.aspects:
            out_path = os.path.join(RESULTS_DIR, f"seed_{seed}_{aspect}.json")
            if os.path.exists(out_path) and not args.no_resume:
                print(f"[skip] {out_path} already exists (use --no-resume to overwrite)")
                continue
            run_one(aspect, seed, tokenizer)

    print("\nAll runs complete. Run `python analyze_results.py` to aggregate.")


if __name__ == "__main__":
    main()