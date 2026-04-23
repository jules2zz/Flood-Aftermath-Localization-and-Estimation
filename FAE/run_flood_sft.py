#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_flood_sft.py  ·  Single-GPU LoRA Fine-Tuning + Stage-Wise Evaluation
-----------------------------------------------------------------------
• Train for 36 epochs in total (3 epochs per stage)
• Run inference on the `FAE_test` dataset after each stage and report 6 metrics
• Sampling parameters: temperature=0.1, top_p=0.001
"""

import subprocess, datetime, json, re, sys, os, importlib.util
from pathlib import Path
from typing import List, Dict, Set
import numpy as np

# ---------------- Basic training arguments (official CLI key fields) ----------------
BASE_ARGS = dict(
    stage="sft",
    do_train="True",
    model_name_or_path="/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct",
    finetuning_type="lora",
    lora_target="all",
    dataset="FAE_train",
    template="qwen2_vl",
    val_size="0.15",
    cutoff_len="4096",
    preprocessing_num_workers="16",
    learning_rate="1e-5",
    per_device_train_batch_size="2",
    gradient_accumulation_steps="8",
    warmup_ratio="0.01",
    lr_scheduler_type="cosine",
    weight_decay="0.05",
    lora_rank="16",
    lora_alpha="32",
    lora_dropout="0.05",
    bf16="True",
    flash_attn="auto",
    freeze_vision_tower="True",
    freeze_multi_modal_projector="True",
    save_strategy="epoch",
    logging_steps="20",
    report_to="none",
    ddp_timeout="180000000",
    trust_remote_code="True"
)

TOTAL_EPOCHS       = 36
EPOCHS_PER_STAGE   = 3
EVAL_DATASET       = "FAE_test"
TEMPERATURE        = "0.1"
TOP_P              = "0.001"

stamp     = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_OUT  = Path(f"saves/Qwen2.5-VL-7B-Instruct/lora/flood_{stamp}")
BASE_OUT.mkdir(parents=True, exist_ok=True)
LOG_FILE  = BASE_OUT / "train.log"

def log(msg: str):
    print(msg)
    LOG_FILE.write_text(LOG_FILE.read_text() + msg + "\n" if LOG_FILE.exists() else msg + "\n")

# ========= 3. Utility functions =========
def run(cmd: List[str]) -> int:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    return proc.returncode

def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.open()]

def find_adapter_dir(base: Path) -> Path:
    """Recursively search for the directory containing adapter_config.json."""
    for root, _, files in os.walk(base):
        if "adapter_config.json" in files:
            return Path(root)
    return None

# ----- Label parsing -----
def labels_from_string(s: str) -> Set[str]:
    return {t.strip() for t in re.split(r"[,\s]+", s) if t.strip()}

def extract_labels(ex: Dict, kind: str) -> Set[str]:
    """
    kind = 'pred' | 'ref'
    Prediction files usually contain `predict` or `text`.
    Reference labels (ground truth) may appear in `label`,
    `assistant_labels`, or `labels`.
    """
    keys_pred = ("predict", "text")
    keys_ref  = ("label", "assistant_labels", "labels")
    keys_sharegpt = ("messages",)

    if kind == "pred":
        for k in keys_pred:
            if k in ex and ex[k]:
                return labels_from_string(ex[k])
    else:  # kind == 'ref'
        for k in keys_ref:
            if k in ex and ex[k]:
                return labels_from_string(ex[k])

    # ShareGPT-style fallback
    if "messages" in ex:
        for m in reversed(ex["messages"]):
            if m.get("role") == ("assistant" if kind == "ref" else "user"):
                return labels_from_string(m.get("content", ""))
    return set()

# ----- Metrics -----
BIG_KEYS = ["FD", "HU", "PD", "BD", "IN", "EC", "EN"]

def to_big(s: Set[str]) -> Set[str]:
    return {re.sub(r"\d$", "", x) for x in s}

def compute_metrics(refs: List[Set[str]], hyps: List[Set[str]]):
    TP = FP = FN = 0
    big_stats = {k: [0, 0, 0] for k in BIG_KEYS}  # tp, fp, fn
    for r, h in zip(refs, hyps):
        TP += len(r & h)
        FP += len(h - r)
        FN += len(r - h)
        br, bh = to_big(r), to_big(h)
        for k in BIG_KEYS:
            big_stats[k][0] += k in br and k in bh
            big_stats[k][1] += k not in br and k in bh
            big_stats[k][2] += k in br and k not in bh
    macro_p = np.mean([tp / (tp + fp + 1e-8) for tp, fp, _ in big_stats.values()])
    macro_r = np.mean([tp / (tp + fn + 1e-8) for tp, _, fn in big_stats.values()])
    macro_f = 2 * macro_p * macro_r / (macro_p + macro_r + 1e-8)
    micro_p = TP / (TP + FP + 1e-8)
    micro_r = TP / (TP + FN + 1e-8)
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-8)
    return macro_p, macro_r, macro_f, micro_p, micro_r, micro_f

# ========= 4. Training & evaluation loop =========
adapter_path = None
for stage in range(1, TOTAL_EPOCHS // EPOCHS_PER_STAGE + 1):
    cur_ep   = stage * EPOCHS_PER_STAGE
    out_base = BASE_OUT / f"epoch_{cur_ep}"
    log(f"\n=== Stage {stage} | Training Epochs {cur_ep - EPOCHS_PER_STAGE + 1}-{cur_ep} ===")

    # --- 4.1 Training ---
    train_cmd = ["llamafactory-cli", "train"]
    for k, v in BASE_ARGS.items():
        train_cmd += [f"--{k}", str(v)]
    train_cmd += [
        "--num_train_epochs", str(EPOCHS_PER_STAGE),
        "--output_dir", str(out_base)
    ]
    if adapter_path:
        train_cmd += ["--adapter_name_or_path", str(adapter_path)]

    if run(train_cmd) != 0:
        sys.exit("Training failed. Please check the log above.")

    # --- 4.2 Find the latest LoRA adapter directory ---
    adapter_path = find_adapter_dir(out_base)
    if adapter_path is None:
        sys.exit("adapter_config.json was not found after training.")

    # --- 4.3 Inference ---
    pred_dir = out_base / "predict_test"
    pred_cmd = [
        "llamafactory-cli", "train",
        "--stage", "sft", "--do_predict",
        "--model_name_or_path", BASE_ARGS["model_name_or_path"],
        "--adapter_name_or_path", str(adapter_path),
        "--template", BASE_ARGS["template"],
        "--eval_dataset", EVAL_DATASET,
        "--output_dir", str(pred_dir),
        "--predict_with_generate", "true",
        "--temperature", TEMPERATURE, "--top_p", TOP_P,
        "--do_sample", "true",
        "--trust_remote_code"
    ]
    if run(pred_cmd) != 0:
        sys.exit("Prediction failed.")

    pred_file = pred_dir / "generated_predictions.jsonl"
    if not pred_file.exists():
        sys.exit("generated_predictions.jsonl is missing.")

    # --- 4.4 Evaluation ---
    preds = load_jsonl(pred_file)

    # If the prediction file already contains label fields, use them directly as references;
    # otherwise, load the external test_15.jsonl file.
    if all("label" in x for x in preds):
        y_true = [extract_labels(x, "ref") for x in preds]
    else:
        refs   = load_jsonl(Path("data") / "test_15.jsonl")
        y_true = [extract_labels(x, "ref") for x in refs]

    y_pred = [extract_labels(x, "pred") for x in preds]

    mP, mR, mF, miP, miR, miF = compute_metrics(y_true, y_pred)
    log(
        f"Metrics @epoch{cur_ep} — "
        f"Macro-P 7cls={mP:.3f} | Macro-R 7cls={mR:.3f} | Macro-F1 7cls={mF:.3f} || "
        f"Micro-P 21cls={miP:.3f} | Micro-R 21cls={miR:.3f} | Micro-F1 21cls={miF:.3f}"
    )

log("\nAll stages finished successfully.")
