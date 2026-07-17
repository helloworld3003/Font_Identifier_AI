"""
evaluate_accuracy.py
--------------------
Evaluates font-identification accuracy on a random sample of fonts.
GPU is kept busy by using a DataLoader with multiple CPU workers to
pipeline font rendering (CPU) with model inference (GPU).

Memory strategy
---------------
- PIL/numpy arrays are NOT accumulated across all samples.
- Each batch saves its test images to disk immediately.
- Only the first VISUAL_SAMPLES PIL arrays are kept in RAM for the
  contact sheet; the rest are discarded after saving.
"""

import os
import math
import random
import string
import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont as PILImageFont
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from train_virtual_epochs import ConvNeXtFontEncoder
from inference import preprocess_inference_crop

# ── Config ──────────────────────────────────────────────────────────────────
EMBEDDING_SIZE = 256
MODEL_PATH     = "best_model.pth"
INDEX_PATH     = "font_embeddings.index"
MAPPING_PATH   = "faiss_mapping.csv"
OUTPUT_DIR     = "eval_results"

# DataLoader tuning
BATCH_SIZE  = 128   # keep lower on Windows (shared-memory limit)
NUM_WORKERS = 2     # 2 workers is enough to overlap rendering with GPU

# Contact sheet: first N samples shown in the visual grid
VISUAL_SAMPLES = 100
CELL_W, CELL_H = 280, 100
COLS           = 5
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ──────────────────────────────────────────────────────────────────
def _random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def _render_font(ttf_path: str, text: str, font_size: int, canvas: int = 224):
    """Render *text* in *ttf_path* and return a (H,W,3) uint8 numpy array."""
    try:
        fnt = PILImageFont.truetype(ttf_path, font_size)
        img = Image.new("RGB", (canvas, canvas), "white")
        drw = ImageDraw.Draw(img)
        bb  = drw.textbbox((0, 0), text, font=fnt)
        x   = (canvas - (bb[2] - bb[0])) / 2
        y   = (canvas - (bb[3] - bb[1])) / 2
        drw.text((x, y), text, font=fnt, fill="black")
        return np.array(img)
    except Exception:
        return np.ones((canvas, canvas, 3), dtype=np.uint8) * 255


class FontEvalDataset(Dataset):
    """
    __getitem__ returns (tensor, faiss_id, font_name, test_word, raw_np).
    PIL rendering runs inside worker processes so the GPU stays fed.
    Only a plain numpy array is returned (no PIL objects) to avoid
    pickling issues across Windows process boundaries.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.df = df.reset_index(drop=True)
        rng = random.Random(seed)
        self.words = [_random_string(rng.randint(4, 7)) for _ in range(len(df))]
        self.sizes = [rng.randint(45, 75)               for _ in range(len(df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        faiss_id  = int(row["faiss_id"])
        font_path = str(row["font_path"])
        font_name = str(row["font_name"])
        word      = self.words[idx]
        size      = self.sizes[idx]

        raw_np = _render_font(font_path, word, size)          # (224,224,3) uint8
        tensor = preprocess_inference_crop(raw_np).squeeze(0) # (3,224,224) float32

        # Return raw_np as a plain numpy array (safe to pickle on Windows)
        return tensor, faiss_id, font_name, word, raw_np


def _collate(batch):
    tensors, ids, names, words, raws = zip(*batch)
    return (
        torch.stack(tensors),   # (B, 3, 224, 224)
        list(ids),
        list(names),
        list(words),
        list(raws),             # list of (224,224,3) uint8 np arrays
    )
# ─────────────────────────────────────────────────────────────────────────────


# ── Contact sheet ─────────────────────────────────────────────────────────────
def build_contact_sheet(sheet_samples, output_path):
    """
    sheet_samples: list of dicts with keys:
        true_font, pred_font, confidence, top1_correct, top5_correct, raw_np
    """
    rows  = math.ceil(len(sheet_samples) / COLS)
    sheet = Image.new("RGB", (CELL_W * COLS, CELL_H * rows), (30, 30, 30))
    draw  = ImageDraw.Draw(sheet)

    try:
        lbl_font = PILImageFont.truetype("arial.ttf", 9)
    except IOError:
        lbl_font = PILImageFont.load_default()

    for idx, r in enumerate(sheet_samples):
        col, row_idx = idx % COLS, idx // COLS
        cx, cy = col * CELL_W, row_idx * CELL_H

        thumb = Image.fromarray(r["raw_np"]).resize((56, 56))
        sheet.paste(thumb, (cx + 4, cy + 4))

        colour = (50, 200, 80) if r["top1_correct"] else (220, 60, 60)
        badge  = ("✓ TOP-1" if r["top1_correct"]
                  else "✓ TOP-5" if r["top5_correct"]
                  else "✗ MISS")

        draw.text((cx + 64, cy +  6), f"True:  {r['true_font'][:30]}", font=lbl_font, fill=(200, 200, 200))
        draw.text((cx + 64, cy + 22), f"Pred:  {r['pred_font'][:30]}", font=lbl_font, fill=colour)
        draw.text((cx + 64, cy + 38), f"Conf:  {r['confidence']:.1f}%", font=lbl_font, fill=(160, 160, 160))
        draw.text((cx + 64, cy + 54), badge,                            font=lbl_font, fill=colour)
        draw.rectangle([cx, cy, cx + CELL_W - 2, cy + CELL_H - 2], outline=colour, width=1)

    sheet.save(output_path)
    print(f"Contact sheet saved  → {output_path}")
# ─────────────────────────────────────────────────────────────────────────────


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate_accuracy(num_samples: int = 100_000,
                      batch_size:  int = BATCH_SIZE,
                      num_workers: int = NUM_WORKERS):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  batch_size={batch_size}  |  workers={num_workers}")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    imgs_dir = Path(OUTPUT_DIR) / "test_images"
    imgs_dir.mkdir(exist_ok=True)

    # ── Model ──
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model weights not found at '{MODEL_PATH}'"); return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()

    # ── Index ──
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        print("Error: index files not found – run build_index.py first."); return
    index      = faiss.read_index(INDEX_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    id2name    = dict(zip(mapping_df["faiss_id"].astype(int), mapping_df["font_name"]))

    # ── Sample ──
    if len(mapping_df) < num_samples:
        print(f"Warning: only {len(mapping_df)} fonts available – evaluating all.")
        sampled_df = mapping_df
    else:
        sampled_df = mapping_df.sample(n=num_samples, random_state=42)

    total = len(sampled_df)
    print(f"Evaluating {total} fonts …")

    dataset = FontEvalDataset(sampled_df)
    loader  = DataLoader(
        dataset,
        batch_size         = batch_size,
        num_workers        = num_workers,
        collate_fn         = _collate,
        pin_memory         = device.type == "cuda",
        prefetch_factor    = 2 if num_workers > 0 else None,
        persistent_workers = num_workers > 0,
    )

    top1_correct  = 0
    top5_correct  = 0
    global_idx    = 0          # running sample counter
    csv_rows      = []         # lightweight dicts (no images)
    sheet_samples = []         # PIL data for first VISUAL_SAMPLES only

    with torch.no_grad():
        for tensors, faiss_ids, font_names, words, raw_nps in tqdm(loader, desc="GPU inference"):
            # ── GPU inference ──
            tensors = tensors.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                embs = model(tensors)
            embs = F.normalize(embs, p=2, dim=1)

            vectors_np           = embs.cpu().numpy().astype("float32")
            distances, faiss_idx = index.search(vectors_np, 5)

            for i, expected_id in enumerate(faiss_ids):
                top_matches = faiss_idx[i]
                top1_ok = bool(expected_id == top_matches[0])
                top5_ok = bool(expected_id in top_matches)

                pred_name  = id2name.get(int(top_matches[0]), "Unknown")
                confidence = float(distances[i][0]) * 100

                if top1_ok: top1_correct += 1
                if top5_ok: top5_correct += 1

                # Save test image to disk immediately (no RAM accumulation)
                status = "correct" if top1_ok else ("top5" if top5_ok else "miss")
                fname  = (f"{global_idx:05d}_{status}_{font_names[i][:30]}.png"
                          .replace("/", "_").replace("\\", "_"))
                Image.fromarray(raw_nps[i]).save(imgs_dir / fname)

                # Accumulate contact-sheet data only for first N samples
                if global_idx < VISUAL_SAMPLES:
                    sheet_samples.append({
                        "true_font":    font_names[i],
                        "pred_font":    pred_name,
                        "confidence":   confidence,
                        "top1_correct": top1_ok,
                        "top5_correct": top5_ok,
                        "raw_np":       raw_nps[i],
                    })

                csv_rows.append({
                    "sample_idx":   global_idx,
                    "true_font":    font_names[i],
                    "pred_font":    pred_name,
                    "test_word":    words[i],
                    "top1_correct": top1_ok,
                    "top5_correct": top5_ok,
                    "confidence":   round(confidence, 2),
                })
                global_idx += 1

    # ── Metrics ──
    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100
    print("\n" + "=" * 50)
    print(f"  Evaluation Results  ({total} fonts)")
    print(f"  Top-1 Accuracy : {top1_acc:.2f}%  ({top1_correct}/{total})")
    print(f"  Top-5 Accuracy : {top5_acc:.2f}%  ({top5_correct}/{total})")
    print("=" * 50)

    # ── CSV report ──
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"CSV report saved    → {csv_path}")
    print(f"Test images saved   → {imgs_dir}/")

    # ── Contact sheet ──
    build_contact_sheet(sheet_samples, os.path.join(OUTPUT_DIR, "contact_sheet.png"))


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    evaluate_accuracy()
