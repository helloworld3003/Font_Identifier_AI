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
from tqdm import tqdm

from train_virtual_epochs import ConvNeXtFontEncoder
from inference import preprocess_inference_crop

EMBEDDING_SIZE = 256
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"
OUTPUT_DIR = "eval_results"

# Contact sheet settings: show this many samples in the visual sheet
VISUAL_SAMPLES = 100  # first 100 of 500 to keep image manageable
CELL_W, CELL_H = 280, 100  # pixels per cell in the contact sheet
COLS = 5                    # columns in the contact sheet

def generate_random_string(length=5):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def render_font_crop(ttf_path, text, canvas_size=224, font_size=60):
    try:
        font = PILImageFont.truetype(str(ttf_path), font_size)
        image = Image.new("RGB", (canvas_size, canvas_size), "white")
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (canvas_size - text_w) / 2
        y = (canvas_size - text_h) / 2
        draw.text((x, y), text, font=font, fill="black")
        return np.array(image), image
    except Exception:
        blank = Image.new("RGB", (canvas_size, canvas_size), "white")
        return np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255, blank

def build_contact_sheet(results, output_path):
    """
    Build a visual grid contact sheet from the first VISUAL_SAMPLES results.
    Each cell shows the rendered test image, the true font name and predicted
    font name, colour-coded green (correct) or red (wrong).
    """
    samples = results[:VISUAL_SAMPLES]
    rows = math.ceil(len(samples) / COLS)
    sheet_w = CELL_W * COLS
    sheet_h = CELL_H * rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    try:
        label_font = PILImageFont.truetype("arial.ttf", 9)
    except IOError:
        label_font = PILImageFont.load_default()

    for idx, r in enumerate(samples):
        col = idx % COLS
        row_idx = idx // COLS
        cx = col * CELL_W
        cy = row_idx * CELL_H

        # Paste the test render thumbnail (56x56)
        thumb = r["pil_image"].resize((56, 56))
        sheet.paste(thumb, (cx + 4, cy + 4))

        # Colour: green if correct, red if wrong
        colour = (50, 200, 80) if r["top1_correct"] else (220, 60, 60)

        # True font name (truncated)
        true_name = r["true_font"][:30]
        pred_name = r["pred_font"][:30]
        conf = r["confidence"]

        draw.text((cx + 64, cy + 6),  f"True:  {true_name}", font=label_font, fill=(200, 200, 200))
        draw.text((cx + 64, cy + 22), f"Pred:  {pred_name}", font=label_font, fill=colour)
        draw.text((cx + 64, cy + 38), f"Conf:  {conf:.1f}%", font=label_font, fill=(160, 160, 160))

        # Top1 status badge
        badge = "✓ TOP-1" if r["top1_correct"] else ("✓ TOP-5" if r["top5_correct"] else "✗ MISS")
        draw.text((cx + 64, cy + 54), badge, font=label_font, fill=colour)

        # Border
        draw.rectangle([cx, cy, cx + CELL_W - 2, cy + CELL_H - 2], outline=colour, width=1)

    sheet.save(output_path)
    print(f"Visual contact sheet saved to: {output_path}")

def evaluate_accuracy(num_samples=100000, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load Model
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load FAISS index & mappings
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        print("Error: Index files not found. Run build_index.py first.")
        return

    index = faiss.read_index(INDEX_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)

    # Sample random fonts
    if len(mapping_df) < num_samples:
        print(f"Warning: Only {len(mapping_df)} fonts available. Evaluating all.")
        sampled_df = mapping_df
    else:
        sampled_df = mapping_df.sample(n=num_samples, random_state=42).reset_index(drop=True)

    print(f"Evaluating accuracy on {len(sampled_df)} randomly sampled fonts...")

    top1_correct = 0
    top5_correct = 0
    total = len(sampled_df)
    all_results = []

    # --- Per-batch processing ---
    for start_idx in tqdm(range(0, total, batch_size), desc="Running evaluation"):
        end_idx = min(start_idx + batch_size, total)
        batch_df = sampled_df.iloc[start_idx:end_idx]

        crops_tensors = []
        pil_images   = []
        test_words   = []
        expected_ids = []
        font_names   = []

        for _, row in batch_df.iterrows():
            faiss_id   = int(row["faiss_id"])
            font_path  = row["font_path"]
            font_name  = row["font_name"]
            test_word  = generate_random_string(random.randint(4, 7))
            font_size  = random.randint(45, 75)

            crop_np, pil_img = render_font_crop(font_path, test_word, font_size=font_size)
            tensor = preprocess_inference_crop(crop_np)

            crops_tensors.append(tensor)
            pil_images.append(pil_img)
            test_words.append(test_word)
            expected_ids.append(faiss_id)
            font_names.append(font_name)

        if not crops_tensors:
            continue

        # GPU batch inference
        batch_tensor = torch.cat(crops_tensors, dim=0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True):
                embeddings = model(batch_tensor)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        vectors_np = embeddings.cpu().numpy().astype("float32")
        distances, indices = index.search(vectors_np, 5)

        for i, expected_id in enumerate(expected_ids):
            top_matches = indices[i]
            top1_ok = bool(expected_id == top_matches[0])
            top5_ok = bool(expected_id in top_matches)

            pred_id   = int(top_matches[0])
            pred_row  = mapping_df[mapping_df["faiss_id"] == pred_id]
            pred_name = pred_row["font_name"].values[0] if len(pred_row) else "Unknown"
            confidence = float(distances[i][0]) * 100

            if top1_ok:
                top1_correct += 1
            if top5_ok:
                top5_correct += 1

            all_results.append({
                "true_font":     font_names[i],
                "pred_font":     pred_name,
                "test_word":     test_words[i],
                "top1_correct":  top1_ok,
                "top5_correct":  top5_ok,
                "confidence":    round(confidence, 2),
                "pil_image":     pil_images[i],   # kept in memory for contact sheet only
            })

    # --- Summary metrics ---
    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100

    print("\n" + "=" * 50)
    print(f"  Accuracy Evaluation Results ({total} fonts)")
    print(f"  Top-1 Accuracy : {top1_acc:.2f}%  ({top1_correct}/{total})")
    print(f"  Top-5 Accuracy : {top5_acc:.2f}%  ({top5_correct}/{total})")
    print("=" * 50)

    # --- Save CSV report ---
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    csv_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pil_image"} for r in all_results])
    csv_df.to_csv(csv_path, index=False)
    print(f"Per-sample CSV report saved to: {csv_path}")

    # --- Save individual test images ---
    imgs_dir = os.path.join(OUTPUT_DIR, "test_images")
    Path(imgs_dir).mkdir(exist_ok=True)
    for idx, r in enumerate(all_results):
        status = "correct" if r["top1_correct"] else ("top5" if r["top5_correct"] else "miss")
        fname = f"{idx:04d}_{status}_{r['true_font'][:30]}.png".replace("/", "_").replace("\\", "_")
        r["pil_image"].save(os.path.join(imgs_dir, fname))
    print(f"Test images saved to: {imgs_dir}/")

    # --- Build contact sheet ---
    sheet_path = os.path.join(OUTPUT_DIR, "contact_sheet.png")
    build_contact_sheet(all_results, sheet_path)

if __name__ == "__main__":
    evaluate_accuracy()
