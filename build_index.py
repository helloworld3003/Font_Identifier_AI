import os
import gc
import sys
import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

# Import the correct modern model architecture
from model import ConvNeXtFontEncoder

# ==========================================
# 1. HARDCODED CONFIGURATION
# ==========================================
EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"
# We drastically reduce CHUNK_SIZE so if 1 corrupted font detonates a 30GB OOM, 
# it only takes down 500 fonts with it instead of 10,000!
CHUNK_SIZE = 500 

CANONICAL_STRINGS = ["AaBbCc", "xyz123", "0OIl", "gjpqy", "Test 00"]

# ==========================================
# 2. UTILITIES & DATASET
# ==========================================
def get_inference_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def render_string(ttf_path, text, target_w=256, target_h=64, font_size=60):
    try:
        font = ImageFont.truetype(str(ttf_path), font_size)
        temp_image = Image.new("RGB", (1, 1), "white")
        temp_draw = ImageDraw.Draw(temp_image)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        
        image = Image.new("RGB", (text_w, text_h), "white")
        draw = ImageDraw.Draw(image)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill="black")
        
        scale = min(target_w / text_w, target_h / text_h)
        new_w = max(1, int(text_w * scale))
        new_h = max(1, int(text_h * scale))
        
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        final_image = Image.new("RGB", (target_w, target_h), "white")
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        final_image.paste(image, (x, y))
        
        del draw
        del temp_draw
        del font
        del temp_image
        del image
        return final_image
    except Exception:
        return Image.new("RGB", (target_w, target_h), "white")

class FontRenderDataset(Dataset):
    def __init__(self, ttf_files, transform=None):
        self.ttf_files = ttf_files
        self.transform = transform

    def __len__(self):
        return len(self.ttf_files)

    def __getitem__(self, idx):
        ttf_path = self.ttf_files[idx]
        tensors = []
        for text in CANONICAL_STRINGS:
            img = render_string(ttf_path, text)
            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            tensors.append(tensor)
        
        return torch.stack(tensors), idx

# ==========================================
# 3. ISOLATED SUBPROCESS WORKER
# ==========================================
def process_chunk_worker(chunk_idx, chunk_files, npy_path, csv_path):
    """
    This function runs in a completely isolated OS process.
    When it returns (or crashes), all C-level memory leaks are aggressively reclaimed by the OS.
    """
    try:
        # Re-initialize CUDA inside the child process to avoid context corruption
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False # Disable benchmarking to prevent VRAM spikes
        
        model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        
        model.to(device)
        model.eval()

        transform = get_inference_transform()
        dataset = FontRenderDataset(chunk_files, transform=transform)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=0, pin_memory=False)

        chunk_vectors = []
        chunk_mapping = []

        print(f"[Worker {chunk_idx // CHUNK_SIZE + 1}] Starting extraction of {len(chunk_files)} fonts...")
        
        # We don't use tqdm here because multiprocessing stdout overlaps. Just print progress periodically.
        for batch_idx, (batch_tensors, indices) in enumerate(dataloader):
            B = batch_tensors.size(0)
            flat_batch = batch_tensors.view(B * 5, 3, 64, 256).to(device)
            
            with torch.no_grad():
                embeddings = model(flat_batch) 
                embeddings = embeddings.view(B, 5, EMBEDDING_SIZE)
                avg_embeddings = torch.mean(embeddings, dim=1) 
                avg_embeddings = F.normalize(avg_embeddings, p=2, dim=1)
                
            chunk_vectors.append(avg_embeddings.cpu().float().numpy())
            
            for idx_in_batch in range(B):
                local_idx = indices[idx_in_batch].item()
                global_idx = chunk_idx + local_idx
                ttf_path = chunk_files[local_idx]
                chunk_mapping.append({
                    "faiss_id": global_idx, 
                    "font_path": str(ttf_path), 
                    "font_name": ttf_path.stem
                })
                
            del flat_batch
            del batch_tensors
            del embeddings
            del avg_embeddings
            gc.collect()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"[Worker {chunk_idx // CHUNK_SIZE + 1}] Processed {(batch_idx + 1) * 128}/{len(chunk_files)} fonts")

        vectors_np = np.vstack(chunk_vectors).astype('float32')
        np.save(npy_path, vectors_np)
        pd.DataFrame(chunk_mapping).to_csv(csv_path, index=False)
        
        print(f"[Worker {chunk_idx // CHUNK_SIZE + 1}] Successfully saved to disk. Committing seppuku to free C-level RAM.")
        sys.exit(0) # Explicitly kill the worker cleanly
        
    except Exception as e:
        print(f"[Worker {chunk_idx // CHUNK_SIZE + 1}] FATAL EXCEPTION: {e}")
        sys.exit(1)

# ==========================================
# 4. ORCHESTRATOR
# ==========================================
def build_index_orchestrator():
    print("=" * 60)
    print("STARTING ROBUST ISOLATED SUBPROCESS INDEX BUILDER")
    print("=" * 60)
    
    # Auto-detect Kaggle Directory
    kaggle_input_dir = Path("/kaggle/input")
    if kaggle_input_dir.exists():
        try:
            target_dir = None
            for d in kaggle_input_dir.rglob("ttf_files"):
                if d.is_dir():
                    target_dir = d
                    break
            if target_dir:
                TTF_DIR = str(target_dir)
            else:
                first_ttf = next(kaggle_input_dir.rglob("*.ttf"))
                TTF_DIR = str(first_ttf.parent)
        except StopIteration:
            TTF_DIR = "ttf_files"
    else:
        TTF_DIR = "ttf_files"

    print(f"\nScanning for font files in {TTF_DIR}...")
    all_files = list(Path(TTF_DIR).rglob("*.ttf")) + list(Path(TTF_DIR).rglob("*.otf"))
    
    print("Validating font headers to prevent C-level Segmentation Faults...")
    
    blacklist = set()
    if os.path.exists("bomb_blacklist.txt"):
        with open("bomb_blacklist.txt", "r") as f:
            blacklist = set([line.strip() for line in f.readlines()])
            
    ttf_files = []
    for f in all_files:
        if str(f) in blacklist:
            continue
        try:
            if os.path.getsize(f) > 1024:
                with open(f, 'rb') as file:
                    head = file.read(4)
                    if head in (b'\x00\x01\x00\x00', b'OTTO', b'ttcf', b'true'):
                        ttf_files.append(f)
        except Exception:
            pass
            
    total_fonts = len(ttf_files)
    print(f"Found {total_fonts} valid fonts (filtered out {len(all_files) - total_fonts} corrupted files).")

    os.makedirs("faiss_chunks", exist_ok=True)
    chunk_paths = []
    
    for chunk_idx in range(0, total_fonts, CHUNK_SIZE):
        chunk_files = ttf_files[chunk_idx:chunk_idx+CHUNK_SIZE]
        chunk_npy_path = f"faiss_chunks/vectors_{chunk_idx}.npy"
        chunk_csv_path = f"faiss_chunks/mapping_{chunk_idx}.csv"
        
        print(f"\n--- Orchestrator: Spawning Isolated Worker for Chunk {chunk_idx // CHUNK_SIZE + 1} ---")
        
        # Spawn an entirely isolated Python process
        p = mp.Process(target=process_chunk_worker, args=(chunk_idx, chunk_files, chunk_npy_path, chunk_csv_path))
        p.start()
        p.join() # The orchestrator waits safely while the worker takes the bullet
        
        if p.exitcode != 0:
            print(f"WARNING: Subprocess {chunk_idx // CHUNK_SIZE + 1} terminated with abnormal exit code: {p.exitcode}")
            if p.exitcode == -11:
                print(">>> This was a Segmentation Fault caused by a malicious font file in the data leak! The worker took the hit, but the main orchestrator survives.")
            # We still append the chunk paths, in case it managed to save before dying, or just skip it if files don't exist
            
        if os.path.exists(chunk_npy_path) and os.path.exists(chunk_csv_path):
            chunk_paths.append((chunk_npy_path, chunk_csv_path))
        else:
            print(f"ERROR: Chunk {chunk_idx // CHUNK_SIZE + 1} failed to write files. It will be skipped from the final FAISS index.")

    print("\n" + "=" * 60)
    print("ALL CHUNKS PROCESSED. ASSEMBLING FINAL FAISS INDEX...")
    
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    all_mapping = []
    
    for npy_path, csv_path in tqdm(chunk_paths, desc="Assembling Index"):
        vectors_np = np.load(npy_path)
        index.add(vectors_np)
        
        chunk_df = pd.read_csv(csv_path)
        all_mapping.append(chunk_df)
        
        del vectors_np
        del chunk_df
        gc.collect()
        
    faiss.write_index(index, INDEX_PATH)
    final_df = pd.concat(all_mapping, ignore_index=True)
    final_df['faiss_id'] = pd.to_numeric(final_df['faiss_id'], downcast='unsigned')
    final_df.to_csv(MAPPING_PATH, index=False)
    
    print(f"SUCCESS! Robust FAISS index saved to {INDEX_PATH}")
    print(f"SUCCESS! Metadata mapping saved to {MAPPING_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    # CRITICAL: Must use 'spawn' to prevent CUDA initialization errors in subprocesses
    mp.set_start_method('spawn', force=True)
    build_index_orchestrator()
