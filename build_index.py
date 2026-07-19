import os
import gc
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

# Import the correct modern model architecture
from model import ConvNeXtFontEncoder

# ==========================================
# 1. HARDCODED CONFIGURATION
# ==========================================
EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"
CHUNK_SIZE = 10000 # Save embeddings to disk every 10,000 fonts to prevent RAM OOM

CANONICAL_STRINGS = ["AaBbCc", "xyz123", "0OIl", "gjpqy", "Test 00"]

# ==========================================
# 2. DATASET ROBUST KAGGLE AUTO-DETECT
# ==========================================
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

# ==========================================
# 3. UTILITIES
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
# 4. CHUNKED FAISS BUILDER
# ==========================================
def build_index():
    print("=" * 60)
    print("STARTING ROBUST T4 GPU CHUNKED INDEX BUILDER")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # Extreme speedhack for static shapes

    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"Loaded trained weights from {MODEL_PATH}")
    
    # Strictly bind to single GPU to absolutely prevent scatter/gather thread leaks
    model.to(device)
    model.eval()

    transform = get_inference_transform()
    
    print(f"\nScanning for font files in {TTF_DIR}...")
    ttf_files = list(Path(TTF_DIR).rglob("*.ttf")) + list(Path(TTF_DIR).rglob("*.otf"))
    total_fonts = len(ttf_files)
    print(f"Found {total_fonts} fonts to index.")

    # Prepare temp directory for chunks
    os.makedirs("faiss_chunks", exist_ok=True)
    
    # Process strictly in chunks
    chunk_paths = []
    
    for chunk_idx in range(0, total_fonts, CHUNK_SIZE):
        chunk_files = ttf_files[chunk_idx:chunk_idx+CHUNK_SIZE]
        print(f"\n--- Processing Chunk {chunk_idx // CHUNK_SIZE + 1} (Fonts {chunk_idx} to {chunk_idx+len(chunk_files)}) ---")
        
        dataset = FontRenderDataset(chunk_files, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=128, 
            num_workers=0, # Physically prevent multiprocessing leaks
            pin_memory=False
        )

        chunk_vectors = []
        chunk_mapping = []

        for batch_tensors, indices in tqdm(dataloader, desc="Extracting chunk"):
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
                
            # Physically purge Python garbage collector
            del flat_batch
            del batch_tensors
            del embeddings
            del avg_embeddings
            gc.collect()

        # Save this chunk immediately to disk to prevent RAM bloat
        vectors_np = np.vstack(chunk_vectors).astype('float32')
        chunk_npy_path = f"faiss_chunks/vectors_{chunk_idx}.npy"
        chunk_csv_path = f"faiss_chunks/mapping_{chunk_idx}.csv"
        
        np.save(chunk_npy_path, vectors_np)
        pd.DataFrame(chunk_mapping).to_csv(chunk_csv_path, index=False)
        chunk_paths.append((chunk_npy_path, chunk_csv_path))
        
        # Purge entire chunk from RAM
        del vectors_np
        del chunk_vectors
        del chunk_mapping
        del dataset
        del dataloader
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Chunk flushed safely to disk. RAM wiped.")

    print("\n" + "=" * 60)
    print("ALL CHUNKS PROCESSED. ASSEMBLING FINAL FAISS INDEX...")
    
    # Initialize FAISS Index
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    all_mapping = []
    
    # Load chunks one by one to assemble FAISS
    for npy_path, csv_path in tqdm(chunk_paths, desc="Assembling Index"):
        vectors_np = np.load(npy_path)
        index.add(vectors_np)
        
        chunk_df = pd.read_csv(csv_path)
        all_mapping.append(chunk_df)
        
        # Purge loaded array
        del vectors_np
        del chunk_df
        gc.collect()
        
    faiss.write_index(index, INDEX_PATH)
    final_df = pd.concat(all_mapping, ignore_index=True)
    final_df.to_csv(MAPPING_PATH, index=False)
    
    print(f"SUCCESS! Robust FAISS index saved to {INDEX_PATH}")
    print(f"SUCCESS! Metadata mapping saved to {MAPPING_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    build_index()
