import os
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
# Check if running on Kaggle and auto-mount the dataset to prevent symlink errors
from pathlib import Path
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
        print(f"Auto-detected Kaggle dataset at: {TTF_DIR}")
    except StopIteration:
        TTF_DIR = "ttf_files"
else:
    TTF_DIR = "ttf_files"

# Hardcoded constraints MUST match train_tpu_8core.py exactly
EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

# Canonical Renders
CANONICAL_STRINGS = ["AaBbCc", "xyz123", "0OIl", "gjpqy", "Test 00"]

def get_inference_transform():
    # Only normalize, no augmentation for clean canonical renders
    # Since these are synthetic internal renders, they are already perfectly clean.
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def render_string(ttf_path, text, target_w=256, target_h=64, font_size=60):
    """Render canonical strings matching the TPU training exact logic."""
    try:
        font = ImageFont.truetype(str(ttf_path), font_size)
        
        # Create a temporary image to measure the text bounding box
        temp_image = Image.new("RGB", (1, 1), "white")
        temp_draw = ImageDraw.Draw(temp_image)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        
        # Render the raw text
        image = Image.new("RGB", (text_w, text_h), "white")
        draw = ImageDraw.Draw(image)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill="black")
        
        # Scale and pad to match training exact dimensions
        scale = min(target_w / text_w, target_h / text_h)
        new_w = max(1, int(text_w * scale))
        new_h = max(1, int(text_h * scale))
        
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        final_image = Image.new("RGB", (target_w, target_h), "white")
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        final_image.paste(image, (x, y))
        return final_image
    except Exception:
        # Fallback empty image if the font is corrupted or fails
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
        
        # Shape: (5, 3, 64, 256)
        return torch.stack(tensors), idx

def build_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # Extreme speedhack for static shapes

    # Load ConvNeXt Model
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"Loaded trained weights from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Using untrained weights for demonstration.")
    
    # Wrap in DataParallel to utilize both Kaggle T4 GPUs
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
        
    model.to(device)
    model.eval()

    transform = get_inference_transform()
    
    ttf_dir_actual = TTF_DIR
    
    print(f"Scanning for font files in {ttf_dir_actual}... (This may take a minute for 185,000+ files)")
    ttf_files = list(Path(ttf_dir_actual).rglob("*.ttf")) + list(Path(ttf_dir_actual).rglob("*.otf"))
    print(f"Found {len(ttf_files)} fonts to index in {ttf_dir_actual}.")

    dataset = FontRenderDataset(ttf_files, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=128, 
        num_workers=0,  # CRITICAL: Kaggle CANNOT handle background TTF rendering. Must be 0.
        pin_memory=False # CRITICAL: pin_memory=True consumes too much host RAM and causes OOM.
    )

    # Initialize FAISS Index (Inner Product for Cosine Similarity since vectors are L2 normalized)
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    mapping_data = []
    font_vectors = []
    
    # Process fonts in batches
    for batch_tensors, indices in tqdm(dataloader, desc="Extracting canonical embeddings"):
        B = batch_tensors.size(0)
        # Reshape to push all canonical renders through the batch dimension
        flat_batch = batch_tensors.view(B * 5, 3, 64, 256).to(device)
        
        with torch.no_grad():
            embeddings = model(flat_batch) # Shape: (B * 5, 512)
            
            # Reshape back to (B, 5, 512)
            embeddings = embeddings.view(B, 5, EMBEDDING_SIZE)
            # Average the 5 embeddings to create a highly stable, uniform signature vector per font
            avg_embeddings = torch.mean(embeddings, dim=1) # Shape: (B, 512)
            # Re-normalize to ensure L2 norm = 1 (Cosine Similarity FAISS prerequisite)
            avg_embeddings = F.normalize(avg_embeddings, p=2, dim=1)
            
        font_vectors.append(avg_embeddings.cpu().float().numpy())
        
        for idx_in_batch in range(B):
            global_idx = indices[idx_in_batch].item()
            ttf_path = ttf_files[global_idx]
            mapping_data.append({
                "faiss_id": global_idx, 
                "font_path": str(ttf_path), 
                "font_name": ttf_path.stem
            })

    # Ingest into FAISS
    vectors_np = np.vstack(font_vectors).astype('float32')
    index.add(vectors_np)
    
    # Save Index
    faiss.write_index(index, INDEX_PATH)
    print(f"Successfully saved FAISS index to {INDEX_PATH}")
    
    # Save Metadata Mapping
    df = pd.DataFrame(mapping_data)
    df.to_csv(MAPPING_PATH, index=False)
    print(f"Successfully saved metadata mapping to {MAPPING_PATH}")

if __name__ == "__main__":
    build_index()
