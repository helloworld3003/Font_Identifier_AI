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

from train_virtual_epochs import ConvNeXtFontEncoder, TTF_DIR

# Hardcoded constraints
EMBEDDING_SIZE = 256
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

def render_string(ttf_path, text, canvas_size=224, font_size=60):
    """Render canonical strings accurately."""
    try:
        font = ImageFont.truetype(str(ttf_path), font_size)
        image = Image.new("RGB", (canvas_size, canvas_size), "white")
        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        x = (canvas_size - text_w) / 2
        y = (canvas_size - text_h) / 2
        
        draw.text((x, y), text, font=font, fill="black")
        return image
    except Exception:
        return Image.new("RGB", (canvas_size, canvas_size), "white")

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
        
        # Shape: (5, 3, 224, 224)
        return torch.stack(tensors), idx

def build_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ConvNeXt Model
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else:
        print(f"Warning: {MODEL_PATH} not found. Using untrained weights for demonstration.")
    model.to(device)
    model.eval()

    transform = get_inference_transform()
    ttf_files = list(Path(TTF_DIR).rglob("*.ttf"))
    print(f"Found {len(ttf_files)} fonts to index.")

    dataset = FontRenderDataset(ttf_files, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True
    )

    # Initialize FAISS Index (Inner Product for Cosine Similarity since vectors are L2 normalized)
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    mapping_data = []
    font_vectors = []
    
    # Process fonts in batches
    for batch_tensors, indices in tqdm(dataloader, desc="Extracting canonical embeddings"):
        B = batch_tensors.size(0)
        flat_batch = batch_tensors.view(B * 5, 3, 224, 224).to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True):
                embeddings = model(flat_batch) # Shape: (B * 5, 256)
            
            # Reshape back to (B, 5, 256)
            embeddings = embeddings.view(B, 5, EMBEDDING_SIZE)
            # Average the 5 embeddings to create a stable vector
            avg_embeddings = torch.mean(embeddings, dim=1) # Shape: (B, 256)
            # Re-normalize to ensure L2 norm = 1 (Cosine Similarity prerequisite)
            avg_embeddings = F.normalize(avg_embeddings, p=2, dim=1)
            
        font_vectors.append(avg_embeddings.cpu().numpy())
        
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
