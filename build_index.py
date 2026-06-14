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

from train_virtual_epochs import FontEmbeddingModel, TTF_DIR

# Hardcoded constraints
EMBEDDING_SIZE = 256
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

# Canonical Renders
CANONICAL_STRINGS = ["AaBbCc", "xyz123", "0OIl", "gjpqy", "Test 00"]

def get_inference_transform():
    # Only normalize, no augmentation for clean canonical renders
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

def build_index():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = FontEmbeddingModel(embedding_size=EMBEDDING_SIZE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    else:
        print(f"Warning: {MODEL_PATH} not found. Using untrained weights for demonstration.")
    model.to(device)
    model.eval()

    transform = get_inference_transform()
    ttf_files = list(Path(TTF_DIR).rglob("*.ttf"))
    print(f"Found {len(ttf_files)} fonts to index.")

    # Initialize FAISS Index (Inner Product for Cosine Similarity since vectors are L2 normalized)
    index = faiss.IndexFlatIP(EMBEDDING_SIZE)
    mapping_data = []
    font_vectors = []
    
    # Process each font sequentially to build a robust index
    for i in tqdm(range(len(ttf_files)), desc="Extracting canonical embeddings"):
        ttf_path = ttf_files[i]
        
        tensors = []
        for text in CANONICAL_STRINGS:
            img = render_string(ttf_path, text)
            tensor = transform(img)
            tensors.append(tensor)
            
        # Shape: (5, 3, 224, 224)
        batch = torch.stack(tensors).to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True):
                embeddings = model(batch) # Shape: (5, 256)
            
            # Average the 5 embeddings to create a stable vector
            avg_embedding = torch.mean(embeddings, dim=0, keepdim=True)
            # Re-normalize to ensure L2 norm = 1 (Cosine Similarity prerequisite)
            avg_embedding = F.normalize(avg_embedding, p=2, dim=1)
            
        font_vectors.append(avg_embedding.cpu().numpy().flatten())
        mapping_data.append({"faiss_id": i, "font_path": str(ttf_path), "font_name": ttf_path.stem})

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
