import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import faiss
from tqdm import tqdm

import torch
from torchvision import transforms

# Import the model architecture from Phase 2
from train_local_subset import MetricResNet18

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The optimal strings to capture maximal geometric variance of the typography
CANONICAL_STRINGS = ['AaBbCc', 'xyz123', '0OIl', 'gjpqy', 'Test 00']

def render_clean_text(font_path: Path, text: str, image_size=(256, 64), font_size=40) -> Image.Image:
    """Renders a clean, unaugmented text image for canonical embedding."""
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception as e:
        logger.error(f"Failed to load font {font_path}: {e}")
        return None

    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (image_size[0] - text_width) // 2
        y = (image_size[1] - text_height) // 2
        
        # Ensure text is slightly padded from edges
        x = max(5, min(x, image_size[0] - 10))
        
        draw.text((x, y), text, font=font, fill='black')
    except Exception as e:
        logger.error(f"Error drawing text '{text}' with font {font_path}: {e}")
        return None
        
    return img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # -----------------------------
    # 1. LOAD MODEL
    # -----------------------------
    model_path = "best_metric_model.pth"
    if not os.path.exists(model_path):
        # Fallback to the POC model if production hasn't run yet
        if os.path.exists("poc_metric_model.pth"):
            model_path = "poc_metric_model.pth"
            logger.warning(f"Production model not found. Falling back to {model_path}.")
        else:
            raise FileNotFoundError(f"Model weights not found. Run Phase 2 first.")
        
    model = MetricResNet18(embedding_size=128)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Using the same transform as training, but without random augmentations
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # -----------------------------
    # 2. LOAD FONTS
    # -----------------------------
    subset_csv = "local_test_fonts.csv"
    if not os.path.exists(subset_csv):
        raise FileNotFoundError(f"Subset CSV not found at {subset_csv}. Run Phase 2 first.")
        
    df = pd.read_csv(subset_csv)
    selected_fonts = set(df['font_name'].tolist())
    
    # Map TTF names to paths
    ttf_dir = Path("./ttf_files")
    ttf_paths = list(ttf_dir.rglob("*.ttf"))
    
    font_to_path = {}
    for p in ttf_paths:
        if p.stem in selected_fonts:
            font_to_path[p.stem] = p
            
    logger.info(f"Found {len(font_to_path)} TTF files out of {len(selected_fonts)} expected.")
    
    # -----------------------------
    # 3. EXTRACT EMBEDDINGS
    # -----------------------------
    embedding_dim = 128
    # Using Inner Product (Cosine Similarity) since embeddings are L2 normalized
    index = faiss.IndexFlatIP(embedding_dim) 
    
    faiss_mapping = []
    embeddings_list = []
    
    logger.info("Extracting canonical embeddings...")
    with torch.no_grad():
        for i, font_name in enumerate(tqdm(df['font_name'])):
            if font_name not in font_to_path:
                logger.warning(f"Skipping {font_name}: TTF file not found.")
                continue
                
            ttf_path = font_to_path[font_name]
            
            font_tensors = []
            for text in CANONICAL_STRINGS:
                img = render_clean_text(ttf_path, text)
                if img is not None:
                    tensor = transform(img)
                    font_tensors.append(tensor)
                    
            if not font_tensors:
                continue
                
            batch = torch.stack(font_tensors).to(device) # Shape: (5, 3, 64, 256)
            
            # Extract
            embeds = model(batch) # Shape: (5, 128)
            
            # Average the embeddings to get the stable canonical representation
            avg_embed = embeds.mean(dim=0, keepdim=True)
            
            # L2 Normalize the averaged embedding to map onto a unit hypersphere
            avg_embed = torch.nn.functional.normalize(avg_embed, p=2, dim=1)
            
            embeddings_list.append(avg_embed.cpu().numpy()[0])
            
            # Save mapping info
            faiss_mapping.append({
                "faiss_id": len(embeddings_list) - 1,
                "font_name": font_name,
                "ttf_path": str(ttf_path)
            })
            
    # -----------------------------
    # 4. BUILD FAISS INDEX
    # -----------------------------
    if not embeddings_list:
        logger.error("No embeddings extracted. Exiting.")
        return
        
    embeddings_array = np.array(embeddings_list).astype('float32')
    index.add(embeddings_array)
    
    # Save the index
    faiss.write_index(index, "font_embeddings.index")
    
    # Save the mapping metadata
    mapping_df = pd.DataFrame(faiss_mapping)
    mapping_df.to_csv("faiss_mapping.csv", index=False)
    
    logger.info(f"Successfully indexed {index.ntotal} fonts into FAISS.")
    logger.info("Saved index to 'font_embeddings.index' and mapping to 'faiss_mapping.csv'.")

if __name__ == "__main__":
    main()
