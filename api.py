import os
import torch
import faiss
import pandas as pd
import numpy as np
import cv2
import io
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image

from model import ConvNeXtFontEncoder

app = FastAPI(title="Font Identifier API", description="Backend API for Font Identifier Web App")

# Allow CORS for Netlify static frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

# Global Variables
device = None
model = None
index = None
mapping_df = None

@app.on_event("startup")
def load_model():
    global device, model, index, mapping_df
    print("Loading AI Model and FAISS Index...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(INDEX_PATH):
        print("WARNING: Model or FAISS index not found! Ensure they are downloaded.")
        return

    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    index = faiss.read_index(INDEX_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    print("Initialization Complete!")

def preprocess_image(image_np):
    image_np = cv2.resize(image_np, (256, 64))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rgb_ready = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (rgb_ready / 255.0 - mean) / std
    
    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

def resolve_local_font_path(kaggle_path):
    if os.path.exists(kaggle_path):
        return kaggle_path
    filename = Path(kaggle_path).name
    # Search locally
    for root, _, files in os.walk("ttf_files"):
        if filename in files:
            return os.path.join(root, filename)
    for root, _, files in os.walk("ttf_files_2"):
        if filename in files:
            return os.path.join(root, filename)
    return None

@app.post("/predict")
async def predict_font(file: UploadFile = File(...)):
    if model is None or index is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Preprocess
        tensor = preprocess_image(image_np).to(device)
        
        # Extract features
        with torch.no_grad():
            embedding = F.normalize(model(tensor), p=2, dim=1)
            
        # FAISS search
        distances, indices = index.search(embedding.cpu().numpy().astype('float32'), 10)
        
        results = []
        for i in range(10):
            confidence = float(((distances[0][i] + 1) / 2) * 100)
            match_row = mapping_df.iloc[indices[0][i]]
            font_path = match_row['font_path']
            filename = Path(font_path).name
            font_name = filename.replace("-", " ").replace(".ttf", "").replace(".otf", "").title()
            
            results.append({
                "rank": i + 1,
                "font_name": font_name,
                "confidence": confidence,
                "filename": filename # We use filename to request the font later
            })
            
        return JSONResponse(content={"results": results})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/font/{filename}")
async def get_font(filename: str):
    """
    Serves the actual TTF font file.
    If it's not found locally, it dynamically downloads just that one file from Kaggle!
    """
    if mapping_df is None:
        raise HTTPException(status_code=503, detail="System initializing.")
        
    # Search locally first (for local testing)
    local_path = None
    for root, _, files in os.walk("ttf_files"):
        if filename in files:
            local_path = os.path.join(root, filename)
            break
            
    if not local_path:
        for root, _, files in os.walk("ttf_files_2"):
            if filename in files:
                local_path = os.path.join(root, filename)
                break
                
    # If not found locally, download it on the fly using Kaggle API
    if not local_path or not os.path.exists(local_path):
        cache_dir = "font_cache"
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(local_path):
            import subprocess
            # Look up the exact dataset path in the mapping dataframe
            match = mapping_df[mapping_df['font_path'].str.contains(filename)]
            if match.empty:
                raise HTTPException(status_code=404, detail="Font not found in mapping.")
                
            kaggle_internal_path = match.iloc[0]['font_path']
            # Convert Windows paths to forward slashes for Kaggle API
            kaggle_internal_path = kaggle_internal_path.replace("\\", "/")
            
            # Use Kaggle API to download JUST this one file!
            cmd = [
                "kaggle", "datasets", "download", 
                "-d", "tapomoysarkar/ttf-files-for-fonts", 
                "-f", kaggle_internal_path, 
                "-p", cache_dir, 
                "--unzip"
            ]
            print(f"Dynamically fetching from Kaggle: {' '.join(cmd)}")
            subprocess.run(cmd)
            
            # The Kaggle CLI might download it inside a nested folder structure within cache_dir
            # We'll just search cache_dir for the filename we want
            for root, _, files in os.walk(cache_dir):
                if filename in files:
                    local_path = os.path.join(root, filename)
                    break
                    
    if not local_path or not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Failed to fetch font file from Kaggle.")
        
    return FileResponse(local_path, media_type="font/ttf", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
