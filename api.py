import os
import faiss
import pandas as pd
import numpy as np
import cv2
import io
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import shutil
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image

app = FastAPI(title="Font Identifier API", description="Backend API for Font Identifier Web App")

# Allow CORS for Netlify static frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Font Identifier API Backend is Running!"}

EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

# Global Variables
ort_session = None
index = None
mapping_df = None

@app.on_event("startup")
async def startup_event():
    global ort_session, index, mapping_df
    
    if not os.path.exists("best_model.onnx") or not os.path.exists(INDEX_PATH):
        print("WARNING: ONNX Model or FAISS index not found! Ensure they are downloaded.")
        return

    print("Loading ONNX Model (Ultra-Low RAM Mode)...")
    ort_session = ort.InferenceSession("best_model.onnx", providers=['CPUExecutionProvider'])
    
    print("Loading FAISS Index with Memory Mapping (Low RAM Mode)...")
    # Use MMAP to avoid loading the massive index entirely into active RAM
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    
    print("Loading Mapping Data...")
    mapping_df = pd.read_csv(MAPPING_PATH)
    print("Initialization Complete!")

def preprocess_image(image_np):
    # Convert to grayscale first
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Binarize using OTSU (THRESH_BINARY keeps background white if it's lighter)
    # The model was trained on white background and black text.
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert it (ensure background is white)
    # If the corners are black (0), it means the background is dark, so we invert.
    corners = [binarized[0,0], binarized[0,-1], binarized[-1,0], binarized[-1,-1]]
    if sum(corners) < 255 * 2: # mostly black corners
        binarized = cv2.bitwise_not(binarized)
        
    # Aspect-ratio preserving resize with white padding
    target_w, target_h = 256, 64
    h, w = binarized.shape
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    resized = cv2.resize(binarized, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create white canvas and paste
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    
    rgb_ready = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_ready / 255.0 - mean) / std
    
    # Transpose to (C, H, W) and add batch dimension (1, C, H, W)
    tensor = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
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
    if ort_session is None or index is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image)
        
        # Preprocess
        tensor = preprocess_image(image_np)
        
        # Extract features using ONNX
        outputs = ort_session.run(None, {'input': tensor})
        embedding = outputs[0]
        
        # Normalize the embedding using numpy (L2 normalization)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norm
            
        # FAISS search
        distances, indices = index.search(embedding.astype('float32'), 10)
        
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

def remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@app.get("/font/{filename}")
def get_font(filename: str, background_tasks: BackgroundTasks):
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
    is_cached = False
    if not local_path or not os.path.exists(local_path):
        cache_dir = "font_cache"
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, filename)
        
        if not os.path.exists(local_path):
            import urllib.parse
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Look up the exact dataset path in the mapping dataframe
            match = mapping_df[mapping_df['font_path'].str.contains(filename, regex=False)]
            if match.empty:
                raise HTTPException(status_code=404, detail="Font not found in mapping.")
                
            kaggle_internal_path = match.iloc[0]['font_path']
            # Convert Windows paths to forward slashes for Kaggle API
            kaggle_internal_path = kaggle_internal_path.replace("\\", "/")
            
            # The Kaggle API expects the path *relative* to the dataset root
            dataset_name = "ttf-files-for-fonts"
            if dataset_name in kaggle_internal_path:
                kaggle_internal_path = kaggle_internal_path.split(dataset_name + "/")[-1]
            
            # Use Kaggle Python API to prevent OOM from spawning 10 subprocesses
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_file(
                dataset="tapomoysarkar/ttf-files-for-fonts",
                file_name=kaggle_internal_path,
                path=cache_dir
            )
            
            # The Kaggle API URL-encodes filenames with spaces (e.g. My%20Font.ttf)
            # We must unquote the downloaded files to match our target filename
            for root, _, files in os.walk(cache_dir):
                for f in files:
                    if urllib.parse.unquote(f) == filename:
                        # We found it! Rename it back to its normal name with spaces so it works seamlessly
                        encoded_path = os.path.join(root, f)
                        normal_path = os.path.join(root, filename)
                        if encoded_path != normal_path:
                            os.rename(encoded_path, normal_path)
                        local_path = normal_path
                        break
        
        is_cached = True
                    
    if not local_path or not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Failed to fetch font file from Kaggle.")
        
    if is_cached:
        background_tasks.add_task(remove_file, local_path)
        
    return FileResponse(local_path, media_type="application/x-font-truetype", filename=filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
