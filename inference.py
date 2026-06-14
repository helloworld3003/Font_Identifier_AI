import sys
import os
import torch
import faiss
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F

from train_virtual_epochs import ConvNeXtFontEncoder

EMBEDDING_SIZE = 256
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

def preprocess_inference_crop(image_np):
    """
    Converts a real-world RGB crop into a binarized, clean image
    to remove backgrounds, lighting variation, and shadows.
    Ensures Train/Test Symmetry with the training augmentations.
    """
    # Resize to match backbone input resolution
    image_np = cv2.resize(image_np, (224, 224))
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 2. Apply Adaptive Gaussian Thresholding
    binarized = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 3. Re-convert to RGB format for the ConvNeXt tensor layout
    rgb_ready = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    # 4. Standard ImageNet Normalization to match training
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (rgb_ready / 255.0 - mean) / std
    
    # 5. Reshape to PyTorch Tensor Format (C, H, W)
    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

def draw_label(draw, bbox, text):
    """Draws red bounding box and solid label tab."""
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    tab_rect = [xmin, ymin - text_h - 4, xmin + text_w + 4, ymin]
    draw.rectangle(tab_rect, fill="red")
    draw.text((xmin + 2, ymin - text_h - 2), text, font=font, fill="white")

def run_inference(image_path, bounding_boxes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ConvNeXt Model
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        print("Error: FAISS index or mapping CSV not found. Run build_index.py first.")
        return
    
    index = faiss.read_index(INDEX_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    original_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    
    print("Performing Font Inference with Adaptive Binarization...")
    
    for item in bounding_boxes:
        bbox = item['box']
        xmin, ymin, xmax, ymax = bbox
        
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(original_img.width, xmax), min(original_img.height, ymax)
        
        crop_img = original_img.crop((xmin, ymin, xmax, ymax))
        crop_np = np.array(crop_img)
        
        # Apply Adaptive Threshold Pipeline
        tensor = preprocess_inference_crop(crop_np).to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True):
                embedding = model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            
        vector_np = embedding.cpu().numpy().astype('float32')
        
        distances, indices = index.search(vector_np, 1)
        
        top1_idx = indices[0][0]
        confidence = distances[0][0] * 100 
        
        match_row = mapping_df[mapping_df['faiss_id'] == top1_idx].iloc[0]
        font_name = match_row['font_name']
        
        label_text = f"{font_name} ({confidence:.1f}%)"
        draw_label(draw, bbox, label_text)
        print(f"Matched Region {bbox} to {font_name} with confidence {confidence:.2f}%")

    output_path = "visual_result.png"
    original_img.save(output_path)
    print(f"\nSaved visual output to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image> [xmin,ymin,xmax,ymax] ...")
        sys.exit(1)
        
    img_path = sys.argv[1]
    boxes = []
    
    for arg in sys.argv[2:]:
        parts = arg.split(',')
        if len(parts) == 4:
            boxes.append({'box': (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))})
            
    if not boxes:
        img_temp = Image.open(img_path)
        boxes.append({'box': (0, 0, img_temp.width, img_temp.height)})
        
    run_inference(img_path, boxes)
