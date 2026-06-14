import sys
import os
import torch
import faiss
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch.nn.functional as F

from train_virtual_epochs import FontEmbeddingModel

EMBEDDING_SIZE = 256
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

def get_inference_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def draw_label(draw, bbox, text):
    """Draws red bounding box and solid label tab."""
    xmin, ymin, xmax, ymax = bbox
    
    # Draw red bounding box
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    
    try:
        # Try to use arial for system text
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # Solid label tab above bounding box
    tab_rect = [xmin, ymin - text_h - 4, xmin + text_w + 4, ymin]
    draw.rectangle(tab_rect, fill="red")
    draw.text((xmin + 2, ymin - text_h - 2), text, font=font, fill="white")

def run_inference(image_path, bounding_boxes):
    """
    bounding_boxes: list of dicts [{'box': (xmin, ymin, xmax, ymax)}]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = FontEmbeddingModel(embedding_size=EMBEDDING_SIZE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Load FAISS & Mapping
    if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
        print("Error: FAISS index or mapping CSV not found. Run build_index.py first.")
        return
    
    index = faiss.read_index(INDEX_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    
    transform = get_inference_transform()
    
    # 3. Load Image
    original_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_img)
    
    print("Performing Font Inference...")
    
    for item in bounding_boxes:
        bbox = item['box']
        xmin, ymin, xmax, ymax = bbox
        
        # Ensure bounding box is within image
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(original_img.width, xmax), min(original_img.height, ymax)
        
        # Crop region
        crop_img = original_img.crop((xmin, ymin, xmax, ymax))
        tensor = transform(crop_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True):
                embedding = model(tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            
        vector_np = embedding.cpu().numpy().astype('float32')
        
        # Query FAISS for Top-1 Cosine Similarity match
        distances, indices = index.search(vector_np, 1)
        
        top1_idx = indices[0][0]
        # For normalized vectors, inner product matches cosine similarity
        # Cosine similarity is usually [-1, 1], convert to pseudo-percentage 0-100%
        confidence = distances[0][0] * 100 
        
        # Retrieve Font Name
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
        print("Example: python inference.py sample.jpg 100,100,300,200 400,100,500,200")
        sys.exit(1)
        
    img_path = sys.argv[1]
    boxes = []
    
    # Parse CLI bounding boxes
    for arg in sys.argv[2:]:
        parts = arg.split(',')
        if len(parts) == 4:
            boxes.append({'box': (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))})
            
    if not boxes:
        print("No valid bounding boxes provided. Providing a dummy evaluation on full image.")
        # If no boxes provided, evaluate on the whole image as a single box
        img_temp = Image.open(img_path)
        boxes.append({'box': (0, 0, img_temp.width, img_temp.height)})
        
    run_inference(img_path, boxes)
