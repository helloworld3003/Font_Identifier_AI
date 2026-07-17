import sys
import os
import torch
import faiss
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F

from train_tpu_8core import ConvNeXtFontEncoder

EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

def preprocess_inference_crop(image_np):
    """
    Converts a real-world RGB crop into a binarized, clean image
    to remove backgrounds, lighting variation, and shadows.
    Ensures Train/Test Symmetry with the training augmentations.
    """
    # Resize to match backbone input resolution exactly (W=256, H=64)
    image_np = cv2.resize(image_np, (256, 64))
    
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

def auto_detect_text_boxes(image_path):
    """
    Automatically detects text regions in the image using adaptive binarization 
    and morphological dilation to merge character contours into horizontal line segments.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize with threshold inversion (text is white on black background for morphological ops)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Kernel size has wide width to merge characters/words horizontally 
    # but thin height so lines do not merge vertically.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 4))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out very small boxes to avoid noise
        if w > 15 and h > 10:
            boxes.append({'box': (x, y, x + w, y + h)})
            
    # Sort bounding boxes top-to-bottom
    boxes = sorted(boxes, key=lambda b: b['box'][1])
    return boxes

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
    
    crops_tensors = []
    valid_bboxes = []
    
    for item in bounding_boxes:
        bbox = item['box']
        xmin, ymin, xmax, ymax = bbox
        
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(original_img.width, xmax), min(original_img.height, ymax)
        
        crop_img = original_img.crop((xmin, ymin, xmax, ymax))
        crop_np = np.array(crop_img)
        
        # Apply Adaptive Threshold Pipeline
        tensor = preprocess_inference_crop(crop_np) # Shape: (1, 3, 64, 256)
        crops_tensors.append(tensor)
        valid_bboxes.append(bbox)
        
    if not crops_tensors:
        print("No valid bounding boxes to run inference on.")
        return

    # Stack all tensors into a single batch tensor: shape (N, 3, 64, 256)
    batch_tensor = torch.cat(crops_tensors, dim=0).to(device)
    
    with torch.no_grad():
        embeddings = model(batch_tensor)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
    vectors_np = embeddings.cpu().numpy().astype('float32')
    
    # Perform batched search in FAISS
    distances, indices = index.search(vectors_np, 1)
    
    for idx, bbox in enumerate(valid_bboxes):
        top1_idx = indices[idx][0]
        # Calculate confidence metric
        confidence = distances[idx][0] * 100 
        
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
        print("No bounding boxes provided. Running auto-detector to locate text regions...")
        boxes = auto_detect_text_boxes(img_path)
        if not boxes:
            print("Auto-detector found no text regions. Defaulting to full image.")
            img_temp = Image.open(img_path)
            boxes.append({'box': (0, 0, img_temp.width, img_temp.height)})
        else:
            print(f"Auto-detected {len(boxes)} text regions.")
        
    run_inference(img_path, boxes)
