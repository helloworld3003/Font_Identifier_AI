import argparse
import logging
import warnings

# Suppress some noisy warnings from easyocr/pytorch
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr
import faiss

import torch
from torchvision import transforms

# Import the model architecture from Phase 2
from train_local_subset import MetricResNet18

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def draw_inference_results(original_image_path, detections, output_path="visual_result.png"):
    """
    Takes the original image and draws bounding boxes with font predictions.
    
    Expected format for 'detections':
    [
        {"box": [x_min, y_min, x_max, y_max], "font_name": "Roboto-Bold", "confidence": 0.94},
        {"box": [x_min, y_min, x_max, y_max], "font_name": "PlayfairDisplay", "confidence": 0.88}
    ]
    """
    # Open image and convert to RGBA to support colored drawing
    img = Image.open(original_image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # Try to load a default font for the annotation labels
    try:
        # If running on Windows, this path usually works. Adjust if on Mac/Linux.
        label_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        label_font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        label_text = f"{det['font_name']} ({det['confidence']*100:.1f}%)"
        
        # 1. Draw the bounding box (Red outline, 3 pixels thick)
        draw.rectangle(box, outline=(255, 0, 0, 255), width=3)
        
        # 2. Calculate label background size for readability
        # (Using a solid background behind the text so it doesn't get lost on messy images)
        text_bbox = draw.textbbox((0, 0), label_text, font=label_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        label_bg_box = [box[0], box[1] - text_height - 6, box[0] + text_width + 6, box[1]]
        
        # 3. Draw label background (Solid Red)
        draw.rectangle(label_bg_box, fill=(255, 0, 0, 255))
        
        # 4. Draw the actual text label (White text)
        draw.text((box[0] + 3, box[1] - text_height - 4), label_text, fill=(255, 255, 255, 255), font=label_font)

    # Save the final annotated image
    img.save(output_path)
    print(f"\n[SUCCESS] Visual output saved to: {output_path}")

def format_inference_crop(crop_img: Image.Image, target_size=(256, 64)) -> Image.Image:
    """
    Resizes and pads a tightly cropped text image to match the fixed-canvas 
    distribution used during Phase 1 training, preserving aspect ratio.
    """
    target_width, target_height = target_size
    
    # In training, font size was ~32-56. Let's target a text height of 44px.
    desired_text_height = 44
    aspect_ratio = crop_img.width / crop_img.height
    new_width = int(desired_text_height * aspect_ratio)
    new_height = desired_text_height
    
    # Scale down if it exceeds canvas width (minus padding)
    max_width = target_width - 20
    if new_width > max_width:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
        
    # Resize the tight crop
    resized_crop = crop_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create the white canvas (training background)
    canvas = Image.new('RGB', target_size, color='white')
    
    # Paste the crop centered on the canvas
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    canvas.paste(resized_crop, (x, y))
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Font Identifier Inference")
    parser.add_argument("image_path", type=str, help="Path to the raw test image containing text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of font matches to return per text region")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------
    # 1. LOAD MODEL
    # -----------------------------
    model_path = "best_metric_model.pth"
    if not os.path.exists(model_path):
        if os.path.exists("poc_metric_model.pth"):
            model_path = "poc_metric_model.pth"
            logger.warning(f"Production model not found. Falling back to {model_path}.")
            
    try:
        model = MetricResNet18(embedding_size=128)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Ensure Phase 2 was completed. Error: {e}")
        return
    
    # -----------------------------
    # 2. LOAD FAISS INDEX & MAPPING
    # -----------------------------
    index_path = "font_embeddings.index"
    mapping_path = "faiss_mapping.csv"
    try:
        index = faiss.read_index(index_path)
        mapping_df = pd.read_csv(mapping_path)
    except Exception as e:
        logger.error(f"Failed to load FAISS index or mapping. Ensure Phase 3 was completed. Error: {e}")
        return
        
    # Standard transform used during training & indexing
    # Notice we REMOVED the Resize((64, 256)) from here, because we handle it 
    # intelligently in format_inference_crop() now!
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # -----------------------------
    # 3. OCR TO DETECT TEXT BOUNDING BOXES
    # -----------------------------
    logger.info(f"Loading image '{args.image_path}' and running text detection via EasyOCR...")
    try:
        reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
        # width_ths=0.1 prevents EasyOCR from merging adjacent words into a single line.
        # This is critical if the user mixes different fonts on the same line!
        ocr_results = reader.readtext(args.image_path, width_ths=0.1)
    except Exception as e:
        logger.error(f"Failed to read image or run OCR. Error: {e}")
        return
        
    if not ocr_results:
        logger.warning("No text detected in the image.")
        return
        
    pil_img = Image.open(args.image_path).convert('RGB')
    
    # -----------------------------
    # 4. PROCESS EACH TEXT CROP
    # -----------------------------
    logger.info(f"Found {len(ocr_results)} distinct text regions. Extracting fonts...")
    print("\n" + "=" * 60)
    
    detections = []
    
    for i, (bbox, text, conf) in enumerate(ocr_results):
        # Extract bounding box coordinates
        # bbox is a list of 4 points: [top_left, top_right, bottom_right, bottom_left]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        left, right = int(min(xs)), int(max(xs))
        top, bottom = int(min(ys)), int(max(ys))
        
        # Add a tiny bit of padding around the crop for safety
        pad = 2
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(pil_img.width, right + pad)
        bottom = min(pil_img.height, bottom + pad)
        
        # Crop the image to just the text
        crop = pil_img.crop((left, top, right, bottom))
        
        # Intelligently pad and resize the crop to match training data
        formatted_crop = format_inference_crop(crop, target_size=(256, 64))
        
        # Prepare for the model
        tensor = transform(formatted_crop).unsqueeze(0).to(device)
        
        # -----------------------------
        # 5. INFERENCE & QUERY
        # -----------------------------
        with torch.no_grad():
            embed = model(tensor)
            embed = torch.nn.functional.normalize(embed, p=2, dim=1)
            embed_np = embed.cpu().numpy().astype('float32')
            
        distances, indices = index.search(embed_np, args.top_k)
        
        print(f"REGION {i+1} | Detected Text: '{text}' (OCR Confidence: {conf:.2f})")
        print(f"Predicted Fonts (Top {args.top_k}):")
        
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            # FAISS Inner Product scores are cosine similarities [-1.0, 1.0]
            # Convert to a percentage for easier reading
            similarity_pct = (dist + 1.0) / 2.0 * 100.0 
            
            # Retrieve the font metadata
            row = mapping_df[mapping_df['faiss_id'] == idx].iloc[0]
            
            # Record the Top-1 prediction for the visual annotation
            if rank == 0:
                detections.append({
                    "box": [left, top, right, bottom],
                    "font_name": row['font_name'],
                    "confidence": (dist + 1.0) / 2.0
                })
            
            print(f"  {rank+1}. {row['font_name']} (Similarity: {similarity_pct:.1f}%)")
        print("-" * 60)
        
    # Generate the visual output
    draw_inference_results(args.image_path, detections, output_path="visual_result.png")
    
    logger.info("Inference complete.")

if __name__ == "__main__":
    main()
