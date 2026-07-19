import os
import sys
import torch
import faiss
import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torch.nn.functional as F

from model import ConvNeXtFontEncoder

EMBEDDING_SIZE = 512
MODEL_PATH = "best_model.pth"
INDEX_PATH = "font_embeddings.index"
MAPPING_PATH = "faiss_mapping.csv"

CANONICAL_STRINGS = ["AaBbCc", "xyz123", "0OIl", "gjpqy", "Test 00"]

# ==========================================
# 1. PREPROCESSING FOR IMAGES
# ==========================================
def preprocess_inference_crop(image_np):
    image_np = cv2.resize(image_np, (256, 64))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    binarized = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    rgb_ready = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (rgb_ready / 255.0 - mean) / std
    
    tensor = torch.tensor(normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return tensor

def auto_detect_text_boxes(image_path):
    img = cv2.imread(image_path)
    if img is None: return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 4))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 10:
            boxes.append({'box': (x, y, x + w, y + h)})
    return sorted(boxes, key=lambda b: b['box'][1])

def draw_label(draw, bbox, text):
    xmin, ymin, xmax, ymax = bbox
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    draw.rectangle([xmin, ymin - text_h - 4, xmin + text_w + 4, ymin], fill="red")
    draw.text((xmin + 2, ymin - text_h - 2), text, font=font, fill="white")

# ==========================================
# 2. PREPROCESSING FOR FONTS
# ==========================================
def render_string(ttf_path, text, target_w=256, target_h=64, font_size=60):
    try:
        font = ImageFont.truetype(str(ttf_path), font_size)
        temp_image = Image.new("RGB", (1, 1), "white")
        temp_draw = ImageDraw.Draw(temp_image)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        
        image = Image.new("RGB", (text_w, text_h), "white")
        draw = ImageDraw.Draw(image)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill="black")
        
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
        return Image.new("RGB", (target_w, target_h), "white")

def resolve_local_font_path(kaggle_path):
    """Kaggle paths won't exist on Windows, so we search locally for the filename"""
    if os.path.exists(kaggle_path):
        return kaggle_path
    
    filename = Path(kaggle_path).name
    # Search locally in ttf_files dir
    for p in Path("ttf_files").rglob("*"):
        if p.name == filename:
            return str(p)
    return None

def format_font_name(row):
    if 'full_name' in row and pd.notna(row['full_name']) and row['full_name'] != "Unknown":
        return f"{row['full_name']} (Family: {row['font_family']})"
    return row['font_name']

# ==========================================
# 3. CORE INFERENCE LOGIC
# ==========================================
class FontIdentifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ConvNeXt on {self.device}...")
        self.model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        print("Loading FAISS Index...")
        self.index = faiss.read_index(INDEX_PATH)
        self.mapping_df = pd.read_csv(MAPPING_PATH)

    def identify_image(self, image_path, top_k=5):
        print(f"\n[INTERACTIVE MODE] Analyzing {image_path}...")
        multi = input("Does this image contain MULTIPLE different fonts? (y/n) [Default: n]: ").strip().lower()
        
        if multi == 'y':
            boxes = auto_detect_text_boxes(image_path)
            if not boxes:
                img = Image.open(image_path)
                boxes = [{'box': (0, 0, img.width, img.height)}]
        else:
            img = Image.open(image_path)
            boxes = [{'box': (0, 0, img.width, img.height)}]
            
        original_img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(original_img)
        
        crops_tensors, valid_bboxes = [], []
        for item in boxes:
            xmin, ymin, xmax, ymax = item['box']
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(original_img.width, xmax), min(original_img.height, ymax)
            
            crop_np = np.array(original_img.crop((xmin, ymin, xmax, ymax)))
            crops_tensors.append(preprocess_inference_crop(crop_np))
            valid_bboxes.append(item['box'])
            
        batch_tensor = torch.cat(crops_tensors, dim=0).to(self.device)
        with torch.no_grad():
            embeddings = F.normalize(self.model(batch_tensor), p=2, dim=1)
            
        distances, indices = self.index.search(embeddings.cpu().numpy().astype('float32'), top_k)
        
        for idx, bbox in enumerate(valid_bboxes):
            print(f"\n--- Detected Text Region {bbox} ---")
            best_row = self.mapping_df.iloc[indices[idx][0]]
            best_name = format_font_name(best_row)
            best_conf = ((distances[idx][0] + 1) / 2) * 100
            draw_label(draw, bbox, f"{best_name} ({best_conf:.1f}%)")
            
            # --- Build the Visualization Board for this Bounding Box ---
            board_w = 1200
            row_h = 180
            viz_board = Image.new("RGB", (board_w, row_h * (top_k + 1)), "white")
            draw_board = ImageDraw.Draw(viz_board)
            try: default_font = ImageFont.truetype("arial.ttf", 26)
            except: default_font = ImageFont.load_default()
            
            def draw_image_row(y_offset, title, font_to_render=None, is_query=False, query_img=None, custom_text=""):
                draw_board.text((30, y_offset + 20), title, font=default_font, fill="#2563eb" if is_query else "#333333")
                if is_query and query_img is not None:
                    crop_w, crop_h = query_img.size
                    # Scale crop so it fits nicely inside the 180px row
                    scale = min((board_w - 60) / crop_w, 100 / crop_h)
                    new_w = max(1, int(crop_w * scale))
                    new_h = max(1, int(crop_h * scale))
                    resized_crop = query_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    viz_board.paste(resized_crop, (30, y_offset + 60))
                else:
                    try:
                        render_font = ImageFont.truetype(font_to_render, 64)
                        draw_board.text((30, y_offset + 70), custom_text, font=render_font, fill="black")
                    except Exception:
                        draw_board.text((30, y_offset + 70), "[Font Rendering Failed]", font=default_font, fill="red")
                draw_board.line([(0, y_offset + row_h - 1), (board_w, y_offset + row_h - 1)], fill="#e5e7eb", width=2)
                
            xmin, ymin, xmax, ymax = bbox
            crop_img = original_img.crop((xmin, ymin, xmax, ymax))
            
            print(f"\n[INTERACTIVE MODE] Processing Box {idx+1} at {bbox}...")
            user_input = input(f"What text is written in this image? (Press Enter for default): ").strip()
            display_text = user_input if user_input else "Sphinx of black quartz, judge my vow."
            
            draw_image_row(0, f"QUERY IMAGE CROP {idx+1}", is_query=True, query_img=crop_img)
            
            y_cursor = row_h
            for i in range(top_k):
                confidence = ((distances[idx][i] + 1) / 2) * 100
                match_row = self.mapping_df.iloc[indices[idx][i]]
                font_name = format_font_name(match_row)
                font_path = match_row['font_path']
                local_path = resolve_local_font_path(font_path)
                
                title = f"MATCH #{i+1} ({confidence:.2f}%): {font_name}"
                if not local_path:
                    title += " [TTF NOT FOUND LOCALLY]"
                
                draw_image_row(y_cursor, title, local_path, custom_text=display_text)
                y_cursor += row_h
                print(f"  Match #{i+1} ({confidence:.2f}%): {font_name}")
                
            out_name = f"visual_result_image_box_{idx+1}.png"
            viz_board.save(out_name)
            print(f"Saved visualization board for Box {idx+1} to {out_name}")
            
        original_img.save("visual_result_image.png")
        print("\nSaved overall bounding box image to visual_result_image.png")

    def identify_font(self, font_path, top_k=5):
        print(f"Analyzing structure of Font: {Path(font_path).name}")
        
        # 1. Render exactly like the training loop
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        tensors = []
        for text in CANONICAL_STRINGS:
            img = render_string(font_path, text)
            tensors.append(transform(img))
            
        batch = torch.stack(tensors).unsqueeze(0).to(self.device) # (1, 5, 3, 64, 256)
        batch = batch.view(5, 3, 64, 256)
        
        # 2. Get Average Embedding
        with torch.no_grad():
            embeddings = self.model(batch).view(1, 5, EMBEDDING_SIZE)
            avg_emb = F.normalize(torch.mean(embeddings, dim=1), p=2, dim=1)
            
        # 3. Search FAISS (We get top_k + 1 in case the query font itself is the #1 match)
        distances, indices = self.index.search(avg_emb.cpu().numpy().astype('float32'), top_k + 1)
        
        # 4. Generate Visualization Board
        board_w = 1200
        row_h = 180
        viz_board = Image.new("RGB", (board_w, row_h * (top_k + 1)), "white")
        draw = ImageDraw.Draw(viz_board)
        
        try: default_font = ImageFont.truetype("arial.ttf", 26)
        except: default_font = ImageFont.load_default()
        
        def draw_row(y_offset, title, font_to_render, is_query=False):
            # Title
            draw.text((30, y_offset + 20), title, font=default_font, fill="#2563eb" if is_query else "#333333")
            # Rendered Sample
            try:
                render_font = ImageFont.truetype(font_to_render, 64)
                # Some fonts have large ascenders/descenders, so we give them plenty of vertical breathing room
                draw.text((30, y_offset + 70), "Sphinx of black quartz, judge my vow.", font=render_font, fill="black")
            except Exception:
                draw.text((30, y_offset + 70), "[Font Rendering Failed]", font=default_font, fill="red")
            # Separator
            draw.line([(0, y_offset + row_h - 1), (board_w, y_offset + row_h - 1)], fill="#e5e7eb", width=2)
            
        # Draw Query Row
        draw_row(0, f"QUERY FONT: {Path(font_path).name}", font_path, is_query=True)
        
        # Draw Matches
        y_cursor = row_h
        drawn_count = 0
        for i in range(top_k + 1):
            if drawn_count >= top_k: break
            
            score = distances[0][i]
            faiss_id = indices[0][i]
            match_row = self.mapping_df.iloc[faiss_id]
            
            # Skip if the closest match is literally the exact same file we queried!
            if match_row['font_name'] == Path(font_path).stem and score > 0.99:
                continue
                
            confidence = ((score + 1) / 2) * 100
            local_path = resolve_local_font_path(match_row['font_path'])
            
            font_name = format_font_name(match_row)
            title = f"MATCH #{drawn_count + 1} ({confidence:.2f}%): {font_name}"
            if not local_path:
                title += " [TTF NOT FOUND LOCALLY]"
                
            draw_row(y_cursor, title, local_path if local_path else font_path)
            y_cursor += row_h
            drawn_count += 1
            print(title)

        viz_board.save("visual_result_fonts.png")
        print("\nSaved similar fonts visual comparison to visual_result_fonts.png!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Font Identifier AI")
    parser.add_argument("--image", type=str, help="Detect fonts inside an image")
    parser.add_argument("--font", type=str, help="Find visually similar fonts to a TTF/OTF file")
    
    args = parser.parse_args()
    
    if not args.image and not args.font:
        parser.print_help()
        sys.exit(1)
        
    app = FontIdentifier()
    if args.image:
        app.identify_image(args.image)
    if args.font:
        app.identify_font(args.font, top_k=5)
