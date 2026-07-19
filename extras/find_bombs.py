import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Configuration
PROGRESS_FILE = "bomb_hunter_progress.txt"
CURRENT_FILE = "bomb_hunter_current.txt"
BLACKLIST_FILE = "bomb_blacklist.txt"

def main():
    print("==================================================")
    print("💣 FONT DECOMPRESSION BOMB HUNTER INITIALIZED 💣")
    print("==================================================")
    
    # 1. Auto-detect TTF directory
    kaggle_input_dir = Path("/kaggle/input")
    if kaggle_input_dir.exists():
        try:
            target_dir = None
            for d in kaggle_input_dir.rglob("ttf_files"):
                if d.is_dir():
                    target_dir = d
                    break
            if target_dir:
                TTF_DIR = str(target_dir)
            else:
                first_ttf = next(kaggle_input_dir.rglob("*.ttf"))
                TTF_DIR = str(first_ttf.parent)
        except StopIteration:
            TTF_DIR = "ttf_files"
    else:
        TTF_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\ttf_files_2\ttf_files"
        
    print(f"Scanning directory: {TTF_DIR}")
    
    if not os.path.exists(TTF_DIR):
        print("ERROR: TTF directory not found!")
        sys.exit(0)
        
    # Gather and SORT files
    all_files = list(Path(TTF_DIR).rglob("*.ttf")) + list(Path(TTF_DIR).rglob("*.otf"))
    all_files.sort()
    
    total_fonts = len(all_files)
    print(f"Found {total_fonts} total font files.")
    
    # Read current progress
    start_idx = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            try:
                start_idx = int(f.read().strip())
            except ValueError:
                start_idx = 0

    # 2. Check for Death Checkpoint
    if os.path.exists(CURRENT_FILE):
        with open(CURRENT_FILE, "r") as f:
            bomb_path = f.read().strip()
            
        print("\n" + "!" * 50)
        print("💥 DECOMPRESSION BOMB DETECTED FROM PREVIOUS RUN 💥")
        print(f"Malicious Font: {bomb_path}")
        print("!" * 50)
        
        # Kaggle input is read-only. We must append to a blacklist instead of deleting.
        with open(BLACKLIST_FILE, "a") as bf:
            bf.write(bomb_path + "\n")
        print(f"-> Added {Path(bomb_path).name} to {BLACKLIST_FILE}!")
        
        # Skip this bomb so we don't infinitely crash on it
        start_idx += 1
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(start_idx))
            
        os.remove(CURRENT_FILE)
        print("Resuming hunt...\n")
                
    if start_idx >= total_fonts:
        print("🎉 HUNT COMPLETE! All fonts verified clean. 🎉")
        if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
        sys.exit(0)
        
    print(f"Starting inspection at font index {start_idx} / {total_fonts}...\n")
    
    # 4. Sequential Scan Loop
    for idx in range(start_idx, total_fonts):
        font_path = all_files[idx]
        
        # WRITE DEATH MARKER BEFORE DOING ANYTHING DANGEROUS
        with open(CURRENT_FILE, "w") as f:
            f.write(str(font_path))
            f.flush()
            os.fsync(f.fileno()) 

        # DANGEROUS ZONE: Attempt to render the font
        try:
            font = ImageFont.truetype(str(font_path), 120)
            img = Image.new("RGB", (1000, 1000), "white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "AaBbCc0123", font=font, fill="black")
            
            del draw
            del font
            del img
            
        except Exception as e:
            pass
            
        # Survived! Delete death marker
        os.remove(CURRENT_FILE)
        
        # Save progress
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(idx + 1))
            
        if (idx + 1) % 500 == 0:
            print(f"Verified {idx + 1}/{total_fonts} fonts safe...")

    print("🎉 HUNT COMPLETE! All fonts verified clean. 🎉")
    if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
    sys.exit(0)

if __name__ == "__main__":
    main()
