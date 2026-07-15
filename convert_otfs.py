import os
import subprocess
from pathlib import Path
import shutil

# Configuration
SOURCE_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\1001_fonts"
TARGET_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\ttf_files"

def check_dependencies():
    """Check if otf2ttf is installed."""
    try:
        subprocess.run(["otf2ttf", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        print("ERROR: 'otf2ttf' is not installed.")
        print("Please install it by running: pip install otf2ttf")
        return False

def convert_fonts():
    if not check_dependencies():
        return
        
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. First, copy all existing TTFs directly to the target directory
    print("--- Copying existing TTF files ---")
    ttf_files = list(Path(SOURCE_DIR).rglob("*.ttf"))
    for ttf in ttf_files:
        target_path = os.path.join(TARGET_DIR, ttf.name)
        if not os.path.exists(target_path):
            shutil.copy2(ttf, target_path)
            print(f"Copied: {ttf.name}")
            
    # 2. Find all OTF files and convert them
    print("\n--- Converting OTF to TTF ---")
    otf_files = list(Path(SOURCE_DIR).rglob("*.otf"))
    
    if not otf_files:
        print("No .otf files found to convert.")
        return
        
    success_count = 0
    fail_count = 0
    
    for otf in otf_files:
        # Create target TTF filename
        ttf_name = otf.stem + ".ttf"
        target_path = os.path.join(TARGET_DIR, ttf_name)
        
        if os.path.exists(target_path):
            print(f"Skipping {otf.name}, TTF already exists.")
            continue
            
        print(f"Converting {otf.name} -> {ttf_name}...")
        try:
            # Run otf2ttf to mathematically convert cubic (CFF) outlines to quadratic (TrueType)
            result = subprocess.run(
                ["otf2ttf", str(otf), "-o", target_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(target_path):
                success_count += 1
            else:
                print(f"  [!] Failed to convert {otf.name}: {result.stderr.strip()}")
                fail_count += 1
                
        except Exception as e:
            print(f"  [!] Exception during conversion: {e}")
            fail_count += 1
            
    print("\n--- Conversion Complete ---")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {fail_count}")
    print(f"All TTF files are stored in: {TARGET_DIR}")

if __name__ == "__main__":
    convert_fonts()
