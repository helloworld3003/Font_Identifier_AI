import os
import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SOURCE_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\1001_fonts"
TARGET_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\ttf_files"
MAX_WORKERS = 16  # Multi-threading for 16x speedup!

def check_dependencies():
    """Check if otf2ttf is installed."""
    try:
        subprocess.run(["otf2ttf", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
        
    except FileNotFoundError:
        print("ERROR: 'otf2ttf' is not installed.")
        print("Please install it by running: pip install otf2ttf")
        return False

def process_single_otf(otf):
    ttf_name = otf.stem + ".ttf"
    target_path = os.path.join(TARGET_DIR, ttf_name)
    
    if os.path.exists(target_path):
        return (True, otf.name, "Skipped, TTF already exists")
        
    try:
        # Run otf2ttf to mathematically convert cubic (CFF) outlines to quadratic (TrueType)
        result = subprocess.run(
            ["otf2ttf", str(otf), "-o", target_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(target_path):
            return (True, otf.name, "Converted")
        else:
            return (False, otf.name, result.stderr.strip())
            
    except Exception as e:
        return (False, otf.name, str(e))

def convert_fonts():
    if not check_dependencies():
        return
        
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. First, copy all existing TTFs directly to the target directory
    print(f"--- Scanning & Copying existing TTF files ---")
    ttf_files = list(Path(SOURCE_DIR).rglob("*.ttf"))
    copied_count = 0
    for ttf in ttf_files:
        target_path = os.path.join(TARGET_DIR, ttf.name)
        if not os.path.exists(target_path):
            shutil.copy2(ttf, target_path)
            copied_count += 1
            
    print(f"Copied {copied_count} new TTF files to the target directory.")
            
    # 2. Find all OTF files and convert them via ThreadPool
    otf_files = list(Path(SOURCE_DIR).rglob("*.otf"))
    print(f"\n--- Multi-Threaded Conversion Started ({len(otf_files)} OTF files found) ---")
    
    if not otf_files:
        print("No .otf files found to convert.")
        return
        
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the thread pool
        futures = {executor.submit(process_single_otf, otf): otf for otf in otf_files}
        
        # Process results as they finish concurrently
        for future in as_completed(futures):
            success, name, msg = future.result()
            
            if success:
                if "Skipped" in msg:
                    skip_count += 1
                else:
                    success_count += 1
                    print(f"[OK] Converted {name}")
            else:
                fail_count += 1
                # Only print the first line of the error to avoid massive traceback spam in terminal
                short_err = msg.split('\n')[-1] if msg else "Unknown Error"
                print(f"[!] Failed {name}: {short_err}")
                
    print("\n--- Conversion Complete ---")
    print(f"Successfully converted (New): {success_count}")
    print(f"Skipped (Already existed): {skip_count}")
    print(f"Failed conversions: {fail_count}")
    print(f"All TTF files are stored in: {TARGET_DIR}")

if __name__ == "__main__":
    convert_fonts()
