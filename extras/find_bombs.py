import os
import sys
import time
import math
import multiprocessing as mp
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BLACKLIST_FILE = "bomb_blacklist.txt"

def worker_fn(worker_id, chunk_files):
    """
    Isolated worker that sequentially processes a chunk of files.
    If it hits a bomb, the OS kills this specific process via SIGKILL.
    """
    PROGRESS_FILE = f"worker_{worker_id}_progress.txt"
    CURRENT_FILE = f"worker_{worker_id}_current.txt"
    
    start_idx = 0
    
    # Check if this worker died previously. If so, skip the bomb that killed it.
    if os.path.exists(CURRENT_FILE):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                try:
                    start_idx = int(f.read().strip())
                except ValueError:
                    start_idx = 0
        
        # We died on start_idx. The Orchestrator already blacklisted it. Skip it!
        start_idx += 1
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write(str(start_idx))
            
        for _ in range(10):
            try:
                if os.path.exists(CURRENT_FILE):
                    os.remove(CURRENT_FILE)
                break
            except Exception:
                time.sleep(0.01)
    else:
        # Normal resume
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                try:
                    start_idx = int(f.read().strip())
                except ValueError:
                    start_idx = 0

    for idx in range(start_idx, len(chunk_files)):
        font_path = chunk_files[idx]
        
        # WRITE DEATH MARKER
        with open(CURRENT_FILE, "w", encoding="utf-8") as f:
            f.write(f"{idx}|{font_path}")
            f.flush()
            os.fsync(f.fileno())

        # DANGEROUS ZONE: Attempt to render the font
        try:
            font = ImageFont.truetype(str(font_path), 120)
            img = Image.new("RGB", (1000, 1000), "white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "AaBbCc0123", font=font, fill="black")
            del draw, font, img
        except Exception:
            # Structurally corrupt (but non-lethal) fonts are caught here.
            pass
            
        # Survived! Delete death marker
        for _ in range(10):
            try:
                if os.path.exists(CURRENT_FILE):
                    os.remove(CURRENT_FILE)
                break
            except Exception:
                time.sleep(0.01)
        
        # Save progress
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write(str(idx + 1))
            
        if (idx + 1) % 500 == 0:
            print(f"[Worker {worker_id}] Verified {idx + 1}/{len(chunk_files)} fonts...")

    # Worker finished its entire chunk successfully!
    if os.path.exists(PROGRESS_FILE): 
        os.remove(PROGRESS_FILE)
    sys.exit(0)


def main():
    print("==================================================")
    print("🚀 MULTIPROCESSING BOMB HUNTER ORCHESTRATOR 🚀")
    print("==================================================")
    
    mp.set_start_method('spawn', force=True)
    
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
        TTF_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\ttf_files"
        
    print(f"Scanning directory: {TTF_DIR}")
    
    if not os.path.exists(TTF_DIR):
        print("ERROR: TTF directory not found!")
        sys.exit(0)
        
    # Gather and SORT files
    all_files = list(Path(TTF_DIR).rglob("*.ttf")) + list(Path(TTF_DIR).rglob("*.otf"))
    all_files.sort()
    
    # Read existing blacklist to avoid scanning known bombs
    blacklist = set()
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, "r", encoding="utf-8") as f:
            blacklist = set([line.strip() for line in f.readlines()])
            
    clean_files = [f for f in all_files if str(f) not in blacklist]
    total_fonts = len(clean_files)
    
    print(f"Found {len(all_files)} total fonts. ({len(blacklist)} already blacklisted). Testing {total_fonts} fonts.")
    
    if total_fonts == 0:
        print("🎉 All fonts processed!")
        sys.exit(0)

    # 2. Divide into Chunks
    num_workers = min(mp.cpu_count(), 8)
    chunk_size = math.ceil(total_fonts / num_workers)
    
    chunks = []
    for i in range(0, total_fonts, chunk_size):
        chunks.append(clean_files[i:i + chunk_size])
        
    print(f"Divided {total_fonts} fonts into {len(chunks)} chunks across {num_workers} CPU cores.")
    
    # 3. Start Orchestrator Loop
    processes = []
    for wid in range(len(chunks)):
        p = mp.Process(target=worker_fn, args=(wid, chunks[wid]))
        p.start()
        processes.append((wid, p))
        
    while len(processes) > 0:
        new_processes = []
        for wid, p in processes:
            p.join(timeout=0.1) # Non-blocking check
            
            if not p.is_alive():
                exitcode = p.exitcode
                if exitcode != 0:
                    # 💥 WORKER CRASHED! (Likely SIGKILL OOM)
                    CURRENT_FILE = f"worker_{wid}_current.txt"
                    if os.path.exists(CURRENT_FILE):
                        with open(CURRENT_FILE, "r", encoding="utf-8") as f:
                            data = f.read().strip()
                            if "|" in data:
                                idx, bomb_path = data.split("|", 1)
                                print("\n" + "!" * 60)
                                print(f"💥 BOMB DETECTED BY WORKER {wid}! The OS assassinated it.")
                                print(f"Malicious Font: {bomb_path}")
                                print("!" * 60)
                                
                                # Add to blacklist
                                with open(BLACKLIST_FILE, "a", encoding="utf-8") as bf:
                                    bf.write(bomb_path + "\n")
                                print(f"-> Added {Path(bomb_path).name} to {BLACKLIST_FILE}!")
                                
                    # RESPAWN WORKER
                    print(f"-> Respawning Worker {wid} to resume its chunk...")
                    new_p = mp.Process(target=worker_fn, args=(wid, chunks[wid]))
                    new_p.start()
                    new_processes.append((wid, new_p))
                else:
                    # Worker finished successfully!
                    print(f"✅ Worker {wid} has completed its chunk!")
            else:
                # Still running normally
                new_processes.append((wid, p))
                
        processes = new_processes
        time.sleep(1) # Prevent 100% CPU usage in orchestrator
        
    print("\n🎉 HUNT COMPLETE! All workers verified their chunks as safe. 🎉")
    sys.exit(0)

if __name__ == "__main__":
    main()
