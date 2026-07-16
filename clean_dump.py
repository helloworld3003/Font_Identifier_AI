import os
import hashlib
from pathlib import Path
from fontTools.ttLib import TTFont
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DUMP_DIR = r"E:\New folder\coding_arc\Font_Identifier_AI\ttf_files"

def process_single_font(filepath):
    """
    Worker function to process a single font.
    Checks size, glyphs, and calculates MD5.
    Returns: (filepath, status, hash_val)
    """
    try:
        # 1. Check for 0 bytes
        if filepath.stat().st_size == 0:
            os.remove(filepath)
            return (filepath, "zero_bytes", None)
            
        # 2. Check for corruption and basic glyphs
        try:
            # Load the font, disabling lazy loading to ensure header is fully parsed
            font = TTFont(filepath, lazy=False)
            cmap = font.getBestCmap()
            if not cmap:
                font.close()
                os.remove(filepath)
                return (filepath, "missing_glyphs", None)
            
            # Required ASCII ranges: 48-57 (0-9), 65-90 (A-Z)
            required_codepoints = list(range(48, 58)) + list(range(65, 91))
            for codepoint in required_codepoints:
                if codepoint not in cmap:
                    font.close()
                    os.remove(filepath)
                    return (filepath, "missing_glyphs", None)
                    
            font.close()
        except Exception:
            try:
                os.remove(filepath)
            except:
                pass
            return (filepath, "corrupt", None)
            
        # 3. Calculate Hash for Deduplication
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            # Read entire file at once since TTFs are relatively small (usually < 1MB)
            hasher.update(f.read())
            
        return (filepath, "healthy", hasher.hexdigest())
        
    except Exception as e:
        return (filepath, f"error: {str(e)}", None)

def clean_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"Error: Directory '{dataset_dir}' not found.")
        return

    print(f"Starting Multi-Core Data Sanitization in {dataset_dir}...")
    
    seen_hashes = set()
    stats = {
        "total_files": 0,
        "deleted_zero_bytes": 0,
        "deleted_corrupt_or_missing_glyphs": 0,
        "deleted_duplicates": 0,
        "retained_files": 0
    }

    # Gather all TTF files
    ttf_files = list(dataset_path.rglob("*.ttf"))
    total = len(ttf_files)
    print(f"Found {total} TTF files. Firing up all CPU cores...")

    # Use ProcessPoolExecutor to bypass Python's GIL and utilize all CPU cores
    max_cores = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_font, p): p for p in ttf_files}
        
        # Process results as they finish
        for i, future in enumerate(as_completed(futures), 1):
            filepath, status, file_hash = future.result()
            stats["total_files"] += 1
            
            if status == "zero_bytes":
                stats["deleted_zero_bytes"] += 1
            elif status in ["corrupt", "missing_glyphs"]:
                stats["deleted_corrupt_or_missing_glyphs"] += 1
            elif status == "healthy":
                # Handle Deduplication in the main thread to safely update the shared set
                if file_hash in seen_hashes:
                    try:
                        os.remove(filepath)
                        stats["deleted_duplicates"] += 1
                    except:
                        pass
                else:
                    seen_hashes.add(file_hash)
                    stats["retained_files"] += 1
                    
            if i % 2500 == 0:
                print(f"Processed {i}/{total} files...")

    print("\n--- Multi-Core Sanitization Complete ---")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Deleted (0 bytes): {stats['deleted_zero_bytes']}")
    print(f"Deleted (Corrupt/Missing Glyphs): {stats['deleted_corrupt_or_missing_glyphs']}")
    print(f"Deleted (Duplicates): {stats['deleted_duplicates']}")
    print(f"Retained healthy, unique files: {stats['retained_files']}")

if __name__ == "__main__":
    # Wrap execution for Windows multiprocessing safety
    clean_dataset(DUMP_DIR)
