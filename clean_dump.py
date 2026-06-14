import os
import hashlib
from pathlib import Path
from fontTools.ttLib import TTFont

# Configuration
# Please point DUMP_DIR to the directory containing your raw .ttf files
DUMP_DIR = "dataset" 

def get_file_hash(filepath):
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def has_basic_glyphs(font_path):
    """Check if the font contains basic A-Z and 0-9 glyphs."""
    try:
        # Load the font, disabling lazy loading to ensure header is fully parsed
        font = TTFont(font_path, lazy=False)
        cmap = font.getBestCmap()
        if not cmap:
            return False
        
        # Required ASCII ranges: 48-57 (0-9), 65-90 (A-Z)
        required_codepoints = list(range(48, 58)) + list(range(65, 91))
        
        for codepoint in required_codepoints:
            if codepoint not in cmap:
                return False
                
        return True
    except Exception as e:
        # Catch corruption or missing tables
        return False

def clean_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"Error: Directory '{dataset_dir}' not found.")
        print("Please update the DUMP_DIR variable to point to your TTF files.")
        return

    print(f"Starting data sanitization in {dataset_dir}...")
    
    seen_hashes = set()
    stats = {
        "total_files": 0,
        "deleted_zero_bytes": 0,
        "deleted_corrupt_or_missing_glyphs": 0,
        "deleted_duplicates": 0,
        "retained_files": 0
    }

    # Iterate through all .ttf files
    for filepath in dataset_path.rglob("*.ttf"):
        stats["total_files"] += 1
        
        if stats["total_files"] % 1000 == 0:
            print(f"Processed {stats['total_files']} files...")

        # 1. Check for 0 bytes
        try:
            if filepath.stat().st_size == 0:
                os.remove(filepath)
                stats["deleted_zero_bytes"] += 1
                continue
        except Exception as e:
            print(f"Error checking size for {filepath}: {e}")
            continue
            
        # 2. Check for corruption and basic glyphs
        if not has_basic_glyphs(filepath):
            try:
                os.remove(filepath)
                stats["deleted_corrupt_or_missing_glyphs"] += 1
            except Exception as e:
                print(f"Error deleting {filepath}: {e}")
            continue
            
        # 3. Deduplication via MD5 hash
        file_hash = get_file_hash(filepath)
        if not file_hash:
            continue
            
        if file_hash in seen_hashes:
            try:
                os.remove(filepath)
                stats["deleted_duplicates"] += 1
            except Exception as e:
                print(f"Error deleting duplicate {filepath}: {e}")
            continue
            
        seen_hashes.add(file_hash)
        stats["retained_files"] += 1

    print("\n--- Sanitization Complete ---")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Deleted (0 bytes): {stats['deleted_zero_bytes']}")
    print(f"Deleted (Corrupt/Missing Glyphs): {stats['deleted_corrupt_or_missing_glyphs']}")
    print(f"Deleted (Duplicates): {stats['deleted_duplicates']}")
    print(f"Retained healthy files: {stats['retained_files']}")

if __name__ == "__main__":
    clean_dataset(r"E:\New folder\coding_arc\Font_Identifier_AI\ttf_files")
