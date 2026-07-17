import os
import csv
import multiprocessing
from pathlib import Path
from fontTools.ttLib import TTFont
from tqdm import tqdm

# Configuration
TTF_DIR = "ttf_files_2/ttf_files"
OUTPUT_CSV = "fonts_metadata.csv"

def get_font_metadata(font_path_str):
    """Worker function to extract metadata from a single TTF file."""
    font_path = Path(font_path_str)
    metadata = {
        "file_name": font_path.name,
        "file_path": str(font_path),
        "file_size_bytes": font_path.stat().st_size if font_path.exists() else 0,
        "font_family": "",
        "font_subfamily": "",
        "full_name": "",
        "postscript_name": ""
    }
    
    try:
        # Load font lazily to save memory
        font = TTFont(font_path, lazy=True)
        if 'name' in font:
            for record in font['name'].names:
                # Common Name IDs: 1=Family, 2=Subfamily, 4=Full Name, 6=PostScript
                if record.nameID == 1 and not metadata["font_family"]:
                    try: metadata["font_family"] = record.toUnicode()
                    except: pass
                elif record.nameID == 2 and not metadata["font_subfamily"]:
                    try: metadata["font_subfamily"] = record.toUnicode()
                    except: pass
                elif record.nameID == 4 and not metadata["full_name"]:
                    try: metadata["full_name"] = record.toUnicode()
                    except: pass
                elif record.nameID == 6 and not metadata["postscript_name"]:
                    try: metadata["postscript_name"] = record.toUnicode()
                    except: pass
        font.close()
    except Exception:
        # Catch any TTF parsing errors from malformed fonts
        pass 
        
    return metadata

def generate_metadata_csv():
    ttf_files = [str(p) for p in Path(TTF_DIR).rglob("*.ttf")]
    if not ttf_files:
        print(f"No TTF files found in {TTF_DIR}.")
        return

    num_cores = multiprocessing.cpu_count()
    print(f"Generating metadata for {len(ttf_files)} fonts using {num_cores} CPU cores...")
    
    fieldnames = [
        "file_name", "file_path", "file_size_bytes", 
        "font_family", "font_subfamily", "full_name", "postscript_name"
    ]
    
    pool = multiprocessing.Pool(processes=num_cores)
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Use imap_unordered for maximum performance
        iterator = pool.imap_unordered(get_font_metadata, ttf_files, chunksize=100)
        for meta in tqdm(iterator, total=len(ttf_files), desc="Extracting Metadata"):
            writer.writerow(meta)
            
    pool.close()
    pool.join()
    print(f"Metadata successfully saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_metadata_csv()
