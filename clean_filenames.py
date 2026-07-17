import os
import re
from pathlib import Path

def clean_filename(filename):
    # Strip trailing spaces before the extension
    name, ext = os.path.splitext(filename)
    name = name.strip()
    
    # Replace Kaggle-forbidden characters with underscores
    # (Kaggle forbids: [, ], &, ', ", and trailing spaces)
    name = re.sub(r"[\[\]&\'\"]", "_", name)
    
    return name + ext

def main():
    ttf_dir = Path("ttf_files")
    if not ttf_dir.exists():
        print("Error: ttf_files directory not found.")
        return
        
    renamed_count = 0
    for filepath in ttf_dir.rglob("*.*"):
        if filepath.is_file():
            cleaned_name = clean_filename(filepath.name)
            
            # If the filename violates Kaggle's rules, rename it!
            if cleaned_name != filepath.name:
                new_filepath = filepath.parent / cleaned_name
                
                # If a file with the cleaned name already exists, add a number to prevent overwriting
                counter = 1
                while new_filepath.exists():
                    name, ext = os.path.splitext(cleaned_name)
                    new_filepath = filepath.parent / f"{name}_{counter}{ext}"
                    counter += 1
                    
                filepath.rename(new_filepath)
                renamed_count += 1
                
    print(f"Successfully cleaned and renamed {renamed_count} files to be Kaggle-compliant.")

if __name__ == "__main__":
    main()
