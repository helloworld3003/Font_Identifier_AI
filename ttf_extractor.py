import shutil
from pathlib import Path

def extract_ttf_files(source_dir, dest_dir):
    """
    Extracts all .ttf files from a source directory (and subdirectories)
    and copies them to a destination directory.
    """
    # Convert string paths to Path objects
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Ensure the destination directory exists; create it if it doesn't
    dest_path.mkdir(parents=True, exist_ok=True)

    # rglob searches recursively for the specified pattern
    ttf_files = list(source_path.rglob("*.ttf"))

    if not ttf_files:
        print(f"No .ttf files found in '{source_dir}'")
        return

    print(f"Found {len(ttf_files)} .ttf file(s). Starting copy process...")
    copied_count = 0

    for file_path in ttf_files:
        # Define the target path (destination folder + original file name)
        target_path = dest_path / file_path.name

        try:
            # shutil.copy2 preserves file metadata (timestamps, etc.)
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path.name}")
            copied_count += 1
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")

    print(f"\nSuccess! Extracted {copied_count} files to '{dest_dir}'")

# --- Setup your paths here ---
# Note: Use forward slashes (/) or raw strings (r"C:\...") for Windows paths
SOURCE_FOLDER = r"E:\New folder\coding_arc\font_identifier\fonts_main\fonts-main\ufl"
DESTINATION_FOLDER = r"E:\New folder\coding_arc\font_identifier\ttf_files"

if __name__ == "__main__":
    extract_ttf_files(SOURCE_FOLDER, DESTINATION_FOLDER)