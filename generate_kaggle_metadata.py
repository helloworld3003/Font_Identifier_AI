import pandas as pd
import json
import os

def generate_metadata():
    print("Loading CSVs...")
    try:
        df1 = pd.read_csv("extras/fonts_metadata.csv")
    except FileNotFoundError:
        df1 = pd.DataFrame()
        
    try:
        df2 = pd.read_csv("extras/fonts_metadata_2.csv")
    except FileNotFoundError:
        df2 = pd.DataFrame()
        
    df = pd.concat([df1, df2]).drop_duplicates(subset=['file_name'])
    
    # Kaggle dataset ID - Replace with actual username/dataset-slug
    kaggle_id = "YOUR_KAGGLE_USERNAME/YOUR_DATASET_SLUG"
    
    metadata = {
        "title": "Font Identifier Dataset",
        "id": kaggle_id,
        "licenses": [{"name": "CC0-1.0"}],
        "resources": []
    }
    
    print(f"Generating descriptions for {len(df)} files...")
    
    for _, row in df.iterrows():
        # Clean up any NaN values
        family = row['font_family'] if pd.notna(row['font_family']) else "Unknown"
        subfamily = row['font_subfamily'] if pd.notna(row['font_subfamily']) else "Unknown"
        full_name = row['full_name'] if pd.notna(row['full_name']) else "Unknown"
        
        description = f"Font Family: {family} | Subfamily: {subfamily} | Full Name: {full_name}"
        
        # The file_path from the CSV (e.g., 'ttf_files_2\\ttf_files\\Abbott W05 Bold.ttf')
        # Kaggle requires forward slashes for paths in the JSON
        kaggle_path = str(row['file_path']).replace("\\", "/")
        
        resource = {
            "path": kaggle_path,
            "description": description
        }
        metadata["resources"].append(resource)
        
    out_file = "dataset-metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nSuccess! Generated {out_file} with {len(metadata['resources'])} file descriptions.")
    print(f"File size: {os.path.getsize(out_file) / (1024*1024):.2f} MB")
    print("\nTo upload to Kaggle, run:")
    print("kaggle datasets version -p . -m \"Added rich file metadata\"")

if __name__ == "__main__":
    generate_metadata()
