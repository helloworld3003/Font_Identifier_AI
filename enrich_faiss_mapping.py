import pandas as pd
import os

def enrich_mapping():
    print("Loading faiss_mapping.csv...")
    mapping_df = pd.read_csv("faiss_mapping.csv")
    
    # Extract file_name from font_path for joining
    mapping_df['file_name'] = mapping_df['font_path'].apply(lambda x: os.path.basename(x))
    
    print("Loading metadata CSVs...")
    try:
        df1 = pd.read_csv("extras/fonts_metadata.csv")
    except FileNotFoundError:
        df1 = pd.DataFrame()
        
    try:
        df2 = pd.read_csv("extras/fonts_metadata_2.csv")
    except FileNotFoundError:
        df2 = pd.DataFrame()
        
    meta_df = pd.concat([df1, df2]).drop_duplicates(subset=['file_name'])
    
    print("Merging metadata...")
    # Select only the columns we want to add
    meta_subset = meta_df[['file_name', 'font_family', 'font_subfamily', 'full_name']]
    
    # Perform a left join
    enriched_df = pd.merge(mapping_df, meta_subset, on='file_name', how='left')
    
    # Fill NaN values for fonts that weren't in the metadata CSVs
    enriched_df.fillna("Unknown", inplace=True)
    
    # Drop the temporary file_name column
    enriched_df.drop(columns=['file_name'], inplace=True)
    
    # Save the enriched mapping
    enriched_df.to_csv("faiss_mapping_enriched.csv", index=False)
    
    print("Success! Created faiss_mapping_enriched.csv with rich font information.")
    
    # Print a sample
    print(enriched_df.head())

if __name__ == "__main__":
    enrich_mapping()
