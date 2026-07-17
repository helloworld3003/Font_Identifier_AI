import csv
import re
from pathlib import Path

# Load familyLabels.csv
font_names = []
with open('familyLabels.csv', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Split by last comma to get name and class
        parts = line.rsplit(',', 1)
        if len(parts) == 2:
            font_names.append(parts[0])

# Load all ttf filenames
ttf_stems = []
for p in Path('ttf_files').rglob('*.ttf'):
    ttf_stems.append(p.stem)
for p in Path('new_fonts').rglob('*.ttf'):
    ttf_stems.append(p.stem)

def normalize(name):
    # Lowercase and remove all non-alphanumeric chars
    return re.sub(r'[^a-z0-9]', '', name.lower())

normalized_stems = [normalize(stem) for stem in ttf_stems]

missing = []
for name in font_names:
    norm_name = normalize(name)
    if not norm_name:
        continue
        
    found = False
    for stem in normalized_stems:
        # If the file stem starts with the font name (e.g., 'abeezeeregular' starts with 'abeezee')
        # Or if the font name is in the file stem
        if stem.startswith(norm_name) or norm_name in stem:
            found = True
            break
        # Sometimes the label has extra info like "Regular" or "Std" which the file stem doesn't
        # Try stripping "std", "pro", "regular", "mt" from norm_name
        stripped_norm = norm_name.replace('std', '').replace('pro', '').replace('regular', '').replace('mt', '')
        if len(stripped_norm) >= 4 and (stem.startswith(stripped_norm) or stripped_norm in stem):
            found = True
            break
            
    if not found:
        missing.append(name)

print(f"Total labeled fonts: {len(font_names)}")
print(f"Total available TTF files: {len(ttf_stems)}")
print(f"Missing fonts: {len(missing)}")

# Write to a text file
with open('missing_family_fonts.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total labeled fonts: {len(font_names)}\n")
    f.write(f"Missing fonts: {len(missing)}\n\n")
    for m in missing:
        f.write(f"{m}\n")
