import os
import random
import string
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import cv2
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SyntheticFontDataGenerator:
    """
    Phase 1: Synthetic Data Pipeline for Font Identifier AI.
    
    Generates synthetic text images from a directory of .ttf fonts, applies 
    realistic augmentations using Albumentations, and saves them to a structured dataset directory.
    """
    
    def __init__(
        self, 
        fonts_dir: str, 
        output_dir: str, 
        samples_per_font: int = 1000,
        image_size: tuple = (256, 64), # width, height
        min_chars: int = 3,
        max_chars: int = 8,
        font_size_range: tuple = (32, 56)
    ):
        self.fonts_dir = Path(fonts_dir)
        self.output_dir = Path(output_dir)
        self.samples_per_font = samples_per_font
        self.image_size = image_size
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.font_size_range = font_size_range
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fonts = self._load_font_paths()
        
        # Define Albumentations pipeline
        self.transform = A.Compose([
            # Background/Texture variations
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.ToGray(p=0.2),
            
            # Noise and Degradation
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0)
            ], p=0.4),
            
            # Blur
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.4),
            
            # Geometric / Perspective
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.6),
            A.Perspective(scale=(0.01, 0.05), keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=(255, 255, 255), p=0.3),
            
            # Occasional Inversion (white text on dark background)
            A.InvertImg(p=0.2),
        ])

    def _load_font_paths(self):
        """Retrieve all .ttf files from the fonts directory."""
        fonts = list(self.fonts_dir.rglob("*.ttf"))
        logger.info(f"Found {len(fonts)} TTF files in {self.fonts_dir}")
        return fonts
        
    def _generate_random_string(self) -> str:
        """Generate a random alphanumeric string."""
        length = random.randint(self.min_chars, self.max_chars)
        # Mix of uppercase, lowercase, and digits
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    def _create_base_image(self, font_path: Path, text: str) -> np.ndarray:
        """Render text onto a plain white PIL image and convert to numpy array."""
        font_size = random.randint(self.font_size_range[0], self.font_size_range[1])
        
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:
            logger.error(f"Failed to load font {font_path}: {e}")
            return None

        # Create a white canvas
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate text bounding box to center it
        try:
            bbox = font.getbbox(text) # left, top, right, bottom
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (self.image_size[0] - text_width) // 2
            y = (self.image_size[1] - text_height) // 2
            
            # Ensure text is not completely cut off
            x = max(10, min(x, self.image_size[0] - 20))
            
            draw.text((x, y), text, font=font, fill='black')
        except Exception as e:
            logger.error(f"Error drawing text with font {font_path}: {e}")
            return None
            
        return np.array(img)

    def _process_single_font(self, font_path: Path):
        """Generate synthetic samples for a single font."""
        font_name = font_path.stem
        font_out_dir = self.output_dir / font_name
        font_out_dir.mkdir(exist_ok=True)
        
        successful_samples = 0
        attempts = 0
        max_attempts = self.samples_per_font * 2
        
        metadata = []
        
        while successful_samples < self.samples_per_font and attempts < max_attempts:
            attempts += 1
            text = self._generate_random_string()
            
            base_img = self._create_base_image(font_path, text)
            if base_img is None:
                continue
                
            # Apply Albumentations
            try:
                augmented = self.transform(image=base_img)
                aug_img = augmented['image']
            except Exception as e:
                logger.error(f"Augmentation failed for {font_name}: {e}")
                continue
                
            # Save the image
            out_filename = f"{successful_samples:05d}_{text}.png"
            out_path = font_out_dir / out_filename
            try:
                Image.fromarray(aug_img).save(out_path)
                successful_samples += 1
                metadata.append({
                    "image_path": f"{font_name}/{out_filename}",
                    "font_name": font_name,
                    "text": text
                })
            except Exception as e:
                logger.error(f"Failed to save image {out_path}: {e}")
                
        if successful_samples < self.samples_per_font:
            logger.warning(f"Could only generate {successful_samples} samples for {font_name}")
            
        return metadata

    def generate(self, num_workers: int = None):
        """Run the generation pipeline in parallel."""
        if not self.fonts:
            logger.error("No fonts to process. Exiting.")
            return

        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)
            
        logger.info(f"Starting generation for {len(self.fonts)} fonts using {num_workers} workers.")
        logger.info(f"Target: {self.samples_per_font} samples per font.")
        
        all_metadata = []
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(self._process_single_font, self.fonts), total=len(self.fonts)))
            for r in results:
                all_metadata.extend(r)
                
        import pandas as pd
        if all_metadata:
            df = pd.DataFrame(all_metadata)
            metadata_path = self.output_dir / "metadata.csv"
            if metadata_path.exists():
                df.to_csv(metadata_path, mode='a', header=False, index=False)
            else:
                df.to_csv(metadata_path, index=False)
            logger.info(f"Saved metadata for {len(df)} images to {metadata_path}")
            
        logger.info(f"Phase 1 complete! Dataset saved to {self.output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic font dataset.")
    parser.add_argument("--fonts_dir", type=str, default="./ttf_files", help="Directory containing .ttf files")
    parser.add_argument("--output_dir", type=str, default="./synthetic_dataset", help="Output directory for generated images")
    parser.add_argument("--samples", type=int, default=1000, help="Number of images to generate per font")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU workers for parallel processing")
    
    args = parser.parse_args()
    
    generator = SyntheticFontDataGenerator(
        fonts_dir=args.fonts_dir,
        output_dir=args.output_dir,
        samples_per_font=args.samples
    )
    generator.generate(num_workers=args.workers)
