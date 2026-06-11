import os
import sys
import shutil
import logging
import random
import subprocess
from pathlib import Path
import pandas as pd
import torch

# Phase 1 imports
from phase1_data_pipeline import SyntheticFontDataGenerator

# Phase 2 imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_metric_learning import losses, miners, samplers
from train_production import FontSubsetDataset, MetricResNet18

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    new_fonts_dir = Path("new_fonts")
    
    # 1. Setup and Initialization
    if not new_fonts_dir.exists():
        new_fonts_dir.mkdir()
        logger.info(f"Created '{new_fonts_dir}' directory.")
        logger.info("Please place your new .ttf files into this directory and run this script again.")
        return
        
    ttf_files = list(new_fonts_dir.glob("*.ttf"))
    if not ttf_files:
        logger.info(f"No .ttf files found in '{new_fonts_dir}'. Nothing to do.")
        return
        
    new_font_names = [f.stem for f in ttf_files]
    logger.info(f"Detected {len(new_font_names)} new font(s): {new_font_names}")
    
    # -----------------------------------------------------------------
    # PHASE 1: GENERATE SYNTHETIC DATA
    # -----------------------------------------------------------------
    logger.info("\n" + "="*50)
    logger.info("PHASE 1: SYNTHETIC DATA GENERATION")
    logger.info("="*50)
    
    generator = SyntheticFontDataGenerator(
        fonts_dir=str(new_fonts_dir),
        output_dir="synthetic_dataset",
        samples_per_font=1000
    )
    # Using 1 worker per font or a few workers for fast generation
    generator.generate()
    
    # Move TTF files to main directory so they are indexed in Phase 3
    main_ttf_dir = Path("ttf_files")
    main_ttf_dir.mkdir(exist_ok=True)
    for ttf in ttf_files:
        dest = main_ttf_dir / ttf.name
        if dest.exists():
            dest.unlink() # Overwrite if it exists
        shutil.move(str(ttf), str(dest))
        
    logger.info(f"Moved {len(ttf_files)} .ttf files to master '{main_ttf_dir}' directory.")

    # -----------------------------------------------------------------
    # PHASE 2: INCREMENTAL TRAINING (Catastrophic Forgetting Safeguard)
    # -----------------------------------------------------------------
    logger.info("\n" + "="*50)
    logger.info("PHASE 2: INCREMENTAL TRAINING (Catastrophic Forgetting Safeguard)")
    logger.info("="*50)
    
    # Read the master tracking log to get old fonts
    tracking_csv = "local_test_fonts.csv"
    old_fonts = []
    
    if os.path.exists(tracking_csv):
        df_old = pd.read_csv(tracking_csv)
        if 'font_name' in df_old.columns:
            all_old = df_old['font_name'].tolist()
            # Randomly pick up to 50 old fonts to anchor the existing metric space
            num_anchors = min(50, len(all_old))
            old_fonts = random.sample(all_old, num_anchors)
    else:
        df_old = pd.DataFrame(columns=['id', 'font_name'])
        
    training_fonts = new_font_names + old_fonts
    logger.info(f"Training Batch: {len(new_font_names)} New Font(s) + {len(old_fonts)} Old Anchor Font(s).")
    
    # Update tracking CSV with the new fonts so they are permanently logged
    start_id = len(df_old)
    new_rows = [{"id": start_id + i, "font_name": font} for i, font in enumerate(new_font_names)]
    pd.DataFrame(new_rows).to_csv(tracking_csv, mode='a', header=not os.path.exists(tracking_csv), index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Build dataset and metric learning sampler
    dataset = FontSubsetDataset(root_dir="synthetic_dataset", selected_fonts=training_fonts, transform=transform)
    sampler = samplers.MPerClassSampler(
        labels=dataset.labels,
        m=4,
        batch_size=128,
        length_before_new_iter=len(dataset)
    )
    
    dataloader = DataLoader(dataset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True)
    
    model = MetricResNet18(embedding_size=128).to(device)
    model_path = "best_metric_model.pth"
    
    if os.path.exists(model_path):
        logger.info(f"Loading existing model weights from '{model_path}'...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        # Fallback to the POC model if best_metric_model isn't there yet
        alt_model_path = "poc_metric_model.pth"
        if os.path.exists(alt_model_path):
             logger.info(f"Loading existing model weights from '{alt_model_path}'...")
             model.load_state_dict(torch.load(alt_model_path, map_location=device, weights_only=True))
        else:
             logger.warning("No existing model weights found! Initializing from scratch.")
        
    # Extremely small learning rate so we don't blow up the established geometric clusters!
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = losses.TripletMarginLoss(margin=0.2)
    miner = miners.BatchHardMiner()
    
    epochs = 20 # 20 epochs is plenty for fine-tuning just a handful of new classes
    model.train()
    
    from tqdm import tqdm
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Fine-Tuning Epoch {epoch}/{epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(images)
            hard_pairs = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, hard_pairs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        
    # Save the dynamically updated weights back to disk
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved updated model weights to '{model_path}'")
    
    # -----------------------------------------------------------------
    # PHASE 3: DATABASE RE-INDEXING
    # -----------------------------------------------------------------
    logger.info("\n" + "="*50)
    logger.info("PHASE 3: REBUILDING FAISS INDEX")
    logger.info("="*50)
    
    # We call the indexing script as a subprocess to keep memory clean.
    # It automatically reads local_test_fonts.csv, sees the new fonts we just appended,
    # loads the newly saved best_metric_model.pth, and builds a massive new database!
    try:
        subprocess.run([sys.executable, "phase3_database_indexing.py"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Phase 3 Indexing failed: {e}")
        return
    
    logger.info("\n" + "*"*60)
    logger.info("🎉 NEW FONTS SUCCESSFULLY ADDED, TRAINED, AND INDEXED! 🎉")
    logger.info("*"*60)

if __name__ == "__main__":
    main()
