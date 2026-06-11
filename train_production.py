import os
import argparse
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from pytorch_metric_learning import losses, miners, samplers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FontSubsetDataset(Dataset):
    def __init__(self, root_dir: Path, selected_fonts: list, transform=None):
        self.root_dir = Path(root_dir)
        self.selected_fonts = sorted(selected_fonts)
        self.transform = transform
        
        self.class_to_idx = {font_name: i for i, font_name in enumerate(self.selected_fonts)}
        self.samples = []
        self.labels = []
        
        logger.info(f"Scanning {len(self.selected_fonts)} selected font folders...")
        for font in self.selected_fonts:
            font_dir = self.root_dir / font
            if not font_dir.is_dir():
                logger.warning(f"Directory not found: {font_dir}")
                continue
            
            # Find all images in this font folder
            for img_path in font_dir.glob("*.png"):
                self.samples.append(str(img_path))
                self.labels.append(self.class_to_idx[font])
                
        logger.info(f"Loaded {len(self.samples)} images for {len(self.selected_fonts)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class MetricResNet18(nn.Module):
    def __init__(self, embedding_size=128):
        super(MetricResNet18, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.embedder = nn.Linear(resnet.fc.in_features, embedding_size)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.embedder(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


def get_next_font_chunk(dataset_dir: str, num_classes: int, tracking_csv: str = "local_test_fonts.csv"):
    """
    Finds fonts that haven't been trained yet, selects the next chunk, 
    and appends them to the tracking CSV so they aren't trained again.
    """
    dataset_path = Path(dataset_dir)
    all_fonts = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    trained_fonts = set()
    if os.path.exists(tracking_csv):
        df_existing = pd.read_csv(tracking_csv)
        if 'font_name' in df_existing.columns:
            trained_fonts = set(df_existing['font_name'].tolist())
            
    remaining_fonts = [f for f in all_fonts if f not in trained_fonts]
    
    if not remaining_fonts:
        logger.info(f"All {len(all_fonts)} fonts have already been trained!")
        return []
        
    selected_fonts = remaining_fonts[:num_classes]
    
    # Append the new fonts to the tracking CSV
    start_id = len(trained_fonts)
    new_rows = [{"id": start_id + i, "font_name": font} for i, font in enumerate(selected_fonts)]
    df_new = pd.DataFrame(new_rows)
    
    if os.path.exists(tracking_csv):
        df_new.to_csv(tracking_csv, mode='a', header=False, index=False)
    else:
        df_new.to_csv(tracking_csv, index=False)
        
    logger.info(f"Selected {len(selected_fonts)} new fonts. Total fonts tracked: {start_id + len(selected_fonts)}.")
    
    return selected_fonts


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Production Metric Learning Training (RTX 5050 Ready)")
    parser.add_argument("--dataset_dir", type=str, default="./synthetic_dataset")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of new fonts to train in this run")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs for this chunk")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for RTX 5050 VRAM")
    parser.add_argument("--m_per_class", type=int, default=4, help="Samples per class in a batch")
    parser.add_argument("--resume_from", type=str, default="poc_metric_model.pth", help="Path to checkpoint to resume training")
    parser.add_argument("--save_name", type=str, default="best_metric_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}. If you have an RTX 5050, this MUST say 'cuda'.")
    
    # Automatically filter out already trained fonts and pick the next 1000
    selected_fonts = get_next_font_chunk(args.dataset_dir, args.chunk_size)
    if not selected_fonts:
        return
    
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FontSubsetDataset(root_dir=args.dataset_dir, selected_fonts=selected_fonts, transform=transform)
    
    actual_batch_size = min(args.batch_size, len(selected_fonts) * args.m_per_class)
    if actual_batch_size < args.batch_size:
        logger.warning(f"Reducing batch_size from {args.batch_size} to {actual_batch_size} to satisfy MPerClassSampler requirements.")

    sampler = samplers.MPerClassSampler(
        labels=dataset.labels,
        m=args.m_per_class,
        batch_size=actual_batch_size,
        length_before_new_iter=len(dataset)
    )
    
    # Pin memory is critical for fast GPU transfer on the RTX 5050
    dataloader = DataLoader(
        dataset, 
        batch_size=actual_batch_size, 
        sampler=sampler, 
        num_workers=8,  # Increased for production hardware
        pin_memory=True
    )
    
    model = MetricResNet18(embedding_size=128).to(device)
    
    if os.path.exists(args.save_name):
        logger.info(f"Resuming training from latest production checkpoint: {args.save_name}...")
        model.load_state_dict(torch.load(args.save_name, map_location=device, weights_only=True))
    elif args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming training from baseline checkpoint: {args.resume_from}...")
        model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True))
    else:
        logger.warning("No checkpoint found. Initializing model from scratch.")
    
    # ---------------------------------------------------------
    # MLOPS SAFEGUARD #1: Cosine Annealing Learning Rate
    # Starts high to push clusters, drops to near zero at the end for fine-tuning
    # ---------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # ---------------------------------------------------------
    # MLOPS SAFEGUARD #3: Harder Mining
    # Triplet margin set to 0.2, BatchHardMiner zeroes in on identical-looking fonts
    # ---------------------------------------------------------
    loss_func = losses.TripletMarginLoss(margin=0.2)
    miner = miners.BatchHardMiner()
    
    logger.info("Starting production training loop...")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Wrapping dataloader in tqdm for a clean progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
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
            
            # Update the progress bar text
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})
                
        # Step the LR scheduler once per epoch
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"====> EPOCH {epoch} COMPLETED | Avg Triplet Loss: {avg_loss:.4f} <====")
        
        # ---------------------------------------------------------
        # MLOPS SAFEGUARD #2: Epoch Checkpointing
        # Save model ONLY when we achieve a historic low loss
        # ---------------------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_name)
            logger.info(f"*** New Historic Low Loss! Checkpoint saved to '{args.save_name}' ***")

if __name__ == "__main__":
    main()
