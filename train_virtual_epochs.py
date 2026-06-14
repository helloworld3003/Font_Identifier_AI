import os
import random
import string
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_metric_learning import losses, miners

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded Hardware Constraints & Hyperparameters
TTF_DIR = "ttf_files" # The directory containing 300k sanitized TTF files
BATCH_SIZE = 64       # Hardcoded to prevent OOM on 8GB RTX 5050
M_PER_CLASS = 4       # Number of instances per class per batch (for BatchHardMiner)
EMBEDDING_SIZE = 256
VIRTUAL_EPOCH_BATCHES = 10000
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5          # Early stopping patience

class VirtualEpochBatchSampler(Sampler):
    """
    Ensures each batch of size BATCH_SIZE contains M_PER_CLASS instances of the same class (font).
    Yields strictly VIRTUAL_EPOCH_BATCHES batches per epoch.
    """
    def __init__(self, num_classes, batch_size, m_per_class, num_batches):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.m_per_class = m_per_class
        self.num_batches = num_batches
        self.classes_per_batch = self.batch_size // self.m_per_class

    def __iter__(self):
        for _ in range(self.num_batches):
            classes = np.random.choice(self.num_classes, self.classes_per_batch, replace=False)
            batch = []
            for c in classes:
                # Append the class index 'm_per_class' times
                batch.extend([c] * self.m_per_class)
            yield batch

    def __len__(self):
        return self.num_batches

class DynamicFontDataset(Dataset):
    """
    Loads a TTF into RAM, renders a random alphanumeric string.
    Applies Albumentations (noise, blur, rotation) dynamically.
    """
    def __init__(self, ttf_dir, transform=None):
        self.ttf_files = list(Path(ttf_dir).rglob("*.ttf"))
        if len(self.ttf_files) == 0:
            raise ValueError(f"No TTF files found in {ttf_dir}. Please check TTF_DIR.")
        self.transform = transform
        logger.info(f"Loaded {len(self.ttf_files)} unique font files into the dataset.")

    def __len__(self):
        return len(self.ttf_files)

    def generate_random_string(self, length=5):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))

    def __getitem__(self, idx):
        ttf_path = self.ttf_files[idx]
        
        try:
            text = self.generate_random_string(random.randint(3, 8))
            
            # Dynamic RAM Rendering
            font_size = random.randint(40, 90)
            font = ImageFont.truetype(str(ttf_path), font_size)
            
            canvas_size = 224
            image = Image.new("RGB", (canvas_size, canvas_size), "white")
            draw = ImageDraw.Draw(image)

            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            x = (canvas_size - text_w) / 2
            y = (canvas_size - text_h) / 2
            
            draw.text((x, y), text, font=font, fill="black")
        except Exception:
            # Fallback to empty canvas if rendering fails to avoid crashing dataloader
            image = Image.new("RGB", (224, 224), "white")
            
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, idx

class FontEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        # PyTorch ResNet50 backbone modified for metric learning
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        # L2-normalized 256D embedding vector
        x = F.normalize(x, p=2, dim=1)
        return x

def get_train_transforms():
    # Albumentations: noise, blur, slight rotation
    return A.Compose([
        A.Rotate(limit=10, p=0.5, border_mode=0, value=(255, 255, 255)),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # ImageNet normalization standard for ResNet
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Targeting device: {device}")
    
    # Initialize Dataset and Transformations
    transform = get_train_transforms()
    dataset = DynamicFontDataset(TTF_DIR, transform=transform)
    
    # Custom Sampler for Virtual Epochs and Triplet Construction
    batch_sampler = VirtualEpochBatchSampler(
        num_classes=len(dataset),
        batch_size=BATCH_SIZE,
        m_per_class=M_PER_CLASS,
        num_batches=VIRTUAL_EPOCH_BATCHES
    )
    
    # Dataloader configurations for maximum PCIe throughput
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    # Initialize Model, Loss, Miner, Optimizer, Scheduler
    model = FontEmbeddingModel(embedding_size=EMBEDDING_SIZE).to(device)
    
    miner = miners.BatchHardMiner()
    loss_fn = losses.TripletMarginLoss(margin=0.2)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Early Stopping tracking
    best_loss = float('inf')
    epochs_no_improve = 0
    
    logger.info("Starting Dynamic Training Pipeline...")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        active_triplets = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # torch.amp Automatic Mixed Precision
            with torch.autocast(device_type=device.type, enabled=True):
                embeddings = model(images)
                hard_pairs = miner(embeddings, labels)
                loss = loss_fn(embeddings, labels, hard_pairs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            active_triplets += miner.num_triplets
            
            if (batch_idx + 1) % 500 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx + 1}/{VIRTUAL_EPOCH_BATCHES} | "
                            f"Loss: {loss.item():.4f} | Active Triplets: {miner.num_triplets}")
        
        # End of Virtual Epoch
        avg_loss = running_loss / VIRTUAL_EPOCH_BATCHES
        scheduler.step()
        
        logger.info(f"=== Epoch {epoch} Summary ===")
        logger.info(f"Average Loss: {avg_loss:.4f} | Total Hard Triplets: {active_triplets}")
        
        # Early Stopping Logic based strictly on lowest recorded training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"New best loss achieved! Saved best_model.pth")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement. Early stopping patience: {epochs_no_improve}/{PATIENCE}")
            
        if epochs_no_improve >= PATIENCE:
            logger.warning(f"Early stopping triggered after {epoch} epochs!")
            break
            
    logger.info("Training Pipeline Complete.")

if __name__ == "__main__":
    train()
