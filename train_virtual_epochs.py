import os
import random
import string
import logging
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

from pytorch_metric_learning import losses, miners

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TTF_DIR = "ttf_files"
BATCH_SIZE = 64
M_PER_CLASS = 4
EMBEDDING_SIZE = 256
VIRTUAL_EPOCH_BATCHES = 10000
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 5

# ==========================================
# 1. TRAIN/TEST SYMMETRY AUGMENTATION
# ==========================================
def simulate_adaptive_threshold(image, **kwargs):
    """
    Simulates OpenCV adaptive thresholding during training 
    so the model is invariant to jagged edges and binary masking.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block_size = np.random.choice([7, 11, 15, 19])
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, int(block_size), 2
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

def get_train_transforms():
    return A.Compose([
        A.Rotate(limit=8, p=0.4),
        A.Perspective(scale=(0.05, 0.09), p=0.3),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.4),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Lambda(image=simulate_adaptive_threshold, p=0.4),
        A.InvertImg(p=0.2), # Handles white-on-black text styles
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class VirtualEpochBatchSampler(Sampler):
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
                batch.extend([c] * self.m_per_class)
            yield batch

    def __len__(self):
        return self.num_batches

class DynamicFontDataset(Dataset):
    def __init__(self, ttf_dir, transform=None):
        self.ttf_files = list(Path(ttf_dir).rglob("*.ttf"))
        if len(self.ttf_files) == 0:
            raise ValueError(f"No TTF files found in {ttf_dir}")
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
            image = Image.new("RGB", (224, 224), "white")
            
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, idx

# ==========================================
# 2. STATE-OF-THE-ART BACKBONE (ConvNeXt)
# ==========================================
class ConvNeXtFontEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(ConvNeXtFontEncoder, self).__init__()
        # Load ConvNeXt-Tiny as a pure feature extractor
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        # Custom projection head for Deep Metric Learning
        self.fc = nn.Linear(num_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.fc(features)
        # Strict L2 Normalization to place embeddings on a hypersphere
        return F.normalize(embeddings, p=2, dim=1)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Targeting device: {device}")
    
    transform = get_train_transforms()
    dataset = DynamicFontDataset(TTF_DIR, transform=transform)
    
    batch_sampler = VirtualEpochBatchSampler(
        num_classes=len(dataset),
        batch_size=BATCH_SIZE,
        m_per_class=M_PER_CLASS,
        num_batches=VIRTUAL_EPOCH_BATCHES
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        num_workers=2, 
        pin_memory=True
    )
    
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE).to(device)
    
    miner = miners.BatchHardMiner()
    
    # ==========================================
    # 3. CROSS-BATCH MEMORY LOGIC
    # ==========================================
    base_loss_function = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
    loss_func = losses.CrossBatchMemory(
        loss=base_loss_function, 
        embedding_size=EMBEDDING_SIZE, 
        memory_size=4096,
        miner=miner
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    logger.info("Starting Gold-Standard Dynamic Training Pipeline...")
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        active_triplets = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=True):
                embeddings = model(images)
                loss = loss_func(embeddings, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            active_triplets += miner.num_triplets
            
            if (batch_idx + 1) % 500 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx + 1}/{VIRTUAL_EPOCH_BATCHES} | "
                            f"Loss: {loss.item():.4f} | Active Triplets: {miner.num_triplets}")
        
        avg_loss = running_loss / VIRTUAL_EPOCH_BATCHES
        scheduler.step()
        
        logger.info(f"=== Epoch {epoch} Summary ===")
        logger.info(f"Average Loss: {avg_loss:.4f} | Total Hard Triplets: {active_triplets}")
        
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
