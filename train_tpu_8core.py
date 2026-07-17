import os
import random
import string
import logging
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

from pytorch_metric_learning import losses, miners
from torch.utils.data import Dataset, DataLoader, Sampler

# --- PyTorch XLA Imports ---
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# Configure global logger (mostly disabled for workers, handled via xm.master_print)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# HYPERPARAMETERS & KAGGLE AUTO-DETECT
# ==========================================
# Check if running on Kaggle and auto-mount the dataset to prevent symlink errors
kaggle_input_dir = Path("/kaggle/input")
if kaggle_input_dir.exists():
    try:
        first_ttf = next(kaggle_input_dir.rglob("*.ttf"))
        TTF_DIR = str(first_ttf.parent)
        print(f"Auto-detected Kaggle dataset at: {TTF_DIR}")
    except StopIteration:
        TTF_DIR = "ttf_files"
else:
    TTF_DIR = "ttf_files"

# IMPORTANT: BATCH_SIZE is per-core! 
# 64 batch size * 8 cores = 512 Effective Batch Size.
BATCH_SIZE = 64  
M_PER_CLASS = 4

EMBEDDING_SIZE = 512
VIRTUAL_EPOCH_BATCHES = 1250
MAX_EPOCHS = 50
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 5e-4
PATIENCE = 15

# ==========================================
# 1. DATASET & AUGMENTATION
# ========================================== 
def simulate_adaptive_threshold(image, **kwargs):
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
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Lambda(image=simulate_adaptive_threshold, p=0.4),
        A.InvertImg(p=0.2), 
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
        self.ttf_files = list(Path(ttf_dir).rglob("*.ttf")) + list(Path(ttf_dir).rglob("*.otf"))
        if len(self.ttf_files) == 0:
            raise ValueError(f"No font files found in {ttf_dir}")
        self.transform = transform
        
        # Only master node should print dataset stats
        xm.master_print(f"Loaded {len(self.ttf_files)} unique font files into the dataset.")

    def __len__(self):
        return len(self.ttf_files)

    def generate_random_string(self, length=5):
        chars = string.ascii_letters + string.digits
        return ''.join(random.choices(chars, k=length))

    def __getitem__(self, idx):
        ttf_path = self.ttf_files[idx]
        try:
            text = self.generate_random_string(random.randint(4, 10))
            font_size = random.randint(40, 80)
            font = ImageFont.truetype(str(ttf_path), font_size)
            
            temp_image = Image.new("RGB", (1, 1), "white")
            temp_draw = ImageDraw.Draw(temp_image)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = max(1, bbox[2] - bbox[0])
            text_h = max(1, bbox[3] - bbox[1])
            
            image = Image.new("RGB", (text_w, text_h), "white")
            draw = ImageDraw.Draw(image)
            draw.text((-bbox[0], -bbox[1]), text, font=font, fill="black")
            
            target_w, target_h = 256, 64
            scale = min(target_w / text_w, target_h / text_h)
            new_w = max(1, int(text_w * scale))
            new_h = max(1, int(text_h * scale))
            
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            final_image = Image.new("RGB", (target_w, target_h), "white")
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            final_image.paste(image, (x, y))
            
        except Exception:
            return self.__getitem__(random.randint(0, len(self.ttf_files) - 1))
            
        image_np = np.array(final_image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, idx

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class HE_Block(nn.Module):
    def __init__(self, drop_ratio=0.15):
        super(HE_Block, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if not self.training:
            return x
            
        batch_size, channels = x.size()
        k = max(1, int(channels * self.drop_ratio))
        _, topk_indices = torch.topk(x, k, dim=1)
        
        mask = torch.ones_like(x)
        mask.scatter_(1, topk_indices, 0.0)
        return x * mask

class ConvNeXtFontEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ConvNeXtFontEncoder, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.he_block = HE_Block(drop_ratio=0.15)
        self.fc = nn.Linear(self.backbone.num_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = self.he_block(features)
        embeddings = self.fc(features)
        return F.normalize(embeddings.float(), p=2, dim=1)

# ==========================================
# 3. MULTIPROCESSING TRAINING LOOP (8-CORES)
# ==========================================
def _mp_fn(index, flags):
    """
    This function is replicated 8 times and runs independently on each TPU core.
    """
    # 1. Acquire current TPU core device (Updated for PyTorch XLA 2.x)
    import torch_xla
    device = torch_xla.device()
    xm.master_print(f"Executing Deep Metric Learning on 8 Google Cloud TPU Cores.")
    
    transform = get_train_transforms()
    dataset = DynamicFontDataset(TTF_DIR, transform=transform)
    
    # 2. Construct Data Samplers
    # Every core independently samples its own batch of 64 images.
    batch_sampler = VirtualEpochBatchSampler(
        num_classes=len(dataset),
        batch_size=BATCH_SIZE,
        m_per_class=M_PER_CLASS,
        num_batches=VIRTUAL_EPOCH_BATCHES
    )
    
    # Num_workers=4 spawns 32 total background threads (4 per TPU core) 
    # to dynamically draw the font images in parallel, un-starving the TPU!
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        num_workers=4, 
        pin_memory=False,
        persistent_workers=True
    )
    
    # 3. Model & Loss Setup
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE).to(device)
    
    # XLA CRITICAL FIX: CrossBatchMemory uses dynamic queues and BatchHardMiner produces 
    # dynamically-shaped index tensors. This forces PyTorch XLA to infinitely recompile the 
    # hardware graph every batch, causing 10,000% CPU and 200GB RAM memory leaks!
    # By using pure MultiSimilarityLoss, the pairwise distance matrix is strictly static (64x64).
    loss_func = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5).to(device)
    
    # 4. Optimizer Setup
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.he_block.parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE_HEAD},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1
    
    CHECKPOINT_PATH = "checkpoint.pth"
    MODEL_PATH = "best_model.pth"
    
    if os.path.exists(CHECKPOINT_PATH):
        xm.master_print(f"Found checkpoint file '{CHECKPOINT_PATH}'. Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        xm.master_print(f"Resumed from epoch {checkpoint['epoch']} with best loss {best_loss:.4f}.")
    elif os.path.exists(MODEL_PATH):
        xm.master_print(f"Checkpoint not found. Loading model weights from '{MODEL_PATH}' to continue training...")
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        xm.master_print(f"Successfully loaded backbone weights. Starting training from Epoch 1.")
        
    xm.master_print("Starting XLA 8-Core Training Pipeline...")
    
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # Parallel Loader forces data directly onto the TPU matrix
        mp_device_loader = pl.MpDeviceLoader(dataloader, device)
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(mp_device_loader):
            optimizer.zero_grad()
            
            # XLA handles bfloat16 seamlessly under the hood
            embeddings = model(images)
            loss = loss_func(embeddings, labels)
            
            loss.backward()
            
            # Use XLA Optimizer Step (This automatically syncs and averages gradients across all 8 cores!)
            xm.optimizer_step(optimizer)
            
            # Extract loss for logging (using .item() forces a graph sync, but required for logging)
            current_loss = loss.item()
            running_loss += current_loss
            
            if (batch_idx + 1) % 100 == 0:
                xm.master_print(f"Epoch {epoch}/{MAX_EPOCHS} | Batch {batch_idx + 1}/{VIRTUAL_EPOCH_BATCHES} | Loss: {current_loss:.4f}")
        
        avg_loss = running_loss / (batch_idx + 1)
        scheduler.step()
        epoch_time = time.time() - start_time
        
        xm.master_print(f"=== Epoch {epoch} Summary ===")
        xm.master_print(f"Average Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save complete training state (Must be executed by all cores, but xm.save manages writing safely)
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'epochs_no_improve': epochs_no_improve
        }
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            xm.save(model.state_dict(), "best_model.pth")
            
            checkpoint_state['best_loss'] = best_loss
            checkpoint_state['epochs_no_improve'] = epochs_no_improve
            xm.save(checkpoint_state, "checkpoint.pth")
            xm.master_print(f"New best loss achieved! Saved best_model.pth and checkpoint.pth")
        else:
            epochs_no_improve += 1
            checkpoint_state['epochs_no_improve'] = epochs_no_improve
            xm.save(checkpoint_state, "checkpoint.pth")
            xm.master_print(f"No improvement. Early stopping patience: {epochs_no_improve}/{PATIENCE}. Saved checkpoint.pth")
            
        import sys
        sys.stdout.flush()
            
        if epochs_no_improve >= PATIENCE:
            xm.master_print(f"Early stopping triggered after {epoch} epochs!")
            break
            
    xm.master_print("Training Pipeline Complete.")

if __name__ == "__main__":
    # Pre-download ConvNeXt weights on the master thread to prevent 8-core race conditions
    print("Pre-downloading ConvNeXt weights to avoid XLA multiprocessing race conditions...")
    _ = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
    
    # Kaggle injects legacy TPU environment variables that violently conflict with modern PyTorch XLA PJRT
    if 'TPU_PROCESS_ADDRESSES' in os.environ:
        del os.environ['TPU_PROCESS_ADDRESSES']
    if 'TPU_NAME' in os.environ:
        del os.environ['TPU_NAME']
        
    flags = {}
    # Use nprocs=None (default) so PJRT automatically detects all 8 TPU cores
    xmp.spawn(_mp_fn, args=(flags,), nprocs=None, start_method='fork')
