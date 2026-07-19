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
import torch.multiprocessing as mp

# Bypass Google Colab's strict 64MB /dev/shm limit by forcing PyTorch to use the disk (/tmp) for IPC
mp.set_sharing_strategy('file_system')

# Prevent OpenCV from spawning thousands of threads and crashing Kaggle's DataLoader
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

from pytorch_metric_learning import losses, miners

import logging
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bruteforce search for the dataset anywhere on Kaggle
TTF_DIR = "ttf_files"
found = False
for search_dir in ["/kaggle/input", "/kaggle/working"]:
    if not os.path.exists(search_dir):
        continue
    for root, dirs, files in os.walk(search_dir, followlinks=True):
        if "ttf_files_2" in root:
            continue
        if any(f.endswith('.ttf') for f in files):
            TTF_DIR = root
            found = True
            break
    if found:
        break
        
logger.info(f"Auto-detected Kaggle dataset at: {TTF_DIR}")
BATCH_SIZE = 512
M_PER_CLASS = 4
EMBEDDING_SIZE = 512
VIRTUAL_EPOCH_BATCHES = 1250
MAX_EPOCHS = 50
LEARNING_RATE_BACKBONE = 2e-5
LEARNING_RATE_HEAD = 5e-4
PATIENCE = 15

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
        self.ttf_files = list(Path(ttf_dir).rglob("*.ttf")) + list(Path(ttf_dir).rglob("*.otf"))
        if len(self.ttf_files) == 0:
            raise ValueError(f"No font files found in {ttf_dir}")
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
            text = self.generate_random_string(random.randint(4, 10))
            font_size = random.randint(40, 80)
            font = ImageFont.truetype(str(ttf_path), font_size)
            
            # Tightly crop bounding box first
            temp_image = Image.new("RGB", (1, 1), "white")
            temp_draw = ImageDraw.Draw(temp_image)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_w = max(1, bbox[2] - bbox[0])
            text_h = max(1, bbox[3] - bbox[1])
            
            # Render exactly around the text
            image = Image.new("RGB", (text_w, text_h), "white")
            draw = ImageDraw.Draw(image)
            draw.text((-bbox[0], -bbox[1]), text, font=font, fill="black")
            
            # Resize and pad into a strictly 64x256 canvas
            target_w, target_h = 256, 64
            scale = min(target_w / text_w, target_h / text_h)
            new_w = max(1, int(text_w * scale))
            new_h = max(1, int(text_h * scale))
            
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            
            # Pad to 64x256
            final_image = Image.new("RGB", (target_w, target_h), "white")
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            final_image.paste(image, (x, y))
            
        except Exception:
            # If rendering fails, recursively pull a different random font sample
            return self.__getitem__(random.randint(0, len(self.ttf_files) - 1))
            
        image_np = np.array(final_image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, idx

# ==========================================
# 2. STATE-OF-THE-ART BACKBONE & HE BLOCK
# ==========================================

class HE_Block(nn.Module):
    """
    Hide & Enhance (HE) Block.
    Actively hides the maximum responding features during training to prevent feature collapse
    and force the network to explore complicated micro-features instead of global styles.
    """
    def __init__(self, drop_ratio=0.15):
        super(HE_Block, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if not self.training:
            return x
            
        batch_size, channels = x.size()
        k = max(1, int(channels * self.drop_ratio))
        
        # Find indices of top k responses
        _, topk_indices = torch.topk(x, k, dim=1)
        
        # Create a mask and zero out the top features
        mask = torch.ones_like(x)
        mask.scatter_(1, topk_indices, 0.0)
        
        return x * mask

class ConvNeXtFontEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ConvNeXtFontEncoder, self).__init__()
        # Load ConvNeXt-Tiny as a pure feature extractor
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        
        # HE Block to prevent feature collapse
        self.he_block = HE_Block(drop_ratio=0.15)
        
        # Custom projection head for Deep Metric Learning
        self.fc = nn.Linear(num_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = self.he_block(features)
        embeddings = self.fc(features)
        # Strict L2 Normalization MUST be in float32 to prevent float16 overflow/underflow
        return F.normalize(embeddings.float(), p=2, dim=1)

def train():
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_tpu = True
        logger.info(f"Targeting Google TPU Device: {device}")
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_tpu = False
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
    
    model = ConvNeXtFontEncoder(embedding_dim=EMBEDDING_SIZE)
    model = model.to(device)
    
    miner = miners.BatchHardMiner()
    
    # ==========================================
    # 3. CROSS-BATCH MEMORY LOGIC
    # ==========================================
    base_loss_function = losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)
    loss_func = losses.CrossBatchMemory(
        loss=base_loss_function, 
        embedding_size=EMBEDDING_SIZE, 
        memory_size=32768,
        miner=miner
    ).to(device)
    
    # Differential Learning Rates
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': LEARNING_RATE_BACKBONE},
        {'params': model.he_block.parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE_HEAD},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(script_dir, "checkpoint.pth")
    MODEL_PATH = os.path.join(script_dir, "best_model.pth")
    
    # Check for checkpoint or model weights to resume
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Found checkpoint file '{CHECKPOINT_PATH}'. Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # TPU checkpoints do not contain a scaler_state_dict because they use native bfloat16.
        # Only load the scaler state if it actually exists in the checkpoint to prevent a KeyError!
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        logger.info(f"Resumed from epoch {checkpoint['epoch']} with best loss {best_loss:.4f}.")
    elif os.path.exists(MODEL_PATH):
        logger.info(f"Checkpoint not found. Loading model weights from '{MODEL_PATH}' to continue training...")
        # Since architecture changed (HE_block added, fc shape changed to 512), strict=False is recommended
        # or we might fail to load. We will use strict=False to gracefully load the backbone at least.
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Successfully loaded backbone weights. Starting training from Epoch 1.")
        
    # CRITICAL: Wrap the model in DataParallel AFTER loading the checkpoint!
    # If we wrap it before loading, PyTorch will add 'module.' prefixes to the state_dict keys,
    # causing TPU checkpoints (which don't have 'module.') to crash when loading!
    if torch.cuda.device_count() > 1 and not is_tpu:
        logger.info(f"Multi-GPU Detected! Engaging {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)
        
    logger.info("Starting Gold-Standard Dynamic Training Pipeline...")
    
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        active_triplets = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", total=VIRTUAL_EPOCH_BATCHES)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            
            if is_tpu:
                # XLA handles bfloat16 natively on TPUs without autocast or scaler
                embeddings = model(images)
                loss = loss_func(embeddings.float(), labels)
                
                if torch.isnan(loss):
                    logger.error("Loss is NaN! Stopping training immediately to prevent checkpoint corruption.")
                    return
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                xm.optimizer_step(optimizer)
            else:
                with torch.autocast(device_type=device.type, enabled=True):
                    embeddings = model(images)
                    
                # Compute loss in float32 to prevent float16 exponential overflow
                loss = loss_func(embeddings.float(), labels)
                
                if torch.isnan(loss):
                    logger.error("Loss is NaN! Stopping training immediately to prevent checkpoint corruption.")
                    return # Exit the train loop entirely
                    
                scaler.scale(loss).backward()
                
                # Unscale gradients and clip to prevent exploding gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            
            running_loss += loss.item()
            active_triplets += miner.num_triplets
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'triplets': miner.num_triplets})
        
        avg_loss = running_loss / VIRTUAL_EPOCH_BATCHES
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"=== Epoch {epoch} Summary ===")
        logger.info(f"Average Loss: {avg_loss:.4f} | Total Hard Triplets: {active_triplets} | LR: {current_lr:.6f}")
        
        # Save complete training state at the end of each epoch for safety
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'epochs_no_improve': epochs_no_improve
        }
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            
            checkpoint_state['best_loss'] = best_loss
            checkpoint_state['epochs_no_improve'] = epochs_no_improve
            torch.save(checkpoint_state, "checkpoint.pth")
            logger.info(f"New best loss achieved! Saved best_model.pth and checkpoint.pth")
        else:
            epochs_no_improve += 1
            checkpoint_state['epochs_no_improve'] = epochs_no_improve
            torch.save(checkpoint_state, "checkpoint.pth")
            logger.info(f"No improvement. Early stopping patience: {epochs_no_improve}/{PATIENCE}. Saved checkpoint.pth")
            
        # Regular milestone backup every 5 epochs
        if epoch % 1 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
            logger.info(f"Milestone backup saved: checkpoint_epoch_{epoch}.pth")
            
        if epochs_no_improve >= PATIENCE:
            logger.warning(f"Early stopping triggered after {epoch} epochs!")
            break
            
    logger.info("Training Pipeline Complete.")

if __name__ == "__main__":
    train()
